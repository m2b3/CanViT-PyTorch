#!/usr/bin/env python3
"""CanViT demo: load pretrained model, run inference, visualize canvas features via PCA.

Usage:
    uv run python demos/basic.py
    uv run python demos/basic.py --image path/to/image.jpg
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

import timm
from dinov3_in1k_probes import DINOv3LinearClassificationHead

from canvit_pytorch import CanViTForPretrainingHFHub, Viewpoint, sample_at_viewpoint
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess


@dataclass
class Config:
    model_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
    probe_repo: str = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
    image: Path = Path("test_data/Cat03.jpg")
    canvas_grid: int = 32
    glimpse_px: int = 128
    output: Path = Path("outputs/demo_pca.png")


def spatial_pca_to_rgb(spatial: torch.Tensor) -> np.ndarray:
    """[G², D] spatial tokens → [G, G, 3] float32 RGB in [0, 1] via PCA."""
    n_tokens, D = spatial.shape
    g = int(np.sqrt(n_tokens))
    assert g * g == n_tokens
    normed = F.layer_norm(spatial, [D]).numpy()
    proj = PCA(n_components=3).fit_transform(normed)
    lo = proj.min(axis=0, keepdims=True)
    hi = proj.max(axis=0, keepdims=True)
    return np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1).reshape(g, g, 3)


def main(cfg: Config) -> None:
    device = torch.device("cpu")

    # --- Load model ---
    print(f"Loading {cfg.model_repo}...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).to(device).eval()
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    cls_std, scene_std = model.standardizers(cfg.canvas_grid)
    assert cls_std.initialized and scene_std.initialized

    # --- Load probe ---
    print(f"Loading probe from {cfg.probe_repo}...")
    probe = DINOv3LinearClassificationHead.from_pretrained(cfg.probe_repo).to(device).eval()

    # --- Load and preprocess image ---
    print(f"Loading {cfg.image}...")
    patch_size = model.backbone.patch_size_px
    img_size = cfg.canvas_grid * patch_size
    pil_img = Image.open(cfg.image).convert("RGB")
    image = preprocess(img_size)(pil_img)
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0).to(device)

    # --- Forward pass ---
    print("Running inference...")
    state = model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    vp = Viewpoint.full_scene(batch_size=1, device=device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=cfg.glimpse_px)

    with torch.inference_mode():
        out = model(glimpse=glimpse, state=state, viewpoint=vp)

    print(f"  canvas: {out.state.canvas.shape}")
    print(f"  recurrent_cls: {out.state.recurrent_cls.shape}")

    # --- Classification ---
    print("\nClassification:")
    with torch.inference_mode():
        cls_pred_std = model.predict_scene_teacher_cls(out.state.recurrent_cls)
        cls_raw = cls_std.destandardize(cls_pred_std.unsqueeze(1)).squeeze(1)
        logits = probe(cls_raw)
        probs = F.softmax(logits, dim=-1)

    ini = timm.data.ImageNetInfo()  # pyright: ignore[reportAttributeAccessIssue]
    topk = logits.topk(5)
    for i, (idx, prob) in enumerate(zip(topk.indices[0], probs[0, topk.indices[0]]), 1):
        label = ini.index_to_description(idx.item())
        print(f"  {i}. {label:40s} {prob.item() * 100:5.2f}%")

    # --- PCA visualization ---
    print("Generating PCA visualization...")
    canvas_spatial = model.get_spatial(out.state.canvas).squeeze(0).cpu().float()
    canvas_pca = spatial_pca_to_rgb(canvas_spatial)

    with torch.inference_mode():
        scene_pred = model.predict_teacher_scene(out.state.canvas)
    scene_pca = spatial_pca_to_rgb(scene_pred.squeeze(0).cpu().float())

    img_display = imagenet_denormalize(image[0].cpu()).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_display)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(canvas_pca)
    axes[1].set_title("Canvas PCA")
    axes[1].axis("off")
    axes[2].imshow(scene_pca)
    axes[2].set_title("Scene Prediction PCA")
    axes[2].axis("off")
    plt.tight_layout()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.output, dpi=150)
    print(f"  saved: {cfg.output}")


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
