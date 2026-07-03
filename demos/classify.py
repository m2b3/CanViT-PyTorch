"""CanViT classification demo: classify an image using sequential glimpses.

Shows both construction paths for CanViTForImageClassification:
  1. Finetuned model checkpoint
  2. Frozen pretrained CanViT checkpoint + fused DINOv3 probe

Both produce the same forward API. The model processes glimpses
sequentially, refining its prediction with each new observation.

Usage:
    uv run --extra demo python demos/classify.py
    uv run --extra demo python demos/classify.py --image path/to/image.jpg
    uv run --extra demo python demos/classify.py --mode frozen
"""

import argparse
import logging
from pathlib import Path
from typing import Literal

import torch
from timm.data.imagenet_info import ImageNetInfo
from PIL import Image

from canvit_pytorch import CanViTForImageClassification, Viewpoint, resolve_canvit_repo, sample_at_viewpoint
from canvit_pytorch.checkpoints import FLAGSHIP_PRETRAIN_REPO
from canvit_pytorch.preprocess import preprocess

log = logging.getLogger("classify")

FINETUNED_REPO = resolve_canvit_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06")
PRETRAINED_REPO = FLAGSHIP_PRETRAIN_REPO
PROBE_REPO = resolve_canvit_repo("dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe")

CANVAS_GRID = 32
GLIMPSE_PX = 128

# Coarse-to-Fine: full scene → quadrants
C2F_VIEWPOINTS = [
    (0.0, 0.0, 1.0),     # full scene
    (-0.5, -0.5, 0.5),   # top-left
    (-0.5, 0.5, 0.5),    # top-right
    (0.5, -0.5, 0.5),    # bottom-left
    (0.5, 0.5, 0.5),     # bottom-right
]


def load_classifier(mode: Literal["finetuned", "frozen"]) -> CanViTForImageClassification:
    if mode == "finetuned":
        return CanViTForImageClassification.from_pretrained(FINETUNED_REPO).eval()
    return CanViTForImageClassification.from_pretrained_with_probe(
        pretrained_repo=PRETRAINED_REPO, probe_repo=PROBE_REPO,
    ).eval()


def classify(clf: CanViTForImageClassification, image: torch.Tensor) -> None:
    """Run C2F glimpses and print per-step top-1 prediction."""
    ini = ImageNetInfo()
    state = clf.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)

    with torch.inference_mode():
        for t, (cy, cx, s) in enumerate(C2F_VIEWPOINTS):
            vp = Viewpoint(
                centers=torch.tensor([[cy, cx]], device=image.device),
                scales=torch.tensor([s], device=image.device),
            )
            glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=GLIMPSE_PX)
            logits, state = clf(glimpse=glimpse, state=state, viewpoint=vp)

            probs = torch.softmax(logits, dim=-1)
            top = probs[0].topk(1)
            label = ini.index_to_description(int(top.indices[0].item())).split(",")[0]
            conf = top.values[0].item()

            region = "full scene" if s == 1.0 else f"({cy:+.1f}, {cx:+.1f}) scale={s}"
            log.info("t=%d [%s] → %s (%.1f%%)", t, region, label, conf * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=Path("test_data/Cat03.jpg"))
    parser.add_argument("--mode", choices=["finetuned", "frozen"], default="finetuned",
                        help="'finetuned' uses the finetuned checkpoint, 'frozen' uses a pretrained CanViT checkpoint with a fused DINOv3 probe")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")

    log.info("Loading classifier (mode=%s)...", args.mode)
    clf = load_classifier(args.mode)
    log.info("Loaded: %d classes, %d-dim CanViT local stream", clf.n_classes, clf.local_dim)

    img_size = CANVAS_GRID * clf.canvit.backbone.patch_size_px
    image = preprocess(img_size)(Image.open(args.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0)
    log.info("Image: %s → %s", args.image, list(image.shape))

    log.info("Classifying with %d C2F glimpses...", len(C2F_VIEWPOINTS))
    classify(clf, image)


if __name__ == "__main__":
    main()
