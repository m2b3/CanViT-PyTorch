"""CanViT demo: canvas PCA visualization across two viewing trajectories.

Two trajectories — overview-first and details-first — process the same scene
through different sequences of glimpses. The canvas PCA at each timestep shows
how CanViT integrates information regardless of viewing order.

Usage:
    uv run --extra demo python demos/basic.py
    uv run --extra demo python demos/basic.py --image path/to/image.jpg
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

from canvit_pytorch import CanViTForPretrainingHFHub, RecurrentState, Viewpoint, sample_at_viewpoint
from canvit_pytorch.checkpoints import FLAGSHIP_PRETRAIN_REPO
from canvit_pytorch.preprocess import imagenet_denormalize, preprocess


@dataclass
class Config:
    model_repo: str = FLAGSHIP_PRETRAIN_REPO
    image: Path = Path("test_data/Places365_IMG_9600.jpeg")
    canvas_grid: int = 32
    glimpse_px: int = 128
    output: Path = Path("outputs/demo.png")


class Step(NamedTuple):
    center_y: float
    center_x: float
    scale: float


# Coordinates: (0,0)=center, (-1,-1)=top-left, (1,1)=bottom-right, (row, col) order.
TRAJECTORIES: dict[str, list[Step]] = {
    "Overview → details": [
        Step(0.0, 0.0, 1.0),
        Step(-0.3, -0.3, 0.35),
        Step(-0.1, 0.15, 0.35),
        Step(0.2, 0.35, 0.3),
    ],
    "Details → overview": [
        Step(0.2, 0.35, 0.3),
        Step(0.3, -0.1, 0.35),
        Step(-0.3, -0.3, 0.35),
        Step(0.0, 0.0, 1.0),
    ],
}

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]


# ── Inference ─────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_trajectory(
    model: CanViTForPretrainingHFHub,
    image: torch.Tensor,
    steps: list[Step],
    *,
    canvas_grid: int,
    glimpse_px: int,
) -> tuple[list[torch.Tensor], list[np.ndarray], RecurrentState]:
    """Run a trajectory, returning per-step (canvas, glimpse_rgb) and final state."""
    state = model.init_state(batch_size=1, canvas_grid_size=canvas_grid)
    canvases: list[torch.Tensor] = []
    glimpses: list[np.ndarray] = []

    for step in steps:
        vp = Viewpoint(
            centers=torch.tensor([[step.center_y, step.center_x]], device=image.device),
            scales=torch.tensor([step.scale], device=image.device),
        )
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=glimpse_px)
        out = model(glimpse=glimpse, state=state, viewpoint=vp)
        state = out.state

        canvases.append(model.get_spatial(state.canvas).squeeze(0).cpu().float())
        glimpses.append(imagenet_denormalize(glimpse[0].cpu()).permute(1, 2, 0).clamp(0, 1).numpy())

    return canvases, glimpses, state


# ── PCA ───────────────────────────────────────────────────────────────────


def fit_shared_pca(canvases: list[torch.Tensor]) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit 3-component PCA on all canvases. Returns (pca, global_min, global_max)."""
    normed = [F.layer_norm(c, [c.shape[-1]]).numpy() for c in canvases]
    pca = PCA(n_components=3)
    pca.fit(np.concatenate(normed, axis=0))
    projected = np.concatenate([pca.transform(n) for n in normed], axis=0)
    return pca, projected.min(axis=0), projected.max(axis=0)


def canvas_to_rgb(canvas: torch.Tensor, pca: PCA, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Project canvas tokens to RGB via shared PCA."""
    n_tokens, D = canvas.shape
    g = int(np.sqrt(n_tokens))
    assert g * g == n_tokens
    proj = pca.transform(F.layer_norm(canvas, [D]).numpy())
    return np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1).reshape(g, g, 3)


# ── Plotting ──────────────────────────────────────────────────────────────


def viewpoint_to_rect(step: Step, img_h: int, img_w: int) -> tuple[float, float, float, float]:
    """Convert viewpoint to matplotlib (x, y, width, height) in pixel coords."""
    y0 = (step.center_y - step.scale + 1) / 2 * img_h
    y1 = (step.center_y + step.scale + 1) / 2 * img_h
    x0 = (step.center_x - step.scale + 1) / 2 * img_w
    x1 = (step.center_x + step.scale + 1) / 2 * img_w
    return x0, y0, x1 - x0, y1 - y0


def plot(
    img_rgb: np.ndarray,
    trajectories: dict[str, tuple[list[Step], list[torch.Tensor], list[np.ndarray]]],
    pca: PCA,
    lo: np.ndarray,
    hi: np.ndarray,
    output: Path,
) -> None:
    traj_names = list(trajectories.keys())
    n_steps = len(next(iter(trajectories.values()))[0])
    n_traj = len(traj_names)
    img_h, img_w = img_rgb.shape[:2]

    fig = plt.figure(figsize=(3.2 * (n_steps + 1), 5.5 * n_traj))
    outer = gridspec.GridSpec(n_traj, 1, figure=fig, hspace=0.22)

    for t_idx, name in enumerate(traj_names):
        steps, canvases, glimpses = trajectories[name]
        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_steps + 1, subplot_spec=outer[t_idx],
            hspace=0.06, wspace=0.08, width_ratios=[1.4] + [1.0] * n_steps,
        )

        # Left panel: source image with viewpoint rectangles
        ax_src = fig.add_subplot(inner[:, 0])
        ax_src.imshow(img_rgb)
        for i, step in enumerate(steps):
            color = COLORS[i % len(COLORS)]
            x, y, w, h = viewpoint_to_rect(step, img_h, img_w)
            ax_src.add_patch(mpatches.FancyBboxPatch(
                (x, y), w, h, linewidth=2.5, edgecolor=color,
                facecolor="none", boxstyle="round,pad=0",
            ))
            ax_src.text(
                x + 3, y + 14, f"t={i}", color=color, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0),
            )
        ax_src.set_title(name, fontsize=10, fontweight="bold")
        ax_src.axis("off")

        # Right panels: glimpses (top row) + canvas PCA (bottom row)
        for i in range(n_steps):
            color = COLORS[i % len(COLORS)]

            ax_g = fig.add_subplot(inner[0, i + 1])
            ax_g.imshow(glimpses[i])
            ax_g.set_title(f"t={i}", fontsize=9, color=color, fontweight="bold")
            if i == 0:
                ax_g.set_ylabel("glimpse", fontsize=9, color="gray")
            for sp in ax_g.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)
            ax_g.set_xticks([])
            ax_g.set_yticks([])

            ax_c = fig.add_subplot(inner[1, i + 1])
            ax_c.imshow(canvas_to_rgb(canvases[i], pca, lo, hi), interpolation="nearest")
            if i == 0:
                ax_c.set_ylabel("canvas", fontsize=9, color="gray")
            for sp in ax_c.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2.5)
            ax_c.set_xticks([])
            ax_c.set_yticks([])

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


# ── Main ──────────────────────────────────────────────────────────────────


def main(cfg: Config) -> None:
    print(f"Loading {cfg.model_repo}...")
    model = CanViTForPretrainingHFHub.from_pretrained(cfg.model_repo).eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    img_size = cfg.canvas_grid * model.backbone.patch_size_px
    image = preprocess(img_size)(Image.open(cfg.image).convert("RGB"))
    assert isinstance(image, torch.Tensor)
    image = image.unsqueeze(0)
    img_rgb = imagenet_denormalize(image[0]).permute(1, 2, 0).clamp(0, 1).numpy()

    # Run trajectories
    results: dict[str, tuple[list[Step], list[torch.Tensor], list[np.ndarray]]] = {}
    for name, steps in TRAJECTORIES.items():
        canvases, glimpses, _ = run_trajectory(
            model, image, steps, canvas_grid=cfg.canvas_grid, glimpse_px=cfg.glimpse_px,
        )
        results[name] = (steps, canvases, glimpses)
        print(f"  {name}: {len(steps)} steps")

    # Shared PCA across all canvases
    all_canvases = [c for _, canvases, _ in results.values() for c in canvases]
    pca, lo, hi = fit_shared_pca(all_canvases)

    plot(img_rgb, results, pca, lo, hi, cfg.output)


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Config))
