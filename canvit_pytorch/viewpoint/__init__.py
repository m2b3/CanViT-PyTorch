"""Glimpse extraction via viewpoint sampling.

Coordinate convention (shared with canvit.coords):
- Coordinates are in [-1, 1]²
- (0, 0) is image center, (-1, -1) is top-left, (1, 1) is bottom-right
- Axis order is (row, col) = (y, x) - matrix indexing convention
  (tensor[..., H, W], meshgrid "ij", imshow). NOT Cartesian (x, y).
- centers specify where the glimpse is centered in scene space
- scales specify the glimpse zoom relative to scene (1 = full scene, 0 = infinitely small)
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from canvit_pytorch.coords import grid_coords


@dataclass
class Viewpoint:
    """A viewpoint specifying where to look in an image.

    Attributes:
        centers: [B, 2] in [-1, 1], (row, col) order — matches tensor indexing,
                 NOT Cartesian (x, y). See module docstring for rationale.
        scales: [B] in (0, 1]
    """

    centers: Tensor
    scales: Tensor

    @staticmethod
    def full_scene(*, batch_size: int, device: torch.device) -> "Viewpoint":
        """Full scene viewpoint: center=(0,0), scale=1.."""
        return Viewpoint(
            centers=torch.zeros(batch_size, 2, device=device, dtype=torch.float32),
            scales=torch.ones(batch_size, device=device, dtype=torch.float32),
        )


def sample_at_viewpoint(
    *,
    spatial: Tensor,
    viewpoint: Viewpoint,
    glimpse_size_px: int,
) -> Tensor:
    """Sample from spatial tensor at viewpoint positions.

    Works for images [B, C, H, W] or latent feature maps [B, D, G, G].

    Args:
        spatial: [B, C, H, W] spatial tensor
        viewpoint: Viewpoint with centers [B, 2] and scales [B], must be float32
        glimpse_size_px: output spatial size in pixels (glimpse_size_px × glimpse_size_px)

    Returns:
        [B, C, glimpse_size_px, glimpse_size_px] bilinearly sampled crop (same dtype as spatial)
    """
    assert viewpoint.centers.dtype == torch.float32, \
        f"viewpoint.centers must be float32, got {viewpoint.centers.dtype}"
    assert viewpoint.scales.dtype == torch.float32, \
        f"viewpoint.scales must be float32, got {viewpoint.scales.dtype}"

    B = viewpoint.centers.shape[0]
    device = spatial.device
    out_dtype = spatial.dtype

    # Grid offsets in [-1, 1] for the output grid (float32)
    offsets = grid_coords(H=glimpse_size_px, W=glimpse_size_px, device=device).unsqueeze(0)

    centers = viewpoint.centers.view(B, 1, 1, 2)
    scales = viewpoint.scales.view(B, 1, 1, 1)
    grid = centers + scales * offsets  # [B, H, W, 2], float32

    # grid_sample expects (x, y), our coords are (y, x)
    grid = grid.flip(-1)

    # grid_sample in float32, cast result back to spatial dtype
    result = F.grid_sample(spatial.float(), grid, mode="bilinear", align_corners=False)
    return result.to(out_dtype)


__all__ = ["Viewpoint", "sample_at_viewpoint"]
