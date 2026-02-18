"""Viewpoint Positional Encoding (VPE) via Random Fourier Features.

COORDINATE CONVENTIONS (matches canvit.coords, canvit.viewpoint):
- y, x: spatiotopic scene coordinates in [-1, 1] (scene center at origin)
- s: scale = FOV_width / scene_width
  - s = 1: full scene visible (max zoom out)
  - s < 1: zoomed in (seeing fraction s of scene)
  - s ∈ (0, 1]

TRANSFORM:
  (y, x, s) → (y/s, x/s, log(s)) → RFF → LayerNorm

y/s, x/s: position in FOV-relative units
"""

import math

import torch
from torch import Tensor, nn


class VPEEncoder(nn.Module):
    """Viewpoint Positional Encoding using Random Fourier Features.

    Encodes (y, x, s) viewpoint into a high-dimensional vector.
    Uses (y, x) order to match canvit.coords and canvit.viewpoint conventions.
    The RFF projection is not trainable.
    """

    B: Tensor  # RFF projection matrix, frozen

    def __init__(self, rff_dim: int, seed: int = 42) -> None:
        """Args:
            rff_dim: Random Fourier Features output dimension (must be positive and even)
            seed: RNG seed for reproducible RFF projection matrix
        """
        super().__init__()
        assert rff_dim > 0 and rff_dim % 2 == 0, "rff_dim must be positive and even"
        self.rff_dim = rff_dim

        generator = torch.Generator().manual_seed(seed)
        B = torch.randn(rff_dim // 2, 3, generator=generator, dtype=torch.float32)
        self.register_buffer("B", B)

        self.norm = nn.LayerNorm(rff_dim)
        self.norm.weight.data.fill_(1.0 / math.sqrt(rff_dim))

    @property
    def output_dim(self) -> int:
        return self.rff_dim

    def forward(self, *, y: Tensor, x: Tensor, s: Tensor) -> Tensor:
        """Encode viewpoints to VPE vectors. Always fp32 (one token).

        Args:
            y, x: [B] or [...] scene position in [-1, 1], must be float32
            s: [B] or [...] scale = FOV/scene (s=1 full scene, s<1 zoomed in), must be float32

        Returns:
            Encoding of shape [..., rff_dim], always float32
        """
        assert y.dtype == torch.float32, f"y must be float32, got {y.dtype}"
        assert x.dtype == torch.float32, f"x must be float32, got {x.dtype}"
        assert s.dtype == torch.float32, f"s must be float32, got {s.dtype}"

        with torch.autocast(device_type=y.device.type, enabled=False):
            z = torch.stack([y / s, x / s, torch.log(s)], dim=-1)
            proj = z @ self.B.T
            enc = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            return self.norm(enc)


__all__ = ["VPEEncoder"]
