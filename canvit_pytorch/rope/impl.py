import math
from typing import NamedTuple

import torch
from torch import Tensor
from canvit_pytorch.correctness import assert_shape


class RoPE(NamedTuple):
    sin: Tensor  # [B, 1, N, head_dim]
    cos: Tensor  # [B, 1, N, head_dim]


def head_dim_from_periods(periods: Tensor) -> int:
    """Inverse of make_rope_periods: recover head_dim from periods tensor."""
    return periods.shape[0] * 4


def make_rope_periods(
    *,
    head_dim: int,
    base: float,
    device: torch.device | None = None,
) -> Tensor:
    """Create frequency periods for 2D RoPE. Always float32.

    Returns geometric progression of wavelengths [1, base].
    base = 100.0 is fine for [-1, +1].
    """
    assert head_dim % 4 == 0, (
        f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
    )
    n_freqs = head_dim // 4
    exponents = torch.arange(n_freqs, device=device, dtype=torch.float32) / n_freqs
    out = base**exponents
    assert_shape(out, (n_freqs,))
    return out


def compute_rope(*, positions: Tensor, periods: Tensor) -> RoPE:
    """Compute RoPE from 2D positions. All computation and output in float32.

    Args:
        positions: [B, N, 2] spatial coordinates, must be float32
        periods: [n_freqs] from make_rope_periods, must be float32
    """
    B, N, _two = positions.shape
    assert_shape(positions, (B, N, 2))
    assert positions.dtype == torch.float32, f"positions must be float32, got {positions.dtype}"
    assert periods.dtype == torch.float32, f"periods must be float32, got {periods.dtype}"
    assert positions.device == periods.device
    head_dim = 4 * periods.shape[0]

    angles = 2 * math.pi * positions.unsqueeze(-1) / periods
    angles = angles.flatten(-2, -1).tile((2,))

    sin = torch.sin(angles)
    cos = torch.cos(angles)
    assert_shape(sin, (B, N, head_dim))
    assert_shape(cos, (B, N, head_dim))

    H = 1
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)
    assert_shape(sin, (B, H, N, head_dim))
    assert_shape(cos, (B, H, N, head_dim))
    return RoPE(sin, cos)


def rope_apply_with_prefix(*, x: Tensor, rope: RoPE) -> Tensor:
    """Apply RoPE rotation to spatial tokens, copying prefix tokens unchanged.

    Rotation is computed in float32 for precision, then cast back to x's dtype.

    Args:
        x: [B, H, N_total, head_dim]
        rope: precomputed sin/cos for N_spatial tokens

    n_prefix = N_total - N_spatial (inferred from shapes)
    """
    assert x.ndim == 4, f"Expected 4D [B, H, N, D], got {x.ndim}D"
    n_prefix = x.shape[2] - rope.sin.shape[2]
    assert n_prefix >= 0, f"RoPE has more tokens ({rope.sin.shape[2]}) than input ({x.shape[2]})"
    half = x.shape[3] // 2

    out = torch.empty_like(x)

    if n_prefix > 0:
        out[:, :, :n_prefix] = x[:, :, :n_prefix]

    assert rope.sin.dtype == torch.float32, f"RoPE must be float32, got {rope.sin.dtype}"
    x_s = x[:, :, n_prefix:].float()
    x_rot = torch.cat([-x_s[..., half:], x_s[..., :half]], dim=-1)
    out[:, :, n_prefix:] = (x_s * rope.cos + x_rot * rope.sin).to(x.dtype)

    return out
