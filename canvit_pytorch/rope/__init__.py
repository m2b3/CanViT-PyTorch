"""RoPE (Rotary Position Embeddings) for 2D spatial positions.

Our standard convention: positions live in [-1, +1]^2 and in scene coordinate space.
"""

from canvit_pytorch.rope.impl import (
    RoPE,
    compute_rope,
    head_dim_from_periods,
    make_rope_periods,
    rope_apply_with_prefix,
)

__all__ = [
    "RoPE",
    "compute_rope",
    "head_dim_from_periods",
    "make_rope_periods",
    "rope_apply_with_prefix",
]
