from typing import override

import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.rope import RoPE, rope_apply_with_prefix


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    """[B, N, D] -> [B, H, N, head_dim]."""
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def from_multihead(x: Tensor) -> Tensor:
    """[B, H, N, head_dim] -> [B, N, D]."""
    B, H, N, hd = x.shape
    return x.transpose(1, 2).reshape(B, N, H * hd)


class CanvasAttention(nn.Module):
    """Base class for asymmetric canvas cross-attention.

    Subclasses (CanvasReadAttention, CanvasWriteAttention) configure which
    transforms are dense (Linear) vs Identity.

    All attention happens in canvas_dim space. The output is projected to out_dim.
    """

    def __init__(
        self,
        *,
        q_in_dim: int,
        kv_in_dim: int,
        canvas_dim: int,
        out_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert canvas_dim % num_heads == 0
        self.canvas_dim: int = canvas_dim
        self.out_dim: int = out_dim
        self.num_heads: int = num_heads

        # Overridden by subclasses
        self.q_proj: nn.Module = nn.Identity()
        self.k_proj: nn.Module = nn.Identity()
        self.v_proj: nn.Module = nn.Identity()
        self.out_proj: nn.Module = nn.Identity()

        self.q_norm = nn.LayerNorm(q_in_dim)
        self.kv_norm = nn.LayerNorm(kv_in_dim)

    @override
    def forward(
        self,
        *,
        query: Tensor,
        kv: Tensor,
        query_rope: RoPE,
        kv_rope: RoPE,
    ) -> Tensor:
        """Cross-attention from query to kv.

        Args:
            query: [B, N_q, q_in_dim]
            kv: [B, N_kv, kv_in_dim] - source for keys and values
            query_rope: RoPE for query positions
            kv_rope: RoPE for key positions

        Returns:
            [B, N_q, out_dim]
        """
        q: Tensor = to_multihead(self.q_proj(self.q_norm(query)), self.num_heads)
        kv_normed = self.kv_norm(kv)
        k: Tensor = to_multihead(self.k_proj(kv_normed), self.num_heads)
        v: Tensor = to_multihead(self.v_proj(kv_normed), self.num_heads)

        q = rope_apply_with_prefix(x=q, rope=query_rope)
        k = rope_apply_with_prefix(x=k, rope=kv_rope)

        out: Tensor = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(from_multihead(out))
