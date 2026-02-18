"""Self-contained ViT backbone for CanViT."""

import logging
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from canvit_pytorch.rope import RoPE, rope_apply_with_prefix

log = logging.getLogger(__name__)


class NormFeatures(NamedTuple):
    patches: Tensor  # [B, H*W, D]
    cls: Tensor  # [B, D]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)  # [B, D, H, W]
        H, W = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), H, W  # [B, H*W, D]


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        mask = torch.ones(dim * 3)
        mask[dim : 2 * dim] = 0
        self.register_buffer("_bias_mask", mask, persistent=False)
        self.proj = nn.Linear(dim, dim)

    _bias_mask: Tensor

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        B, N, D = x.shape
        assert self.qkv.bias is not None
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self._bias_mask).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]  # [B, H, N, D_h]
        q = rope_apply_with_prefix(x=q, rope=rope)
        k = rope_apply_with_prefix(x=k, rope=rope)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, layerscale_init: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, layerscale_init)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim, layerscale_init)

    def forward(self, x: Tensor, rope: RoPE) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ViTBackbone(nn.Module):
    """ViT backbone: patch embedding + transformer blocks."""

    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size_px: int
    ffn_ratio: float
    rope_base: float
    layerscale_init: float

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        n_blocks: int,
        patch_size: int,
        ffn_ratio: float,
        rope_base: float,
        layerscale_init: float,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_blocks = n_blocks
        self.patch_size_px = patch_size
        self.ffn_ratio = ffn_ratio
        self.rope_base = rope_base
        self.layerscale_init = layerscale_init

        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, ffn_ratio, layerscale_init) for _ in range(n_blocks)
        ])
        self._init_weights()

    def _init_weights(self) -> None:
        """Match DINOv3 init: trunc_normal_(std=0.02) for Linear weights, zeros for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerScale):
                nn.init.constant_(module.gamma, self.layerscale_init)
            elif isinstance(module, PatchEmbed):
                proj = module.proj
                k = 1.0 / (proj.in_channels * proj.kernel_size[0] * proj.kernel_size[1])
                nn.init.uniform_(proj.weight, -math.sqrt(k), math.sqrt(k))
                if proj.bias is not None:
                    nn.init.uniform_(proj.bias, -math.sqrt(k), math.sqrt(k))

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads
