"""Backbone factory: create ViT backbones by name."""

import logging
from dataclasses import dataclass
from typing import Literal

from canvit_pytorch.backbone.vit import ViTBackbone

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackboneConfig:
    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size: int = 16
    ffn_ratio: float = 4.0
    rope_base: float = 100.0
    layerscale_init: float = 1e-5


BackboneName = Literal[
    "vits16",
    "vitb16",
    "vitl16",
]

REGISTRY: dict[str, BackboneConfig] = {
    "vits16": BackboneConfig(embed_dim=384, num_heads=6, n_blocks=12),
    "vitb16": BackboneConfig(embed_dim=768, num_heads=12, n_blocks=12),
    "vitl16": BackboneConfig(embed_dim=1024, num_heads=16, n_blocks=24),
}


def create_backbone(name: str) -> ViTBackbone:
    """Create a ViT backbone by name (random weights)."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown backbone: {name!r}. Available: {available}")
    cfg = REGISTRY[name]
    backbone = ViTBackbone(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        n_blocks=cfg.n_blocks,
        patch_size=cfg.patch_size,
        ffn_ratio=cfg.ffn_ratio,
        rope_base=cfg.rope_base,
        layerscale_init=cfg.layerscale_init,
    )
    log.info(
        "Created %s: %d blocks, embed_dim=%d",
        name, cfg.n_blocks, cfg.embed_dim,
    )
    return backbone
