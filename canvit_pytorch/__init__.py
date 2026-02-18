"""CanViT: Dual-stream vision transformer with canvas cross-attention."""

from canvit_pytorch.backbone import BackboneName, ViTBackbone, create_backbone
from canvit_pytorch.model import (
    CanViT,
    CanViTConfig,
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    CanViTForPretrainingHFHub,
    CanViTOutput,
    RecurrentState,
)
from canvit_pytorch.norm import CLSStandardizer, PatchStandardizer, PositionAwareStandardizer
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.vpe import VPEEncoder

__all__ = [
    "BackboneName",
    "CLSStandardizer",
    "CanViT",
    "CanViTConfig",
    "CanViTForPretraining",
    "CanViTForPretrainingConfig",
    "CanViTForPretrainingHFHub",
    "CanViTOutput",
    "PatchStandardizer",
    "PositionAwareStandardizer",
    "RecurrentState",
    "VPEEncoder",
    "Viewpoint",
    "ViTBackbone",
    "create_backbone",
    "sample_at_viewpoint",
]
