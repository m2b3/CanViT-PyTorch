"""HuggingFace Hub integration for CanViTForRGBReconstruction."""

import logging
from typing import Any, cast

from huggingface_hub import PyTorchModelHubMixin

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.pretrain_common import (
    pretrain_repo_id_stem,
    upload_pretrain_to_hf,
)

from ..impl import CanViTForRGBReconstruction

log = logging.getLogger(__name__)


def make_rgb_repo_id(
    *, owner: str, backbone_name: str, glimpse_size_px: int, scene_size_px: int,
    dataset: str, enable_vpe: bool = True, canvas_update_mode: str = "additive",
) -> str:
    """HF Hub repo ID for an RGB-reconstruction checkpoint. Teacher-free: ``-rgbrecon`` suffix."""
    stem = pretrain_repo_id_stem(
        owner=owner, backbone_name=backbone_name, glimpse_size_px=glimpse_size_px,
        scene_size_px=scene_size_px, dataset=dataset, enable_vpe=enable_vpe,
        canvas_update_mode=canvas_update_mode,
    )
    return f"{stem}-rgbrecon"


def upload_to_hf(
    model: CanViTForRGBReconstruction,
    repo_id: str,
    *,
    private: bool = True,
    extra_metadata: dict | None = None,
) -> str:
    """Upload an RGB-reconstruction model. No standardizer grids to serialize (teacher-free)."""
    assert isinstance(model, CanViTForRGBReconstruction)
    return upload_pretrain_to_hf(model, repo_id, private=private, extra_metadata=extra_metadata)


def push_to_hf_hub(
    model: CanViTForRGBReconstruction,
    *,
    owner: str,
    dataset: str,
    glimpse_size_px: int,
    scene_size_px: int,
    private: bool = True,
) -> str:
    """Push with an auto-generated repo_id from metadata. Returns repo_id."""
    repo_id = make_rgb_repo_id(
        owner=owner,
        backbone_name=model.backbone_name,
        glimpse_size_px=glimpse_size_px,
        scene_size_px=scene_size_px,
        dataset=dataset,
        enable_vpe=model.cfg.enable_vpe,
        canvas_update_mode=model.cfg.canvas_update_mode,
    )
    return upload_to_hf(model, repo_id, private=private)


class CanViTForRGBReconstructionHFHub(
    CanViTForRGBReconstruction,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """CanViTForRGBReconstruction with HuggingFace Hub integration.

    Usage::

        model = CanViTForRGBReconstructionHFHub.from_pretrained("<org>/canvitb16-...-rgbrecon")
    """

    def __init__(
        self,
        backbone_name: str,
        model_config: dict[str, Any],
    ):
        cfg = CanViTConfig(**{k: v for k, v in model_config.items() if k in CanViTConfig.__dataclass_fields__})
        super().__init__(
            backbone=create_backbone(cast(BackboneName, backbone_name)),
            cfg=cfg,
            backbone_name=backbone_name,
        )
