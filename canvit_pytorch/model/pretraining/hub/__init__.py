"""HuggingFace Hub integration for CanViTForPretraining."""

import logging
from typing import Any, cast

from huggingface_hub import PyTorchModelHubMixin

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.pretrain_common import (
    pretrain_repo_id_stem,
    upload_pretrain_to_hf,
)

from ..impl import CanViTForPretraining, CanViTForPretrainingConfig

log = logging.getLogger(__name__)

# Teacher full name → hub shortname
TEACHER_SHORT = {
    "dinov3_vits16": "dv3s16",
    "dinov3_vitb16": "dv3b16",
    "dinov3_vitl16": "dv3l16",
}


def teacher_shortname(teacher_name: str) -> str:
    assert teacher_name in TEACHER_SHORT, f"Unknown teacher {teacher_name!r}, known: {sorted(TEACHER_SHORT)}"
    return TEACHER_SHORT[teacher_name]


def make_repo_id(
    *, owner: str, backbone_name: str, glimpse_size_px: int, scene_size_px: int,
    dataset: str, teacher_name: str, enable_vpe: bool = True,
    canvas_update_mode: str = "additive",
) -> str:
    """Compute HF Hub repo ID from checkpoint metadata. Single source of truth."""
    stem = pretrain_repo_id_stem(
        owner=owner, backbone_name=backbone_name, glimpse_size_px=glimpse_size_px,
        scene_size_px=scene_size_px, dataset=dataset, enable_vpe=enable_vpe,
        canvas_update_mode=canvas_update_mode,
    )
    return f"{stem}-{teacher_shortname(teacher_name)}"


def upload_to_hf(
    model: CanViTForPretraining,
    repo_id: str,
    *,
    private: bool = True,
    extra_metadata: dict | None = None,
) -> str:
    """Upload model to HuggingFace Hub under the given repo_id. Returns repo_id.

    extra_metadata is merged into config.json alongside model_config,
    backbone_name, and canvas_patch_grid_sizes (the teacher standardizer grid sizes).
    """
    assert isinstance(model.cfg, CanViTForPretrainingConfig)
    return upload_pretrain_to_hf(
        model,
        repo_id,
        extra_config={"canvas_patch_grid_sizes": model.canvas_patch_grid_sizes},
        private=private,
        extra_metadata=extra_metadata,
    )


def push_to_hf_hub(
    model: CanViTForPretraining,
    *,
    owner: str,
    dataset: str,
    teacher_name: str,
    glimpse_size_px: int,
    private: bool = True,
) -> str:
    """Push model with auto-generated repo_id from metadata. Returns repo_id."""
    assert len(model.canvas_patch_grid_sizes) == 1, f"Expected single grid size, got {model.canvas_patch_grid_sizes}"

    grid_size = model.canvas_patch_grid_sizes[0]
    scene_size_px = grid_size * model.backbone.patch_size_px
    repo_id = make_repo_id(
        owner=owner,
        backbone_name=model.backbone_name,
        glimpse_size_px=glimpse_size_px,
        scene_size_px=scene_size_px,
        dataset=dataset,
        teacher_name=teacher_name,
        enable_vpe=model.cfg.enable_vpe,
        canvas_update_mode=model.cfg.canvas_update_mode,
    )
    return upload_to_hf(model, repo_id, private=private)


class CanViTForPretrainingHFHub(
    CanViTForPretraining,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """CanViTForPretraining with HuggingFace Hub integration.

    Usage:
        model = CanViTForPretrainingHFHub.from_pretrained("<org>/canvitb16-add-vpe-pretrain-...")
    """

    def __init__(
        self,
        backbone_name: str,
        model_config: dict[str, Any],
        canvas_patch_grid_sizes: list[int],
    ):
        super().__init__(
            backbone=create_backbone(cast(BackboneName, backbone_name)),
            cfg=CanViTForPretrainingConfig(**model_config),
            backbone_name=backbone_name,
            canvas_patch_grid_sizes=canvas_patch_grid_sizes,
        )
