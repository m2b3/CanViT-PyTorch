"""HuggingFace Hub integration for CanViTForPretraining."""

import logging
from pathlib import Path
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
    card: str | None = None,
) -> str:
    """Upload model to HuggingFace Hub under the given repo_id. Returns repo_id.

    extra_metadata is merged into config.json alongside model_config,
    backbone_name, and canvas_patch_grid_sizes (the teacher standardizer grid
    sizes). card, when given, is written as README.md (the Hub model card).
    """
    assert isinstance(model.cfg, CanViTForPretrainingConfig)
    return upload_pretrain_to_hf(
        model,
        repo_id,
        extra_config={"canvas_patch_grid_sizes": model.canvas_patch_grid_sizes},
        private=private,
        extra_metadata=extra_metadata,
        card=card,
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


# Model-descriptive checkpoint fields published in the Hub config.json. Whitelist,
# not blacklist: training provenance that can carry usernames / host / paths
# (hostname, cmdline, slurm_*, comet_id, *_history) is never published, so a
# public repo can't deanonymize.
_PUBLIC_METADATA_FIELDS = (
    "dataset", "teacher_name", "teacher_repo_id", "teacher_dim",
    "glimpse_grid_size", "scene_resolution", "step", "train_loss",
    "timestamp", "git_commit", "git_dirty",
)


def descriptive_metadata(raw: dict) -> dict:
    return {k: raw[k] for k in _PUBLIC_METADATA_FIELDS if k in raw}


def _migrate_standardizers_in_place(raw: dict) -> None:
    """Migrate legacy standardizer keys into state_dict if present. Mutates raw."""
    scene_legacy = raw.get("scene_norm_state")
    cls_legacy = raw.get("cls_norm_state")
    if scene_legacy is None:
        return  # no legacy keys to migrate

    assert cls_legacy is not None, "scene_norm_state present but cls_norm_state missing"
    assert scene_legacy["_initialized"].item(), "Legacy scene stats not initialized"
    assert cls_legacy["_initialized"].item(), "Legacy cls stats not initialized"

    grids = raw["canvas_patch_grid_sizes"]
    assert len(grids) == 1, f"Expected 1 grid size, got {grids}"
    G = str(grids[0])
    sd = raw["state_dict"]
    for prefix, legacy in [("scene_standardizers", scene_legacy), ("cls_standardizers", cls_legacy)]:
        for stat_name in ["mean", "var", "_initialized"]:
            sd[f"{prefix}.{G}.{stat_name}"] = legacy[stat_name]
    del raw["scene_norm_state"]
    del raw["cls_norm_state"]
    log.info("Migrated standardizers in-memory (grid=%s)", G)


def reconstruct_pretrain_model(raw: dict) -> CanViTForPretraining:
    """Rebuild CanViTForPretraining from a raw checkpoint dict (migrating legacy
    standardizers in-memory) and assert its standardizers are initialized."""
    _migrate_standardizers_in_place(raw)
    cfg = CanViTForPretrainingConfig(**raw["model_config"])
    model = CanViTForPretraining(
        backbone=create_backbone(raw["backbone_name"]),
        cfg=cfg,
        backbone_name=raw["backbone_name"],
        canvas_patch_grid_sizes=raw["canvas_patch_grid_sizes"],
    )
    model.load_state_dict(raw["state_dict"])
    for G in model.canvas_patch_grid_sizes:
        _, scene_std = model.standardizers(G)
        assert scene_std.initialized, (
            f"Standardizer not initialized for grid {G} after loading — checkpoint may be corrupt."
        )
    return model


def push_checkpoint_file(
    ckpt_path: Path,
    repo_id: str,
    *,
    private: bool = True,
    with_card: bool = True,
) -> str:
    """Reconstruct a pretraining model from a .pt and push it to the Hub with
    full provenance metadata and (optionally) a generated model card."""
    import torch

    from .model_card import pretrain_model_card

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = reconstruct_pretrain_model(raw)
    meta = descriptive_metadata(raw)

    card = None
    if with_card:
        card = pretrain_model_card(
            repo_id,
            dataset=raw["dataset"],
            teacher_name=raw["teacher_name"],
            scene_px=raw["scene_resolution"],
            glimpse_px=raw["glimpse_grid_size"] * model.backbone.patch_size_px,
            canvas_grid=raw["canvas_patch_grid_sizes"][0],
            step=raw.get("step"),
        )
    return upload_to_hf(model, repo_id, private=private, extra_metadata=meta, card=card)


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
