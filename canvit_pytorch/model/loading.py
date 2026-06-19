"""Objective-agnostic loading of the base :class:`CanViT` from any pretrained checkpoint.

Downstream feature tasks — ADE20K segmentation probes, canvas feature extraction — need
only the base CanViT weights (canvas attention, backbone, VPE, recurrent state), regardless
of whether the checkpoint was produced by teacher distillation or RGB reconstruction. The
objective-specific decoder heads (teacher projections + standardizers, or the RGB head) are
training-time machinery and are dropped here. This is the single entry point for that load,
so probe code does not need to know which pretraining objective produced a checkpoint.
"""

import json
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from canvit_pytorch.backbone import create_backbone
from canvit_pytorch.model.base import CanViT, CanViTConfig

log = logging.getLogger(__name__)

# Decoder-head / standardizer prefixes dropped when loading base canvas features.
# Covers both pretraining objectives (teacher distillation + RGB reconstruction).
_PRETRAIN_HEAD_PREFIXES = (
    "scene_cls_head.",
    "scene_patches_head.",
    "cls_standardizers.",
    "scene_standardizers.",
    "rgb_head.",
)


def load_canvit_base(repo_id_or_path: str, *, map_location: str = "cpu") -> CanViT:
    """Load a bare :class:`CanViT` (canvas features only) from an HF repo id or local dir.

    Objective-agnostic: works for teacher-distilled and RGB-reconstruction checkpoints alike.
    Asserts every base-CanViT weight is present; the only tolerated absences are the
    objective-specific head/standardizer keys, which are intentionally dropped.
    """
    p = Path(repo_id_or_path)
    if p.is_dir():
        config_path = p / "config.json"
        weights_path = p / "model.safetensors"
    else:
        config_path = Path(hf_hub_download(repo_id_or_path, "config.json"))
        weights_path = Path(hf_hub_download(repo_id_or_path, "model.safetensors"))

    config = json.loads(config_path.read_text())
    backbone_name = config["backbone_name"]
    model_config = config["model_config"]
    cfg = CanViTConfig(**{k: v for k, v in model_config.items() if k in CanViTConfig.__dataclass_fields__})

    model = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)

    full_sd = load_file(weights_path, device=map_location)
    base_sd = {
        k: v for k, v in full_sd.items()
        if not any(k.startswith(pfx) for pfx in _PRETRAIN_HEAD_PREFIXES)
    }
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    assert not missing, f"Missing base CanViT weights in {repo_id_or_path}: {missing}"
    assert not unexpected, (
        f"Unexpected non-base keys after head filter in {repo_id_or_path}: {unexpected}"
    )
    log.info("Loaded base CanViT from %s (backbone=%s, canvas_dim=%d)", repo_id_or_path, backbone_name, model.canvas_dim)
    return model
