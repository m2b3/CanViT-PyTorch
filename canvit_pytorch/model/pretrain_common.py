"""Shared plumbing for CanViT pretraining variants.

Both pretraining variants — teacher distillation (:mod:`canvit_pytorch.model.pretraining`)
and RGB reconstruction (:mod:`canvit_pytorch.model.rgb_pretraining`) — are a base
:class:`CanViT` plus objective-specific decoder heads, serialized to / from the HF Hub
with the same ``config.json`` schema (``backbone_name`` + ``model_config`` + optional
extras). That serialization, the repo-id stem, and the decoder-head LayerNorm init are
shared here; objective-specific heads and naming suffixes live in each variant's module.
"""

import json
import logging
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from safetensors.torch import save_file
from torch import nn

log = logging.getLogger(__name__)


def init_decoder_ln_weight(ln: nn.LayerNorm, dim: int) -> None:
    """Init a decoder-head LayerNorm weight to 1/sqrt(dim). Shared by all pretraining heads."""
    ln.weight.data.fill_(1.0 / math.sqrt(dim))


def pretrain_repo_id_stem(
    *, owner: str, backbone_name: str, glimpse_size_px: int, scene_size_px: int,
    dataset: str, enable_vpe: bool = True, canvas_update_mode: str = "additive",
) -> str:
    """Objective-neutral repo-id stem, e.g. ``owner/canvitb16-add-vpe-pretrain-g128px-s512px-in1k``.

    Each variant appends its own suffix (``-dv3b16`` for a teacher, ``-rgbrecon`` for pixels).
    """
    assert backbone_name.startswith("vit"), f"Expected vit* backbone, got {backbone_name}"
    short = backbone_name.removeprefix("vit")
    update_tag = {"additive": "-add", "convex": "-cvx"}[canvas_update_mode]
    variant = update_tag + ("-vpe" if enable_vpe else "")
    return f"{owner}/canvit{short}{variant}-pretrain-g{glimpse_size_px}px-s{scene_size_px}px-{dataset}"


def upload_pretrain_to_hf(
    model: nn.Module,
    repo_id: str,
    *,
    extra_config: dict | None = None,
    private: bool = True,
    extra_metadata: dict | None = None,
    card: str | None = None,
) -> str:
    """Serialize ``{backbone_name, model_config, **extra_config}`` + weights and push. Returns repo_id.

    ``model`` must expose ``.backbone_name`` (str) and ``.cfg`` (a dataclass). ``extra_config``
    holds variant-specific top-level fields (e.g. teacher standardizer grid sizes).
    ``card``, when given, is written as README.md (the Hub model card).
    """
    backbone_name = getattr(model, "backbone_name", None)
    assert backbone_name is not None, "model.backbone_name not set — load via from_checkpoint"

    cfg: Any = model.cfg  # CanViTConfig subclass (dataclass); model is nn.Module so attr is loosely typed
    config: dict = {"backbone_name": backbone_name, "model_config": asdict(cfg)}
    if extra_config:
        config.update(extra_config)
    if extra_metadata is not None:
        config["metadata"] = extra_metadata

    log.info("Pushing to %s (private=%s)", repo_id, private)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        (tmppath / "config.json").write_text(json.dumps(config, indent=2, default=str))
        save_file(model.state_dict(), tmppath / "model.safetensors")
        if card is not None:
            (tmppath / "README.md").write_text(card)

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)

    log.info("Pushed to https://huggingface.co/%s", repo_id)
    return repo_id
