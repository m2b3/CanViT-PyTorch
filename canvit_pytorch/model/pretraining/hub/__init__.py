"""HuggingFace Hub integration for CanViTForPretraining."""

import logging
from typing import Any

from huggingface_hub import PyTorchModelHubMixin

from canvit_pytorch.backbone import create_backbone

from ..impl import CanViTForPretraining, CanViTForPretrainingConfig

log = logging.getLogger(__name__)


class CanViTForPretrainingHFHub(
    CanViTForPretraining,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/yberreby/CanViT-PyTorch",
):
    """CanViTForPretraining with HuggingFace Hub integration.

    Usage:
        model = CanViTForPretrainingHFHub.from_pretrained("canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    """

    def __init__(
        self,
        backbone_name: str,
        model_config: dict[str, Any],
        canvas_patch_grid_sizes: list[int],
    ):
        unknown = model_config.keys() - CanViTForPretrainingConfig.__dataclass_fields__.keys()
        if unknown:
            log.warning("Ignoring unknown config keys: %s", sorted(unknown))
        known = {k: v for k, v in model_config.items() if k not in unknown}
        super().__init__(
            backbone=create_backbone(backbone_name),
            cfg=CanViTForPretrainingConfig(**known),
            backbone_name=backbone_name,
            canvas_patch_grid_sizes=canvas_patch_grid_sizes,
        )
