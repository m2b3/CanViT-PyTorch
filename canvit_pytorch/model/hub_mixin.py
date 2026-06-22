"""Safe HF Hub loading mixin — checks for key mismatches that PyTorchModelHubMixin silently ignores."""

import logging
from typing import TypeVar

from huggingface_hub import ModelHubMixin
from torch import nn

log = logging.getLogger(__name__)

# Matches PyTorchModelHubMixin._load_as_safetensor's TypeVar so the override is
# type-compatible; every concrete subclass is also an nn.Module (asserted below).
T = TypeVar("T", bound=ModelHubMixin)


class SafeHubMixin:
    """Mixin that checks for key mismatches when loading HF Hub checkpoints.

    HuggingFace's PyTorchModelHubMixin loads with strict=False and ignores the
    (missing, unexpected) return value, silently leaving mismatched parameters
    at random init.  This mixin overrides _load_as_safetensor to check.

    Must appear BEFORE PyTorchModelHubMixin in the MRO so that this
    _load_as_safetensor is found first.
    """

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        import safetensors.torch
        assert isinstance(model, nn.Module)
        missing, unexpected = safetensors.torch.load_model(model, model_file, strict=False, device=map_location)
        if missing or unexpected:
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  WARNING: Checkpoint key mismatch during model loading!     ║\n"
                "╚══════════════════════════════════════════════════════════════╝\n"
            )
            if missing:
                msg += (
                    f"\n  {len(missing)} model keys not found in checkpoint:\n"
                    f"    {sorted(missing)[:5]}\n"
                    "  → These parameters are left at random initialization.\n"
                    "    The model will produce garbage outputs.\n"
                )
            if unexpected:
                msg += (
                    f"\n  {len(unexpected)} checkpoint keys not found in model:\n"
                    f"    {sorted(unexpected)[:5]}\n"
                    "  → These weights were silently dropped.\n"
                )
            msg += (
                "\n  This usually means your canvit-pytorch version doesn't match\n"
                "  the checkpoint. Update with:\n"
                "\n"
                '    uv lock --upgrade-package canvit-pytorch && uv sync\n'
                "\n"
                "  or:\n"
                "\n"
                '    uv add "canvit-pytorch @ git+https://github.com/m2b3/CanViT-PyTorch.git"\n'
            )
            log.warning(msg)
        return model
