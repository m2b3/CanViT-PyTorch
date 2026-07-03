"""Teacher model loading via HuggingFace transformers.

Provides frozen DINOv3 teacher for:
- Teacher baseline comparison during evaluation
- Target features during CanViT training
"""

import logging

import torch
from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel

from canvit_pytorch.backbone.vit import NormFeatures

log = logging.getLogger(__name__)

DINOV3_VITB16_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
N_PREFIX_TOKENS = 5  # 1 CLS + 4 register tokens


class DINOv3Teacher(nn.Module):
    """Frozen DINOv3 teacher loaded from HuggingFace Hub."""

    model: PreTrainedModel

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size  # type: ignore[return-value]

    @property
    def n_blocks(self) -> int:
        return self.model.config.num_hidden_layers  # type: ignore[return-value]

    def forward_norm_features(self, images: Tensor) -> NormFeatures:
        """Forward pass returning post-norm patches and CLS token."""
        out = self.model(images).last_hidden_state
        assert out.shape[1] > N_PREFIX_TOKENS
        return NormFeatures(patches=out[:, N_PREFIX_TOKENS:], cls=out[:, 0])


def load_teacher(
    repo_id: str,
    device: torch.device,
) -> DINOv3Teacher:
    """Load frozen DINOv3 teacher from HuggingFace Hub.

    Args:
        repo_id: HuggingFace model ID, e.g. "facebook/dinov3-vitb16-pretrain-lvd1689m"
        device: Target device
    """
    log.info("Loading teacher: %s", repo_id)
    log.info("  device: %s", device)

    hf_model = AutoModel.from_pretrained(repo_id, dtype=torch.float32)
    assert isinstance(hf_model, PreTrainedModel)
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    hf_model.to(device)  # pyright: ignore[reportArgumentType]

    teacher = DINOv3Teacher(hf_model)
    log.info("  embed_dim: %d, n_blocks: %d", teacher.embed_dim, teacher.n_blocks)
    return teacher
