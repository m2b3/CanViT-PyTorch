"""CanViT for RGB-reconstruction pretraining: a teacher-free, MAE-style control.

The flagship objective reconstructs DINOv3 teacher features from the canvas
(:mod:`canvit_pytorch.model.pretraining`). This variant instead reconstructs raw
pixels, isolating the architecture's contribution from the teacher — the pixel-space
counterpart the paper contrasts itself against (DINOv3 latent space "rather than in
pixel space like passive MAEs"). Each canvas patch decodes one
``patch_size_px x patch_size_px`` RGB patch, so the head is resolution-agnostic: the
canvas grid is an inference-time choice and the reconstruction resolution scales with it.

Like MAE, this objective reconstructs patches only — there is no CLS head. Loss
computation lives in the trainer, mirroring :class:`CanViTForPretraining`.
"""

import logging
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn

from canvit_pytorch.backbone import ViTBackbone, create_backbone
from canvit_pytorch.model.base import CanViT
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.model.pretrain_common import init_decoder_ln_weight

log = logging.getLogger(__name__)

RGB_CHANNELS = 3


def patchify(image: Tensor, patch_px: int) -> Tensor:
    """``[B, 3, S, S] -> [B, (S/p)^2, 3*p*p]`` — row-major, channel-major within a patch.

    Inverse of :func:`unpatchify`. The RGB reconstruction target; matches the per-patch
    layout of :meth:`CanViTForRGBReconstruction.predict_rgb_patches`.
    """
    B, C, H, W = image.shape
    assert C == RGB_CHANNELS, f"Expected {RGB_CHANNELS} channels, got {C}"
    assert H == W, f"Expected square scene, got {H}x{W}"
    assert H % patch_px == 0, f"Scene size {H} not divisible by patch_px {patch_px}"
    g = H // patch_px
    x = image.reshape(B, C, g, patch_px, g, patch_px)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, g, g, C, p, p]
    return x.reshape(B, g * g, C * patch_px * patch_px)


def unpatchify(patches: Tensor, patch_px: int) -> Tensor:
    """``[B, g^2, 3*p*p] -> [B, 3, g*p, g*p]`` — inverse of :func:`patchify`."""
    B, n, d = patches.shape
    g = int(round(n ** 0.5))
    assert g * g == n, f"Patch count {n} is not a perfect square"
    assert d == RGB_CHANNELS * patch_px * patch_px, (
        f"Patch dim {d} != 3*{patch_px}^2 = {RGB_CHANNELS * patch_px * patch_px}"
    )
    x = patches.reshape(B, g, g, RGB_CHANNELS, patch_px, patch_px)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, g, p, g, p]
    return x.reshape(B, RGB_CHANNELS, g * patch_px, g * patch_px)


class CanViTForRGBReconstruction(CanViT):
    """CanViT with a per-patch RGB decoder head for pixel-space pretraining.

    The head decodes each canvas spatial token into a ``patch_size_px`` square RGB patch.
    No CLS head, no standardizers — the only objective is dense pixel reconstruction.
    """

    def __init__(
        self,
        *,
        backbone: ViTBackbone,
        cfg: CanViTConfig,
        backbone_name: str,
    ) -> None:
        super().__init__(backbone=backbone, cfg=cfg)

        canvas_dim = cfg.canvas_dim
        self.patch_px = backbone.patch_size_px
        out_dim = RGB_CHANNELS * self.patch_px * self.patch_px

        self.rgb_head = nn.ModuleDict({
            "norm": nn.LayerNorm(canvas_dim),
            "proj": nn.Linear(canvas_dim, out_dim),
        })
        head_norm = self.rgb_head["norm"]
        assert isinstance(head_norm, nn.LayerNorm)
        init_decoder_ln_weight(head_norm, canvas_dim)

        self.backbone_name = backbone_name

    def predict_rgb_patches(self, canvas: Tensor) -> Tensor:
        """Canvas spatial tokens -> per-patch RGB. ``[B, Ncan, Dcan] -> [B, g^2, 3*p*p]``."""
        x = self.get_spatial(canvas)
        return self.rgb_head["proj"](self.rgb_head["norm"](x)).contiguous()

    def predict_rgb_image(self, canvas: Tensor) -> Tensor:
        """Folded reconstruction image for viz/eval. ``-> [B, 3, g*p, g*p]``."""
        return unpatchify(self.predict_rgb_patches(canvas), self.patch_px)

    @classmethod
    def from_checkpoint(cls, path: Path | str, *, map_location: str | torch.device = "cpu") -> Self:
        """Load from a local ``.pt`` checkpoint (``state_dict`` + ``model_config`` + ``backbone_name``)."""
        log.info("Loading RGB-reconstruction checkpoint from %s (map_location=%s)", path, map_location)
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model_config = ckpt["model_config"]
        extra_keys = set(model_config) - set(CanViTConfig.__dataclass_fields__)
        if extra_keys:
            log.info("Ignoring non-CanViTConfig keys in model_config: %s", sorted(extra_keys))
        cfg = CanViTConfig(**{k: v for k, v in model_config.items() if k in CanViTConfig.__dataclass_fields__})
        model = cls(
            backbone=create_backbone(ckpt["backbone_name"]),
            cfg=cfg,
            backbone_name=ckpt["backbone_name"],
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
