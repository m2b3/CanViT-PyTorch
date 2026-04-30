"""CanViT for image classification."""

import logging
from pathlib import Path
from typing import cast, get_args

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from canvit_pytorch.backbone import BackboneName, create_backbone
from canvit_pytorch.model.hub_mixin import SafeHubMixin
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.model.base.impl import CanViT, RecurrentState
from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
from canvit_pytorch.viewpoint import Viewpoint

log = logging.getLogger(__name__)


def fuse_probe(
    *,
    W_proj: Tensor,
    b_proj: Tensor,
    mu: Tensor,
    sigma: Tensor,
    W_probe: Tensor,
    b_probe: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse proj → destandardize → probe into a single linear transform.

    The pretrained eval chain after LayerNorm is three affine transforms::

        s = W_proj @ z + b_proj
        d = σ ⊙ s + μ
        logits = W_probe @ d + b_probe

    Since affine ∘ affine = affine, these collapse into::

        W_fused = W_probe @ diag(σ) @ W_proj
        b_fused = W_probe @ (σ ⊙ b_proj + μ) + b_probe

    Returns (W_fused [n_classes, D], b_fused [n_classes]).
    """
    teacher_dim, D = W_proj.shape
    n_classes = W_probe.shape[0]
    assert b_proj.shape == (teacher_dim,)
    assert mu.shape == (teacher_dim,) and sigma.shape == (teacher_dim,)
    assert W_probe.shape == (n_classes, teacher_dim) and b_probe.shape == (n_classes,)

    out_dtype = W_proj.dtype
    W_proj = W_proj.double()
    b_proj = b_proj.double()
    mu = mu.double()
    sigma = sigma.double()
    W_probe = W_probe.double()
    b_probe = b_probe.double()

    B_mat = sigma.unsqueeze(1) * W_proj
    assert B_mat.shape == (teacher_dim, D)
    b_mid = sigma * b_proj + mu
    assert b_mid.shape == (teacher_dim,)
    W_fused = W_probe @ B_mat
    assert W_fused.shape == (n_classes, D)
    b_fused = W_probe @ b_mid + b_probe
    assert b_fused.shape == (n_classes,)
    return W_fused.to(out_dtype), b_fused.to(out_dtype)


class CanViTForImageClassification(
    nn.Module,
    SafeHubMixin,
    PyTorchModelHubMixin,
    library_name="canvit-pytorch",
    repo_url="https://github.com/m2b3/CanViT-PyTorch",
):
    """:class:`CanViT` + LN → Linear classification head.

    Wraps a bare :class:`CanViT` (no pretraining heads, no standardizers).
    The classification head is always LN(D) → Linear(D, n_classes).

    Example::

        # From a fused checkpoint:
        clf = CanViTForImageClassification.from_pretrained("<org>/<repo>").eval()

        # From a pretrained CanViT + probe (fuses at construction time):
        clf = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo="<org>/canvitb16-add-vpe-pretrain-...",
            probe_repo="<org>/dinov3-vitb16-...-linear-clf-probe",
        ).eval()

        # Both have the same forward:
        state = clf.init_state(batch_size=B, canvas_grid_size=32)
        logits, state = clf(glimpse=glimpse, state=state, viewpoint=vp)
    """

    def __init__(
        self,
        *,
        backbone_name: BackboneName,
        model_config: dict,
        n_classes: int,
    ):
        super().__init__()
        # Filter to known CanViTConfig fields (HF config.json may have extras)
        known_fields = CanViTConfig.__dataclass_fields__
        cfg_dict = {k: v for k, v in model_config.items() if k in known_fields}
        cfg = CanViTConfig(**cfg_dict)
        self.canvit = CanViT(backbone=create_backbone(backbone_name), cfg=cfg)
        D = self.canvit.local_dim
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)

    @property
    def local_dim(self) -> int:
        """Embedding dimension of the CanViT local stream and head input."""
        return self.canvit.local_dim

    @property
    def n_classes(self) -> int:
        return self.head.out_features

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

    def forward(
        self, *, glimpse: Tensor, state: RecurrentState, viewpoint: Viewpoint,
    ) -> tuple[Tensor, RecurrentState]:
        """Returns (logits [B, n_classes], new_state).

        For CanViT-only execution without the classification head, call ``self.canvit(...)`` directly.
        For head-only on a cached CLS token, wrap in ``torch.autocast(enabled=False)``
        and call ``self.head(self.norm(cls.float()))`` — bare ``.float()`` is undone
        by ``nn.Linear``/``nn.LayerNorm`` autocast policies.
        """
        out = self.canvit(glimpse=glimpse, state=state, viewpoint=viewpoint)
        cls = out.state.recurrent_cls[:, 0]
        with torch.autocast(device_type=cls.device.type, enabled=False):
            logits = self.head(self.norm(cls.float()))
        return logits, out.state

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_repo: str,
        probe_repo: str,
        canvas_grid: int = 32,
    ) -> "CanViTForImageClassification":
        """Load a pretrained CanViT, fuse proj → destandardize → probe into LN → Linear.

        Loads the full pretraining model temporarily to extract fusion ingredients
        (scene_cls_head, standardizers, probe), then copies only the base CanViT
        weights into the classifier. Pretraining heads are discarded.

        See :func:`fuse_probe` for the algebra.
        """
        log.info("Loading pretrained model from %s", pretrained_repo)
        pretrained = CanViTForPretrainingHFHub.from_pretrained(pretrained_repo)
        D = pretrained.local_dim

        log.info("Loading probe from %s", probe_repo)
        probe_path = Path(probe_repo)
        probe_st = probe_path / "model.safetensors" if probe_path.is_dir() else hf_hub_download(probe_repo, "model.safetensors")
        probe_sd = load_file(probe_st)

        # Validate probe/projection compatibility — the probe consumes the
        # output of scene_cls_head["proj"] (a Linear layer in the pretrained
        # CanViT's CLS head, NOT the inner ViT backbone).
        proj = pretrained.scene_cls_head["proj"]
        assert isinstance(proj, nn.Linear)
        probe_in_dim = probe_sd["weight"].shape[1]
        assert probe_in_dim == proj.out_features, (
            f"Probe/projection dim mismatch: probe expects {probe_in_dim}, "
            f"scene_cls_head proj produces {proj.out_features}"
        )

        cls_std, _ = pretrained.standardizers(canvas_grid)
        assert cls_std.initialized, "CLS standardizer not initialized — wrong canvas_grid?"

        W_fused, b_fused = fuse_probe(
            W_proj=proj.weight.data,
            b_proj=proj.bias.data,
            mu=cls_std.mean.squeeze(0),
            sigma=(cls_std.var.squeeze(0) + cls_std.eps).sqrt(),
            W_probe=probe_sd["weight"],
            b_probe=probe_sd["bias"],
        )

        # Build classifier wrapping a bare CanViT (no pretraining heads)
        n_classes = W_fused.shape[0]
        cfg = pretrained.cfg
        assert pretrained.backbone_name in get_args(BackboneName), f"Unknown backbone: {pretrained.backbone_name!r}"
        model = cls(
            backbone_name=cast(BackboneName, pretrained.backbone_name),
            model_config={k: v for k, v in vars(cfg).items() if k in CanViTConfig.__dataclass_fields__},
            n_classes=n_classes,
        )

        # Copy base CanViT weights from pretrained (excluding pretraining heads)
        base_sd = {k: v for k, v in pretrained.state_dict().items()
                   if not any(k.startswith(pfx) for pfx in
                              ("scene_cls_head.", "scene_patches_head.",
                               "cls_standardizers.", "scene_standardizers."))}
        missing, unexpected = model.canvit.load_state_dict(base_sd, strict=False)
        assert not missing, f"Missing CanViT keys: {missing}"
        assert not unexpected, f"Unexpected CanViT keys: {unexpected}"

        # Set fused head weights
        model.head.weight.data.copy_(W_fused)
        model.head.bias.data.copy_(b_fused)
        pretrained_norm = pretrained.scene_cls_head["norm"]
        assert isinstance(pretrained_norm, nn.LayerNorm)
        model.norm.weight.data.copy_(pretrained_norm.weight.data)
        model.norm.bias.data.copy_(pretrained_norm.bias.data)

        log.info("Fused classifier: LN(%d) → Linear(%d, %d), pretraining heads discarded", D, D, n_classes)
        return model
