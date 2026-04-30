"""Tests for CanViTForImageClassification."""

import pytest
import torch
from PIL import Image

from canvit_pytorch import (
    CanViTForImageClassification,
    CanViTForPretrainingHFHub,
    Viewpoint,
    fuse_probe,
    resolve_canvit_repo,
    sample_at_viewpoint,
)
from canvit_pytorch.preprocess import preprocess

PRETRAINED_REPO = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
PROBE_REPO = resolve_canvit_repo("dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe")
FINETUNED_REPO = resolve_canvit_repo("canvitb16-add-vpe-finetune-g128px-s512px-in1k-2026-04-06")
CAT_IMAGE = "test_data/Cat03.jpg"
CANVAS_GRID = 32
B = 2

# IN1k class index for "tabby cat" (281), "tiger cat" (282), "Egyptian cat" (285)
CAT_CLASSES = {281, 282, 285}


@pytest.fixture
def dummy_input():
    scene = torch.randn(B, 3, 512, 512)
    vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=vp, glimpse_size_px=128)
    return glimpse, vp


class TestFromPretrainedWithProbe:
    @pytest.fixture(scope="class")
    def clf(self):
        return CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=PRETRAINED_REPO, probe_repo=PROBE_REPO,
        ).eval()

    def test_forward_shape(self, clf, dummy_input):
        glimpse, vp = dummy_input
        state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            logits, new_state = clf(glimpse=glimpse, state=state, viewpoint=vp)
        assert logits.shape == (B, 1000)
        assert new_state.recurrent_cls.shape == (B, 1, clf.local_dim)

    def test_canvit_and_head_match_forward(self, clf, dummy_input):
        glimpse, vp = dummy_input
        state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            logits_combined, _ = clf(glimpse=glimpse, state=state, viewpoint=vp)
            out = clf.canvit(glimpse=glimpse, state=state, viewpoint=vp)
            logits_split = clf.head(clf.norm(out.state.recurrent_cls[:, 0].float()))
        assert (logits_combined - logits_split).abs().max() < 1e-6

    def test_forward_keeps_head_in_fp32_under_autocast(self, clf, dummy_input):
        """forward() must return fp32 logits even when outer autocast is bf16.

        The naive ``self.head(self.norm(cls.float()))`` is undone by Linear/LayerNorm
        autocast policies that re-cast inputs to bf16. The fix is an explicit nested
        ``torch.autocast(enabled=False)``. Regression here is silent (~0.4% flip rate),
        so this assertion is the only signal.
        """
        glimpse, vp = dummy_input
        state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = clf(glimpse=glimpse, state=state, viewpoint=vp)
        assert logits.dtype == torch.float32, f"expected fp32, got {logits.dtype}"

    def test_fusion_matches_unfused(self, clf, dummy_input):
        """Fused LN → Linear must match the original 4-stage unfused pipeline."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        glimpse, vp = dummy_input

        # Load the full pretraining model for the unfused reference path
        pretrained = CanViTForPretrainingHFHub.from_pretrained(PRETRAINED_REPO).eval()

        with torch.inference_mode():
            # Run both CanViT instances on the same input; they have the same weights.
            state_fused = clf.canvit.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
            state_unfused = pretrained.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)

            out_fused = clf.canvit(glimpse=glimpse, state=state_fused, viewpoint=vp)
            out_unfused = pretrained(glimpse=glimpse, state=state_unfused, viewpoint=vp)

            # Unfused: norm → proj → destandardize → probe
            rcls = out_unfused.state.recurrent_cls
            cls_pred = pretrained.predict_scene_teacher_cls(rcls)
            cls_std, _ = pretrained.standardizers(CANVAS_GRID)
            cls_destd = cls_std.destandardize(cls_pred)
            probe_sd = load_file(hf_hub_download(PROBE_REPO, "model.safetensors"))
            probe = torch.nn.Linear(probe_sd["weight"].shape[1], probe_sd["weight"].shape[0])
            probe.load_state_dict({"weight": probe_sd["weight"], "bias": probe_sd["bias"]})
            logits_unfused = probe(cls_destd)

            # Fused
            logits_fused = clf.head(clf.norm(out_fused.state.recurrent_cls[:, 0].float()))

        assert (logits_fused - logits_unfused).abs().max() < 1e-4


class TestFromPretrained:
    @pytest.fixture(scope="class")
    def clf(self):
        return CanViTForImageClassification.from_pretrained(FINETUNED_REPO).eval()

    def test_forward_shape(self, clf, dummy_input):
        glimpse, vp = dummy_input
        state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            logits, _ = clf(glimpse=glimpse, state=state, viewpoint=vp)
        assert logits.shape == (B, 1000)

    def test_properties(self, clf):
        assert clf.n_classes == 1000
        assert clf.local_dim == 768


class TestEndToEnd:
    """Verify that a cat image is classified as a cat via both paths."""

    @staticmethod
    def _classify_cat(clf: CanViTForImageClassification) -> int:
        img_size = CANVAS_GRID * clf.canvit.backbone.patch_size_px
        image = preprocess(img_size)(Image.open(CAT_IMAGE).convert("RGB"))
        assert isinstance(image, torch.Tensor)
        image = image.unsqueeze(0)

        vp = Viewpoint.full_scene(batch_size=1, device=torch.device("cpu"))
        glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=128)

        with torch.inference_mode():
            state = clf.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
            logits, _ = clf(glimpse=glimpse, state=state, viewpoint=vp)
        return logits.argmax(dim=-1).item()

    def test_fused_probe_classifies_cat(self):
        clf = CanViTForImageClassification.from_pretrained_with_probe(
            pretrained_repo=PRETRAINED_REPO, probe_repo=PROBE_REPO,
        ).eval()
        pred = self._classify_cat(clf)
        assert pred in CAT_CLASSES, f"Expected cat class, got {pred}"

    def test_finetuned_classifies_cat(self):
        clf = CanViTForImageClassification.from_pretrained(FINETUNED_REPO).eval()
        pred = self._classify_cat(clf)
        assert pred in CAT_CLASSES, f"Expected cat class, got {pred}"


class TestFuseProbe:
    def test_shapes(self):
        D, teacher_dim, n_classes = 768, 768, 1000
        W_fused, b_fused = fuse_probe(
            W_proj=torch.randn(teacher_dim, D),
            b_proj=torch.randn(teacher_dim),
            mu=torch.randn(teacher_dim),
            sigma=torch.rand(teacher_dim).abs() + 0.1,
            W_probe=torch.randn(n_classes, teacher_dim),
            b_probe=torch.randn(n_classes),
        )
        assert W_fused.shape == (n_classes, D)
        assert b_fused.shape == (n_classes,)

    def test_correctness(self):
        D, teacher_dim, n_classes = 16, 16, 5
        W_proj = torch.randn(teacher_dim, D)
        b_proj = torch.randn(teacher_dim)
        mu = torch.randn(teacher_dim)
        sigma = torch.rand(teacher_dim).abs() + 0.1
        W_probe = torch.randn(n_classes, teacher_dim)
        b_probe = torch.randn(n_classes)

        W_fused, b_fused = fuse_probe(
            W_proj=W_proj, b_proj=b_proj,
            mu=mu, sigma=sigma,
            W_probe=W_probe, b_probe=b_probe,
        )

        x = torch.randn(4, D)
        logits_seq = (sigma * (x @ W_proj.T + b_proj) + mu) @ W_probe.T + b_probe
        logits_fused = x @ W_fused.T + b_fused
        assert (logits_seq - logits_fused).abs().max() < 1e-5
