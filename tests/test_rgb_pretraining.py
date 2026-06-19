"""Tests for CanViTForRGBReconstruction + the objective-agnostic base loader.

CPU-only architecture sanity checks (no network). Also guards that routing the
teacher-distillation hub through the shared pretrain_common helpers did not change
its public repo-id strings or state_dict layout.
"""

import torch

from canvit_pytorch import (
    CanViT,
    CanViTForPretraining,
    CanViTForPretrainingConfig,
    CanViTForRGBReconstruction,
    CanViTForRGBReconstructionHFHub,
    Viewpoint,
    create_backbone,
    load_canvit_base,
    make_rgb_repo_id,
    patchify,
    sample_at_viewpoint,
    unpatchify,
)
from canvit_pytorch.model.pretraining.hub import make_repo_id

B = 2
CANVAS_GRID = 8  # small for fast CPU test
PATCH_PX = 16  # vits16 / vitb16 patch size
SCENE_PX = CANVAS_GRID * PATCH_PX  # 128
EXPECTED_CANVAS_DIM = 64  # canvas_num_heads(2) * canvas_head_dim(32)
RGB_PATCH_DIM = 3 * PATCH_PX * PATCH_PX  # 768

_REDUCED_CFG = {
    "n_canvas_registers": 4,
    "canvas_num_heads": 2,
    "canvas_head_dim": 32,
    "rw_stride": 2,
    "canvas_update_mode": "additive",
    "enable_vpe": False,
}


def _reduced_canvit_config(**overrides):
    from canvit_pytorch import CanViTConfig
    return CanViTConfig(**{**_REDUCED_CFG, **overrides})


def _rgb_model() -> CanViTForRGBReconstruction:
    return CanViTForRGBReconstruction(
        backbone=create_backbone("vits16"),
        cfg=_reduced_canvit_config(),
        backbone_name="vits16",
    )


def _dummy_glimpse():
    scene = torch.randn(B, 3, SCENE_PX, SCENE_PX)
    vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
    glimpse = sample_at_viewpoint(spatial=scene, viewpoint=vp, glimpse_size_px=SCENE_PX)
    return glimpse, vp


class TestPatchify:
    def test_roundtrip(self):
        img = torch.randn(B, 3, SCENE_PX, SCENE_PX)
        patches = patchify(img, PATCH_PX)
        assert patches.shape == (B, CANVAS_GRID ** 2, RGB_PATCH_DIM)
        recon = unpatchify(patches, PATCH_PX)
        assert recon.shape == img.shape
        assert torch.allclose(recon, img, atol=1e-6)

    def test_patch_count_scales_with_resolution(self):
        # Same patch_px, larger scene -> more patches (resolution-agnostic head target).
        big = torch.randn(1, 3, 256, 256)
        patches = patchify(big, PATCH_PX)
        assert patches.shape == (1, (256 // PATCH_PX) ** 2, RGB_PATCH_DIM)


class TestRGBReconstructionModel:
    def test_head_dims(self):
        model = _rgb_model()
        assert model.patch_px == PATCH_PX
        assert model.canvas_dim == EXPECTED_CANVAS_DIM
        assert model.rgb_head["proj"].in_features == EXPECTED_CANVAS_DIM
        assert model.rgb_head["proj"].out_features == RGB_PATCH_DIM

    def test_predict_shapes(self):
        model = _rgb_model().eval()
        state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            patches = model.predict_rgb_patches(state.canvas)
            image = model.predict_rgb_image(state.canvas)
        assert patches.shape == (B, CANVAS_GRID ** 2, RGB_PATCH_DIM)
        assert image.shape == (B, 3, SCENE_PX, SCENE_PX)

    def test_resolution_agnostic_same_weights(self):
        """One head reconstructs at any canvas grid; no per-grid parameters exist."""
        model = _rgb_model().eval()
        for grid in (CANVAS_GRID, CANVAS_GRID * 2):
            state = model.init_state(batch_size=1, canvas_grid_size=grid)
            with torch.inference_mode():
                image = model.predict_rgb_image(state.canvas)
            assert image.shape == (1, 3, grid * PATCH_PX, grid * PATCH_PX)

    def test_forward_advances_state(self):
        model = _rgb_model().eval()
        glimpse, vp = _dummy_glimpse()
        state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            out = model(glimpse=glimpse, state=state, viewpoint=vp)
            patches = model.predict_rgb_patches(out.state.canvas)
        assert out.state.canvas.shape == state.canvas.shape
        assert patches.shape == (B, CANVAS_GRID ** 2, RGB_PATCH_DIM)

    def test_state_dict_is_base_plus_rgb_head_only(self):
        """Teacher-free: no scene_*/standardizer keys; exactly one rgb_head added to base."""
        model = _rgb_model()
        bare = CanViT(backbone=create_backbone("vits16"), cfg=_reduced_canvit_config())
        base_keys = set(bare.state_dict().keys())
        keys = set(model.state_dict().keys())
        head_keys = keys - base_keys
        assert head_keys == {
            "rgb_head.norm.weight", "rgb_head.norm.bias",
            "rgb_head.proj.weight", "rgb_head.proj.bias",
        }
        assert not any("scene_" in k or "standardizer" in k for k in keys)

    def test_from_checkpoint_roundtrip(self, tmp_path):
        model = _rgb_model().eval()
        ckpt = {
            "state_dict": model.state_dict(),
            "model_config": vars(model.cfg),
            "backbone_name": "vits16",
        }
        path = tmp_path / "rgb.pt"
        torch.save(ckpt, path)
        loaded = CanViTForRGBReconstruction.from_checkpoint(path).eval()
        state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            assert torch.allclose(
                model.predict_rgb_patches(state.canvas),
                loaded.predict_rgb_patches(state.canvas),
            )


class TestHubRoundtrip:
    def test_save_and_load_pretrained(self, tmp_path):
        model = CanViTForRGBReconstructionHFHub(backbone_name="vits16", model_config=_REDUCED_CFG)
        model.eval()
        model.save_pretrained(tmp_path)
        loaded = CanViTForRGBReconstructionHFHub.from_pretrained(tmp_path).eval()
        state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            assert torch.allclose(
                model.predict_rgb_patches(state.canvas),
                loaded.predict_rgb_patches(state.canvas),
            )


class TestBaseLoaderObjectiveAgnostic:
    """load_canvit_base must recover identical canvas features from either objective."""

    def test_loads_rgb_checkpoint_base(self, tmp_path):
        model = CanViTForRGBReconstructionHFHub(backbone_name="vits16", model_config=_REDUCED_CFG).eval()
        model.save_pretrained(tmp_path)
        base = load_canvit_base(str(tmp_path)).eval()
        assert isinstance(base, CanViT)
        glimpse, vp = _dummy_glimpse()
        state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        with torch.inference_mode():
            ref = model(glimpse=glimpse, state=state, viewpoint=vp)
            got = base(glimpse=glimpse, state=base.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID), viewpoint=vp)
        # Same base weights -> identical canvas evolution.
        assert torch.allclose(ref.state.canvas, got.state.canvas, atol=1e-6)

    def test_loads_teacher_checkpoint_base(self, tmp_path):
        from canvit_pytorch.model.pretraining.hub import CanViTForPretrainingHFHub
        model = CanViTForPretrainingHFHub(
            backbone_name="vits16",
            model_config={**_REDUCED_CFG, "teacher_dim": 384},
            canvas_patch_grid_sizes=[CANVAS_GRID],
        ).eval()
        model.save_pretrained(tmp_path)
        base = load_canvit_base(str(tmp_path)).eval()
        assert isinstance(base, CanViT)
        # Teacher heads + standardizers must have been dropped, base weights present.
        assert base.canvas_dim == EXPECTED_CANVAS_DIM


class TestTeacherRefactorCharacterization:
    """Guards that DRY-ing the teacher hub through pretrain_common changed nothing observable."""

    def test_make_repo_id_unchanged(self):
        # The exact flagship-style string the downstream probe naming depends on.
        repo = make_repo_id(
            owner="canvit", backbone_name="vitb16", glimpse_size_px=128, scene_size_px=512,
            dataset="in21k", teacher_name="dinov3_vitb16", enable_vpe=True, canvas_update_mode="additive",
        )
        assert repo == "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16"

    def test_make_rgb_repo_id(self):
        repo = make_rgb_repo_id(
            owner="canvit", backbone_name="vitb16", glimpse_size_px=128, scene_size_px=512,
            dataset="in1k", enable_vpe=True, canvas_update_mode="additive",
        )
        assert repo == "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in1k-rgbrecon"

    def test_teacher_head_keys_present(self):
        model = CanViTForPretraining(
            backbone=create_backbone("vits16"),
            cfg=CanViTForPretrainingConfig(**{**_REDUCED_CFG, "teacher_dim": 384}),
            backbone_name="vits16",
            canvas_patch_grid_sizes=[CANVAS_GRID],
        )
        keys = set(model.state_dict().keys())
        assert {"scene_patches_head.proj.weight", "scene_cls_head.proj.weight"} <= keys
        assert any("scene_standardizers" in k for k in keys)
