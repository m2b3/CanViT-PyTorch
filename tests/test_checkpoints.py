"""Naming SSOT: published repo names and the defaults they inherit.

These strings are load-bearing — probe training records them, canvit-eval
resolves them, and the HF Hub repos are already published under them.
"""

import inspect

import pytest

from canvit_pytorch.checkpoints import (
    ABLATION_CHECKPOINTS,
    ABLATION_MODEL_SHORTS,
    CANVIT_REPO_ROOT,
    ade20k_dinov3_probe_name,
    ade20k_probe_name,
    ade20k_probe_repo,
    resolve_canvit_repo,
)
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.model.pretrain_common import pretrain_repo_id_stem
from canvit_pytorch.model.pretraining.hub import make_repo_id, teacher_shortname
from canvit_pytorch.model.rgb_pretraining.hub import make_rgb_repo_id


class TestProbeNames:
    def test_flagship_ade20k_probe_name(self):
        # The published flagship probe (README quickstart).
        assert ade20k_probe_name("in21k", scene=512, grid=64) == "probe-ade20k-40k-s512-c64-in21k"

    def test_ade20k_probe_repo_prefixes_root(self):
        repo = ade20k_probe_repo("in21k", scene=512, grid=64)
        assert repo == f"{CANVIT_REPO_ROOT}/probe-ade20k-40k-s512-c64-in21k"

    def test_dinov3_probe_name(self):
        assert ade20k_dinov3_probe_name("dv3b16", resolution=512) == "probe-ade20k-40k-dv3b16-512px"

    def test_resolve_canvit_repo(self):
        assert resolve_canvit_repo("x") == f"{CANVIT_REPO_ROOT}/x"


class TestAblationRegistry:
    def test_shorts_keyed_by_checkpoint_repo(self):
        assert set(ABLATION_MODEL_SHORTS) == set(ABLATION_CHECKPOINTS.values())

    def test_shorts_follow_slug_rule(self):
        for slug, repo in ABLATION_CHECKPOINTS.items():
            assert ABLATION_MODEL_SHORTS[repo] == f"abl-{slug}"


class TestTeacherShortname:
    def test_flagship_teacher(self):
        assert teacher_shortname("dinov3_vitb16") == "dv3b16"

    def test_unknown_teacher_raises(self):
        with pytest.raises(AssertionError):
            teacher_shortname("dinov2_vitb14")


def test_repo_id_defaults_match_canvit_config():
    """Hub naming defaults once drifted from CanViTConfig; pin them together."""
    cfg = CanViTConfig()
    for fn in (pretrain_repo_id_stem, make_repo_id, make_rgb_repo_id):
        params = inspect.signature(fn).parameters
        assert params["enable_vpe"].default == cfg.enable_vpe, fn.__name__
        assert params["canvas_update_mode"].default == cfg.canvas_update_mode, fn.__name__
