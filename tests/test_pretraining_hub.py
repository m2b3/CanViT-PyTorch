"""Tests for CanViTForPretraining Hub helpers: repo-id registry + model card."""

from canvit_pytorch.checkpoints import (
    FLAGSHIP_PRETRAIN_REPO,
    PRETRAIN_CHECKPOINTS,
    PRETRAIN_MODEL_SHORTS,
)
from canvit_pytorch.model.pretraining.hub.model_card import pretrain_model_card


def test_pretrain_model_shorts_is_exact_inverse():
    assert PRETRAIN_MODEL_SHORTS == {repo: slug for slug, repo in PRETRAIN_CHECKPOINTS.items()}
    assert FLAGSHIP_PRETRAIN_REPO == PRETRAIN_CHECKPOINTS["in21k"]
    for slug, repo in PRETRAIN_CHECKPOINTS.items():
        assert PRETRAIN_MODEL_SHORTS[repo] == slug


def test_model_card_fills_data_specific_fields():
    repo = PRETRAIN_CHECKPOINTS["in1k"]
    card = pretrain_model_card(
        repo, dataset="in1k", teacher_name="dinov3_vitb16",
        scene_px=512, glimpse_px=128, canvas_grid=32, step=2_001_792,
    )
    for placeholder in ("{repo_id}", "{dataset_label}", "{step_clause}", "{scene_px}", "{canvas_grid}"):
        assert placeholder not in card
    assert repo in card
    assert "ImageNet-1k" in card
    assert "2,001,792" in card
    assert "berreby2026canvit" in card


def test_model_card_omits_step_clause_when_unknown():
    card = pretrain_model_card(
        PRETRAIN_CHECKPOINTS["in1k"], dataset="in1k", teacher_name="dinov3_vitb16",
        scene_px=512, glimpse_px=128, canvas_grid=32, step=None,
    )
    assert "None" not in card
