"""Tests for CanViTForPretraining Hub helpers: repo-id registry + model card."""

from canvit_pytorch.checkpoints import (
    FLAGSHIP_PRETRAIN_REPO,
    PRETRAIN_CHECKPOINTS,
    PRETRAIN_MODEL_SHORTS,
)
from canvit_pytorch.model.pretraining.hub import descriptive_metadata
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


def test_descriptive_metadata_excludes_deanonymizing_fields():
    raw = {
        "dataset": "in1k", "step": 100, "git_commit": "abc", "train_loss": 0.9,
        "hostname": "g13.nibi.sharcnet", "cmdline": ["/scratch/yberreby/x"],
        "slurm_job_id": "1", "comet_id": "x", "provenance_history": {},
        "training_config_history": {},
    }
    meta = descriptive_metadata(raw)
    for leak in ("hostname", "cmdline", "slurm_job_id", "comet_id",
                 "provenance_history", "training_config_history"):
        assert leak not in meta
    assert meta["dataset"] == "in1k" and meta["step"] == 100 and meta["git_commit"] == "abc"


def test_model_card_omits_step_clause_when_unknown():
    card = pretrain_model_card(
        PRETRAIN_CHECKPOINTS["in1k"], dataset="in1k", teacher_name="dinov3_vitb16",
        scene_px=512, glimpse_px=128, canvas_grid=32, step=None,
    )
    assert "None" not in card
