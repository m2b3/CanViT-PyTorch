"""Repo-id construction for CanViT-owned models and probes.

Single source of truth for the CanViT checkpoint root. Every CanViT-owned
repo-id flows through :func:`resolve_canvit_repo`; third-party repos
(``facebook/...``, etc.) stay as bare string literals.

The default ``"canvit"`` resolves to Hub IDs prefixed with ``canvit/``
(e.g. ``canvit/canvitb16-...``). Override via ``$CANVIT_REPO_ROOT`` to
redirect every load — the value can be either an HF org prefix or a local path:

    # Local checkpoint bundle: ship checkpoints in a directory tree.
    export CANVIT_REPO_ROOT="$(pwd)/canvit_checkpoints"

Both shapes work transparently because ``PyTorchModelHubMixin.from_pretrained``
takes either an HF repo-id or a local directory path.
"""

import os

CANVIT_REPO_ROOT = os.environ.get("CANVIT_REPO_ROOT", "canvit").rstrip("/")


def resolve_canvit_repo(name: str) -> str:
    return f"{CANVIT_REPO_ROOT}/{name}"


def ade20k_probe_name(model_short: str, *, scene: int, grid: int, steps_k: int = 40) -> str:
    """Repo name for a canvas-feature ADE20K segmentation probe."""
    return f"probe-ade20k-{steps_k}k-s{scene}-c{grid}-{model_short}"


def ade20k_dinov3_probe_name(model_short: str, *, resolution: int, steps_k: int = 40) -> str:
    """Repo name for a DINOv3-feature ADE20K segmentation probe."""
    return f"probe-ade20k-{steps_k}k-{model_short}-{resolution}px"


def ade20k_probe_repo(model_short: str, *, scene: int, grid: int, steps_k: int = 40) -> str:
    return resolve_canvit_repo(ade20k_probe_name(model_short, scene=scene, grid=grid, steps_k=steps_k))


# Pretraining-ablation checkpoints (slug -> repo-id). Slugs are shared with
# canvit-eval's batch runner and the paper exporter's ablation registry.
ABLATION_CHECKPOINTS: dict[str, str] = {
    slug: resolve_canvit_repo(name)
    for slug, name in {
        "baseline":      "canvitb16-abl-baseline-2026-03-02",
        "dcan256":       "canvitb16-abl-dcan256-2026-03-02",
        "qkvo-dcan256":  "canvitb16-abl-qkvo-dcan256-2026-03-02",
        "qkvo-dcan384":  "canvitb16-abl-qkvo-dcan384-2026-03-02",
        "no-reads":      "canvitb16-abl-no-reads-2026-03-02",
        "rw-stride6":    "canvitb16-abl-rw-stride6-2026-03-03",
        "no-dense":      "canvitb16-abl-no-dense-2026-03-02",
        "no-fiid-1riid": "canvitb16-abl-no-fiid-1riid-2026-03-02",
        "no-fiid-2riid": "canvitb16-abl-no-fiid-2riid-2026-03-06",
        "no-bptt":       "canvitb16-abl-no-bptt-2026-03-06",
        "vit-s":         "canvitb16-abl-vit-s-2026-03-03",
        "no-vpe":        "canvitb16-abl-no-vpe-2026-03-03",
    }.items()
}

# Probe-repo naming suffix per ablation checkpoint, keyed by repo-id to match
# how probe training records `model_repo` in its checkpoint config.
ABLATION_MODEL_SHORTS: dict[str, str] = {
    repo: f"abl-{slug}" for slug, repo in ABLATION_CHECKPOINTS.items()
}
