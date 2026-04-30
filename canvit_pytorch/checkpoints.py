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
