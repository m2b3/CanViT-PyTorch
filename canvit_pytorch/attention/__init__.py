"""Asymmetric cross-attention for canvas-based vision transformers.

Canvas Attention uses dense projections (Linear) on the local stream (few tokens)
and Identity on the canvas stream (many tokens).

- CanvasReadAttention (CRA): local queries canvas
- CanvasWriteAttention (CWA): canvas queries local
"""

from canvit_pytorch.attention.base import CanvasAttention
from canvit_pytorch.attention.read import CanvasReadAttention
from canvit_pytorch.attention.write import CanvasWriteAttention


__all__ = [
    "CanvasAttention",
    "CanvasReadAttention",
    "CanvasWriteAttention",
]
