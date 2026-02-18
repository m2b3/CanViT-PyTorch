from typing import final

from torch import nn

from canvit_pytorch.attention.base import CanvasAttention


@final
class CanvasWriteAttention(CanvasAttention):
    """Canvas queries local (CWA).

    Dense projections (Linear) on local side, Identity on canvas side.
    Canvas is updated additively: canvas = canvas + CWA(canvas, local).
    """

    def __init__(
        self,
        *,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(
            q_in_dim=canvas_dim,
            kv_in_dim=local_dim,
            canvas_dim=canvas_dim,
            out_dim=canvas_dim,
            num_heads=num_heads,
        )
        self.k_proj = nn.Linear(local_dim, canvas_dim)
        self.v_proj = nn.Linear(local_dim, canvas_dim)
