from typing import final

from torch import nn

from canvit_pytorch.attention.base import CanvasAttention


@final
class CanvasReadAttention(CanvasAttention):
    """Local queries canvas (CRA).

    Dense projections (Linear) on local side, Identity on canvas side.
    Local is projected up to canvas_dim for attention, output projected back to local_dim.
    """

    def __init__(
        self,
        *,
        local_dim: int,
        canvas_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__(
            q_in_dim=local_dim,
            kv_in_dim=canvas_dim,
            canvas_dim=canvas_dim,
            out_dim=local_dim,
            num_heads=num_heads,
        )
        self.q_proj = nn.Linear(local_dim, canvas_dim)
        self.out_proj = nn.Linear(canvas_dim, local_dim)
