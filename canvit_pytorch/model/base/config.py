"""CanViT configuration."""

from dataclasses import dataclass


@dataclass
class CanViTConfig:
    """CanViT configuration."""

    rw_stride: int = 2
    enable_reads: bool = True
    n_backbone_registers: int = 5
    n_canvas_registers: int = 16
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    enable_vpe: bool = False

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim
