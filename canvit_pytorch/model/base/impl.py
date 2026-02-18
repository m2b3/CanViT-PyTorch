"""CanViT: Dual-stream vision transformer with canvas cross-attention."""

import logging
import math
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch
from torch import Tensor, nn

from canvit_pytorch.attention import CanvasReadAttention, CanvasWriteAttention
from canvit_pytorch.backbone.vit import ViTBackbone
from canvit_pytorch.coords import canvas_coords_for_glimpse, grid_coords
from canvit_pytorch.model.base.config import CanViTConfig
from canvit_pytorch.rope import RoPE, compute_rope, make_rope_periods
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from canvit_pytorch.vpe import VPEEncoder

T = TypeVar("T")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalTokens:
    """Tokens in the local stream (processed by backbone blocks).

    Layout: [vpe?, recurrent_cls, registers, patches]
    """

    vpe: Tensor | None     # [B, 1, D] - viewpoint encoding (optional)
    recurrent_cls: Tensor  # [B, 1, D] - persists across timesteps
    registers: Tensor      # [B, n_regs, D] - backbone register tokens
    patches: Tensor        # [B, H*W, D] - image patch tokens

    def pack(self) -> Tensor:
        parts = []
        if self.vpe is not None:
            parts.append(self.vpe)
        parts.extend([self.recurrent_cls, self.registers, self.patches])
        return torch.cat(parts, dim=1)

    @property
    def n_prefix(self) -> int:
        return (1 if self.vpe is not None else 0) + 1 + self.registers.shape[1]

    @staticmethod
    def unpack(x: Tensor, *, has_vpe: bool, n_registers: int, n_patches: int) -> "LocalTokens":
        idx = 0
        vpe: Tensor | None = None
        if has_vpe:
            vpe = x[:, idx : idx + 1]
            idx += 1
        recurrent_cls = x[:, idx : idx + 1]
        idx += 1
        registers = x[:, idx : idx + n_registers]
        idx += n_registers
        patches = x[:, idx : idx + n_patches]
        return LocalTokens(vpe, recurrent_cls, registers, patches)


@dataclass
class RecurrentState:
    canvas: Tensor  # [B, n_canvas_registers + G², canvas_dim]
    recurrent_cls: Tensor  # [B, 1, local_dim]


@dataclass
class CanViTOutput:
    state: RecurrentState
    local_patches: Tensor  # [B, H*W, local_dim]
    vpe: Tensor | None  # [B, local_dim]


def compute_rw_positions(
    n_blocks: int, rw_stride: int, *, enable_reads: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute read/write block positions.

    Pattern: [rw_stride blocks] R [rw_stride blocks] W ... Always ends with W.
    When enable_reads is False, the write schedule is unchanged but no reads are placed.
    """
    rw_positions = list(range(rw_stride - 1, n_blocks, rw_stride))

    read_after: list[int] = []
    write_after: list[int] = []
    for i, pos in enumerate(rw_positions):
        if i % 2 == 0:
            read_after.append(pos)
        else:
            write_after.append(pos)

    last_block = n_blocks - 1
    if not write_after or write_after[-1] != last_block:
        write_after.append(last_block)

    if not enable_reads:
        read_after = []

    return tuple(read_after), tuple(write_after)


class CanViT(nn.Module):
    """Dual-stream vision transformer with canvas cross-attention.

    Canvas layout: [registers | spatial].
    CLS token is recurrent in the ViT stream (not part of canvas).
    Generalizes to any canvas and glimpse grid sizes at runtime.
    """

    read_after_blocks: tuple[int, ...]
    write_after_blocks: tuple[int, ...]

    def __init__(
        self,
        *,
        backbone: ViTBackbone,
        cfg: CanViTConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        n_blocks = backbone.n_blocks
        local_dim = backbone.embed_dim
        canvas_dim = cfg.canvas_dim

        assert cfg.rw_stride >= 1

        read_after, write_after = compute_rw_positions(n_blocks, cfg.rw_stride, enable_reads=cfg.enable_reads)
        self.read_after_blocks = read_after
        self.write_after_blocks = write_after

        log.info(f"CanViT: {n_blocks} blocks, rw_stride={cfg.rw_stride}, read_after={read_after}, write_after={write_after}")

        self.canvas_read = nn.ModuleList([
            CanvasReadAttention(local_dim=local_dim, canvas_dim=canvas_dim, num_heads=cfg.canvas_num_heads)
            for _ in range(len(read_after))
        ])
        self.canvas_write = nn.ModuleList([
            CanvasWriteAttention(local_dim=local_dim, canvas_dim=canvas_dim, num_heads=cfg.canvas_num_heads)
            for _ in range(len(write_after))
        ])

        log.info(f"Canvas attention: {len(read_after)} reads, {len(write_after)} writes, vpe={cfg.enable_vpe}")

        canvas_scale = 1.0 / math.sqrt(canvas_dim)
        self.canvas_register_init = nn.Parameter(torch.randn(1, cfg.n_canvas_registers, canvas_dim) * canvas_scale)
        self.canvas_spatial_init = nn.Parameter(torch.randn(1, 1, canvas_dim) * canvas_scale)
        log.info(f"Canvas registers: {cfg.n_canvas_registers}")

        local_scale = 1.0 / math.sqrt(local_dim)
        self.recurrent_cls_init = nn.Parameter(torch.randn(1, 1, local_dim) * local_scale)
        self.backbone_registers = nn.Parameter(torch.empty(1, cfg.n_backbone_registers, local_dim))
        nn.init.normal_(self.backbone_registers, std=0.02)

        log.info(f"Backbone registers: {cfg.n_backbone_registers}")

        self.vpe: VPEEncoder | None = None
        if cfg.enable_vpe:
            assert local_dim % 2 == 0, "embed_dim must be even for VPE"
            self.vpe = VPEEncoder(rff_dim=local_dim)
            log.info("VPE enabled")

    @property
    def canvas_dim(self) -> int:
        return self.cfg.canvas_dim

    @property
    def local_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def n_canvas_registers(self) -> int:
        return self.cfg.n_canvas_registers

    def get_spatial(self, canvas: Tensor) -> Tensor:
        return canvas[:, self.n_canvas_registers:]

    def init_canvas(self, *, batch_size: int, canvas_grid_size: int) -> Tensor:
        n_spatial = canvas_grid_size ** 2
        canvas_registers = self.canvas_register_init.expand(batch_size, -1, -1)
        canvas_spatial = self.canvas_spatial_init.expand(batch_size, n_spatial, -1)
        return torch.cat([canvas_registers, canvas_spatial], dim=1)

    def init_state(self, *, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return RecurrentState(
            canvas=self.init_canvas(batch_size=batch_size, canvas_grid_size=canvas_grid_size),
            recurrent_cls=self.recurrent_cls_init.expand(batch_size, -1, -1),
        )

    def _get_spatial_positions(self, canvas: Tensor, canvas_grid_size: int | None = None) -> Tensor:
        if canvas_grid_size is None:
            n_spatial = canvas.shape[1] - self.n_canvas_registers
            canvas_grid_size = int(math.sqrt(n_spatial))
            assert canvas_grid_size * canvas_grid_size == n_spatial
        return grid_coords(H=canvas_grid_size, W=canvas_grid_size, device=canvas.device).flatten(0, 1)

    def _compute_local_positions(self, viewpoint: Viewpoint, glimpse_grid_size: int) -> Tensor:
        return canvas_coords_for_glimpse(
            center=viewpoint.centers,
            scale=viewpoint.scales,
            H=glimpse_grid_size,
            W=glimpse_grid_size,
        ).flatten(1, 2)

    def forward(
        self,
        *,
        glimpse: Tensor,
        state: RecurrentState,
        viewpoint: Viewpoint,
        canvas_grid_size: int | None = None,
        canvas_rope: RoPE | None = None,
    ) -> CanViTOutput:
        B = glimpse.shape[0]
        recurrent_cls = state.recurrent_cls
        canvas = state.canvas

        patches, H, W = self.backbone.patch_embed(glimpse)
        glimpse_grid_size = H
        assert H == W, f"Expected square grid, got H={H}, W={W}"

        n_regs = self.cfg.n_backbone_registers
        n_patches = H * W
        registers = self.backbone_registers.expand(B, -1, -1)

        vpe_tok: Tensor | None = None
        has_vpe = self.vpe is not None
        if has_vpe:
            assert self.vpe is not None
            vpe_tok = self.vpe(
                y=viewpoint.centers[:, 0],
                x=viewpoint.centers[:, 1],
                s=viewpoint.scales,
            ).unsqueeze(1).to(patches.dtype)

        tokens = LocalTokens(vpe=vpe_tok, recurrent_cls=recurrent_cls, registers=registers, patches=patches)
        local = tokens.pack()

        n_prefix = tokens.n_prefix
        assert local.shape[1] == n_prefix + n_patches

        local_pos = self._compute_local_positions(viewpoint, glimpse_grid_size)

        device = glimpse.device
        rope_base = self.backbone.rope_base
        backbone_periods = make_rope_periods(head_dim=self.backbone.head_dim, base=rope_base, device=device)
        canvas_periods = make_rope_periods(head_dim=self.cfg.canvas_head_dim, base=rope_base, device=device)

        local_rope_backbone = compute_rope(positions=local_pos, periods=backbone_periods)
        local_rope_xattn = compute_rope(positions=local_pos, periods=canvas_periods)

        if canvas_rope is not None:
            spatial_rope = canvas_rope
        else:
            spatial_pos = self._get_spatial_positions(canvas, canvas_grid_size).unsqueeze(0).expand(B, -1, -1)
            spatial_rope = compute_rope(positions=spatial_pos, periods=canvas_periods)

        read_idx = 0
        write_idx = 0

        for block_idx in range(self.backbone.n_blocks):
            local = self.backbone.blocks[block_idx](local, local_rope_backbone)

            if read_idx < len(self.read_after_blocks) and block_idx == self.read_after_blocks[read_idx]:
                read_out = self.canvas_read[read_idx](
                    query=local, kv=canvas, query_rope=local_rope_xattn, kv_rope=spatial_rope
                )
                local = local + read_out
                read_idx += 1

            if write_idx < len(self.write_after_blocks) and block_idx == self.write_after_blocks[write_idx]:
                write_out = self.canvas_write[write_idx](
                    query=canvas, kv=local, query_rope=spatial_rope, kv_rope=local_rope_xattn
                )
                canvas = canvas + write_out
                write_idx += 1

        out = LocalTokens.unpack(local, has_vpe=has_vpe, n_registers=n_regs, n_patches=n_patches)

        vpe_processed = out.vpe.squeeze(1) if out.vpe is not None else None

        new_state = RecurrentState(
            canvas=canvas,
            recurrent_cls=out.recurrent_cls.contiguous(),
        )
        return CanViTOutput(
            state=new_state,
            local_patches=out.patches.contiguous(),
            vpe=vpe_processed,
        )

    def forward_reduce(
        self,
        *,
        image: Tensor,
        viewpoints: list[Viewpoint],
        glimpse_size_px: int,
        canvas_grid_size: int,
        init_fn: Callable[[RecurrentState], T],
        step_fn: Callable[[T, CanViTOutput, Viewpoint, Tensor], T],
        state: RecurrentState | None = None,
    ) -> tuple[T, RecurrentState]:
        assert len(viewpoints) > 0

        batch_size = image.shape[0]
        if state is None:
            state = self.init_state(batch_size=batch_size, canvas_grid_size=canvas_grid_size)

        acc = init_fn(state)

        for vp in viewpoints:
            glimpse = sample_at_viewpoint(
                spatial=image, viewpoint=vp, glimpse_size_px=glimpse_size_px
            )
            out = self.forward(glimpse=glimpse, state=state, viewpoint=vp)
            state = out.state
            acc = step_fn(acc, out, vp, glimpse)

        return acc, state

