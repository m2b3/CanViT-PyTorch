"""Integration test for torch.export compatibility.

Verifies CanViT can be exported without graph breaks.
"""

import pytest
import torch

from canvit_pytorch import (
    CanViT,
    CanViTConfig,
    CanViTOutput,
    RecurrentState,
    Viewpoint,
    create_backbone,
    sample_at_viewpoint,
)


def _make_model() -> CanViT:
    backbone = create_backbone("vits16")
    cfg = CanViTConfig(rw_stride=4, n_canvas_registers=4, canvas_num_heads=4, canvas_head_dim=64)
    return CanViT(backbone=backbone, cfg=cfg)


@pytest.mark.slow
def test_torch_export_no_graph_breaks() -> None:
    """torch.export succeeds (no graph breaks) and output matches eager."""
    model = _make_model().eval()

    B, canvas_grid, glimpse_px = 1, 8, 64
    image = torch.randn(B, 3, 128, 128)
    state = model.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    vp = Viewpoint.full_scene(batch_size=B, device=image.device)
    glimpse = sample_at_viewpoint(spatial=image, viewpoint=vp, glimpse_size_px=glimpse_px)

    # Eager forward
    with torch.inference_mode():
        eager_out = model(glimpse=glimpse, state=state, viewpoint=vp)

    # Export
    torch.export.register_dataclass(Viewpoint, serialized_type_name="canvit_pytorch.Viewpoint")
    torch.export.register_dataclass(RecurrentState, serialized_type_name="canvit_pytorch.RecurrentState")
    torch.export.register_dataclass(CanViTOutput, serialized_type_name="canvit_pytorch.CanViTOutput")

    exported = torch.export.export(
        model,
        args=(),
        kwargs={"glimpse": glimpse, "state": state, "viewpoint": vp},
    )

    # Exported forward
    exported_out = exported.module()(glimpse=glimpse, state=state, viewpoint=vp)
    assert isinstance(exported_out, CanViTOutput)

    # Outputs match
    diff = (eager_out.state.canvas - exported_out.state.canvas).abs().max().item()
    assert diff < 1e-5, f"Export output differs from eager: max diff = {diff}"
