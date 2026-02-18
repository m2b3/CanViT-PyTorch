"""Coordinate utilities for canvas and glimpse grids.

Two coordinate systems:
- Canvas: global [-1, 1]² space where the scene lives
- Retinotopic: local [-1, 1]² relative to a glimpse grid

Transform: canvas_pos = glimpse_center + glimpse_scale * retinotopic_pos

Usage::

    from canvit_pytorch.coords import grid_coords, canvas_coords_for_glimpse

    # Canvas coords (fixed grid)
    canvas_pos = grid_coords(H=16, W=16, device=device)  # [H, W, 2]

    # Glimpse coords in canvas space
    glimpse_pos = canvas_coords_for_glimpse(
        center=center, scale=scale, H=14, W=14
    )  # [B, H, W, 2]
"""

from canvit_pytorch.coords.impl import canvas_coords_for_glimpse, grid_coords

__all__ = ["grid_coords", "canvas_coords_for_glimpse"]
