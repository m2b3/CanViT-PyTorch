"""Shape assertion utilities."""

from torch import Tensor


def assert_shape(x: Tensor, expected: tuple[int, ...]) -> None:
    assert x.shape == expected, f"Expected shape {expected}, got {x.shape}"
