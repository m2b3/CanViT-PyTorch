"""Fixed standardization for token sequences."""

import torch
from torch import Tensor, nn


class PositionAwareStandardizer(nn.Module):
    """Per-position z-score standardization for [B, N, D] sequences.

    Standardization: (x - mean) / std, producing mean=0, std=1.
    Stats shape: [N, D] - one mean/var per token position per dimension.
    Stats are computed once via set_stats() and then frozen.
    """

    mean: Tensor
    var: Tensor
    _initialized: Tensor

    def __init__(self, n_tokens: int, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_tokens, embed_dim))
        self.register_buffer("var", torch.ones(n_tokens, embed_dim))
        self.register_buffer("_initialized", torch.tensor(False))

    @property
    def initialized(self) -> bool:
        val = self._initialized.item()
        assert isinstance(val, bool)
        return val

    def set_stats(self, data: Tensor) -> None:
        """Compute and freeze stats from data. Args: data [N_samples, N_tokens, D]."""
        assert data.dim() == 3, f"Expected [N, tokens, D], got {data.shape}"
        assert data.shape[1] == self.n_tokens, f"Token mismatch: {data.shape[1]} vs {self.n_tokens}"
        assert data.shape[2] == self.embed_dim, f"Dim mismatch: {data.shape[2]} vs {self.embed_dim}"

        with torch.no_grad():
            self.mean.copy_(data.mean(dim=0))
            self.var.copy_(data.var(dim=0, unbiased=False))
            self._initialized.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        """Standardize: (x - mean) / std. Shape: [B, N, D] -> [B, N, D]."""
        assert self.initialized, "Stats not initialized - call set_stats() or load_state_dict() first"
        return (x - self.mean) / (self.var + self.eps).sqrt()

    def destandardize(self, x: Tensor) -> Tensor:
        """Invert standardization: x * std + mean."""
        return x * (self.var + self.eps).sqrt() + self.mean


class CLSStandardizer(PositionAwareStandardizer):
    """Standardizer for single CLS token."""

    def __init__(self, embed_dim: int, **kwargs: float):
        super().__init__(n_tokens=1, embed_dim=embed_dim, **kwargs)


class PatchStandardizer(PositionAwareStandardizer):
    """Standardizer for spatial patch grid (grid_size x grid_size tokens)."""

    def __init__(self, grid_size: int, embed_dim: int, **kwargs: float):
        super().__init__(n_tokens=grid_size * grid_size, embed_dim=embed_dim, **kwargs)
        self.grid_size = grid_size


__all__ = ["CLSStandardizer", "PatchStandardizer", "PositionAwareStandardizer"]
