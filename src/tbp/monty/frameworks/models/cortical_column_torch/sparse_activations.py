# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sparse continuous activation utilities for CorticalColumnTorch.

Enforces SDR-like sparsity on continuous (float32) tensors via per-minicolumn
top-k selection.  Values are preserved (not binarized), giving the network
continuous attractor dynamics while maintaining the combinatorial capacity
and similarity-via-overlap properties of sparse codes.
"""

from __future__ import annotations

import torch


def enforce_sparsity(
    x: torch.Tensor,
    n_minicolumns: int,
    cells_per_mc: int,
    k_per_mc: int = 1,
) -> torch.Tensor:
    """Keep only the top-k activations per minicolumn, zero the rest.

    Parameters
    ----------
    x : Tensor of shape ``(n_cells,)`` or ``(batch, n_cells)``
        Continuous cell activations.
    n_minicolumns : int
        Number of minicolumns.
    cells_per_mc : int
        Cells per minicolumn.
    k_per_mc : int
        How many cells to keep active per minicolumn (default 1).

    Returns
    -------
    Tensor, same shape as *x*, with at most *k_per_mc* nonzeros per MC.
    """
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    batch = x.shape[0]
    n_cells = n_minicolumns * cells_per_mc
    # Reshape to (batch, n_minicolumns, cells_per_mc)
    x_mc = x[:, :n_cells].reshape(batch, n_minicolumns, cells_per_mc)

    if k_per_mc >= cells_per_mc:
        out = x_mc
    else:
        # Top-k per minicolumn
        topk_vals, topk_idx = torch.topk(x_mc, k_per_mc, dim=-1)
        out = torch.zeros_like(x_mc)
        out.scatter_(-1, topk_idx, topk_vals)

    result = out.reshape(batch, n_cells)
    if squeeze:
        result = result.squeeze(0)
    return result


def sparse_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute overlap between two sparse continuous patterns.

    Overlap = sum of element-wise min of absolute values, normalized by
    the smaller of the two L1 norms.
    """
    min_abs = torch.min(a.abs(), b.abs())
    overlap = min_abs.sum()
    norm = min(a.abs().sum(), b.abs().sum())
    if norm < 1e-8:
        return torch.tensor(0.0, device=a.device)
    return overlap / norm


def active_minicolumn_mask(
    x: torch.Tensor,
    n_minicolumns: int,
    cells_per_mc: int,
) -> torch.Tensor:
    """Return a boolean mask of shape ``(n_minicolumns,)`` indicating which
    minicolumns have any nonzero activation."""
    x_mc = x[:n_minicolumns * cells_per_mc].reshape(n_minicolumns, cells_per_mc)
    return x_mc.abs().sum(dim=-1) > 0
