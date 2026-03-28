# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Recurrent connections for attractor dynamics within a cortical column.

In L2/3 of real cortex, cells have recurrent excitatory connections to other
cells in the same layer. These form attractor basins: a partial input causes
the network to settle into the nearest stored pattern (pattern completion).

Each cell has a fixed number of outgoing connections (fan-out K) to cells
in *other* minicolumns, stored as index arrays. This is biologically realistic
(each pyramidal cell forms ~2000-5000 synapses) and memory-efficient:
O(n_cells * K) instead of O(n_cells^2).

Learning uses simple Hebbian correlation: when two connected cells are
co-active during training, strengthen their connection. Anti-Hebbian:
when only one fires, weaken.

Settling: given an initial activation, iteratively apply recurrent input +
re-inhibit to maintain sparsity, until the active set stabilizes.
"""

from __future__ import annotations

import numpy as np


class RecurrentConnections:
    """Recurrent (L2/3) connections for attractor dynamics.

    Uses sparse fixed fan-out: each cell has K outgoing connections to
    randomly chosen cells in other minicolumns. Weights are float32.

    Parameters
    ----------
    n_cells : int
        Total cells in the column.
    n_minicolumns : int
        Number of minicolumns (for inhibition during settling).
    n_cells_per_minicolumn : int
        Cells per minicolumn.
    sparsity : float
        Target fraction of active minicolumns.
    fan_out : int
        Number of outgoing connections per cell.
    learning_rate : float
        Hebbian learning rate for recurrent weights.
    decay_rate : float
        Per-step decay on all recurrent weights (prevents saturation).
    max_settling_iterations : int
        Maximum iterations for settling loop.
    convergence_threshold : int
        If fewer than this many cells change between iterations, stop.
    recurrent_strength : float
        Scaling factor for recurrent input (relative to feedforward).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        n_minicolumns: int = 2048,
        n_cells_per_minicolumn: int = 8,
        sparsity: float = 0.03,
        fan_out: int = 128,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
        max_settling_iterations: int = 5,
        convergence_threshold: int = 0,
        recurrent_strength: float = 0.5,
        seed: int = 42,
    ):
        self.n_cells = n_cells
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_minicolumn = n_cells_per_minicolumn
        self._sparsity = sparsity
        self._n_active_mcs = max(1, int(sparsity * n_minicolumns))
        self._fan_out = fan_out
        self._lr = learning_rate
        self._decay = decay_rate
        self._max_iters = max_settling_iterations
        self._conv_threshold = convergence_threshold
        self._strength = recurrent_strength
        self._rng = np.random.RandomState(seed)

        # Sparse connectivity: each cell connects to K cells in other
        # minicolumns. _targets[i, j] = cell index that cell i's j-th
        # connection projects to. _weights[i, j] = strength of that
        # connection.
        k = n_cells_per_minicolumn
        self._targets = np.zeros((n_cells, fan_out), dtype=np.int32)
        self._weights = np.zeros((n_cells, fan_out), dtype=np.float32)

        # Wire up: for each cell, sample fan_out targets from other
        # minicolumns (exclude own minicolumn).
        for cell_i in range(n_cells):
            mc_i = cell_i // k
            mc_start = mc_i * k
            mc_end = mc_start + k

            # Build candidate pool: all cells except own minicolumn
            candidates = np.concatenate([
                np.arange(0, mc_start, dtype=np.int32),
                np.arange(mc_end, n_cells, dtype=np.int32),
            ])

            # Sample fan_out targets (or all if fewer candidates)
            n_sample = min(fan_out, len(candidates))
            chosen = self._rng.choice(candidates, size=n_sample, replace=False)

            self._targets[cell_i, :n_sample] = chosen
            if n_sample < fan_out:
                # Pad with -1 sentinel (won't be used — weights stay 0)
                self._targets[cell_i, n_sample:] = -1

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, active_mask: np.ndarray) -> None:
        """Hebbian learning on existing connections.

        For each active cell, strengthen connections to co-active targets
        and weaken connections to inactive targets.

        Parameters
        ----------
        active_mask : np.ndarray, shape (n_cells,), bool
            Currently active cells.
        """
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            return

        # For active cells: check which of their targets are also active
        # targets shape: (n_active, fan_out)
        targets = self._targets[active_idx]  # (n_active, fan_out)

        # Look up whether each target is active. Use a padded array so
        # index -1 maps to False.
        padded_active = np.empty(self.n_cells + 1, dtype=np.bool_)
        padded_active[:self.n_cells] = active_mask
        padded_active[-1] = False

        target_active = padded_active[targets]  # (n_active, fan_out), bool

        # Hebbian: co-active → strengthen
        self._weights[active_idx] += self._lr * target_active

        # Anti-Hebbian: active→inactive → weaken (smaller rate)
        self._weights[active_idx] -= self._lr * 0.25 * (~target_active)

        # Clip to [0, 1]
        np.clip(self._weights, 0.0, 1.0, out=self._weights)

    def decay(self) -> None:
        """Global weight decay to prevent saturation."""
        if self._decay > 0:
            self._weights = np.maximum(
                0.0, self._weights - self._decay
            )

    # ------------------------------------------------------------------
    # Settling (attractor convergence)
    # ------------------------------------------------------------------

    def settle(
        self,
        initial_active: np.ndarray,
        ff_overlap: np.ndarray = None,
        boost_factors: np.ndarray = None,
    ) -> tuple:
        """Run settling loop until convergence.

        Minicolumn selection is FIXED by the feedforward spatial pooler —
        settling only determines which cell within each active minicolumn
        fires, using recurrent input as context. This prevents recurrent
        weights from overriding feedforward object identity.

        Parameters
        ----------
        initial_active : np.ndarray, shape (n_cells,), bool
            Initial cell activation from feedforward + dendritic prediction.
        ff_overlap : np.ndarray, shape (n_minicolumns,), float or None
            Unused (kept for API compatibility).
        boost_factors : np.ndarray, shape (n_minicolumns,), float or None
            Unused (kept for API compatibility).

        Returns
        -------
        settled_active : np.ndarray, shape (n_cells,), bool
            Converged cell activation pattern.
        settled_mc : np.ndarray, shape (n_minicolumns,), bool
            Minicolumn activation (unchanged from feedforward).
        n_iterations : int
            Number of settling iterations.
        """
        k = self.n_cells_per_minicolumn
        current = initial_active.copy()

        # Minicolumns are fixed by feedforward — never changed by settling
        current_2d = current.reshape(self.n_minicolumns, k)
        fixed_mc = current_2d.any(axis=1).copy()
        active_mc_idx = np.where(fixed_mc)[0]

        if len(active_mc_idx) == 0:
            return current, fixed_mc, 0

        for iteration in range(self._max_iters):
            # Compute recurrent input per cell via sparse gather
            padded_active = np.empty(self.n_cells + 1, dtype=np.float32)
            padded_active[:self.n_cells] = current.astype(np.float32)
            padded_active[-1] = 0.0

            target_active = padded_active[self._targets]  # (n_cells, fan_out)
            recurrent_input = (self._weights * target_active).sum(axis=1)

            # Within each fixed minicolumn, use recurrent input to pick
            # the best cell (disambiguate from burst → single cell)
            new_active = np.zeros(self.n_cells, dtype=np.bool_)
            new_2d = new_active.reshape(self.n_minicolumns, k)

            n_changed = 0
            for mc in active_mc_idx:
                s, e = mc * k, (mc + 1) * k
                cell_scores = recurrent_input[s:e]
                was_active = current[s:e]

                if cell_scores.max() > 0:
                    # Pick the cell with strongest recurrent support
                    best = cell_scores.argmax()
                    new_2d[mc, best] = True
                else:
                    # No recurrent info — keep current activation
                    new_2d[mc] = was_active

                if not np.array_equal(new_2d[mc], was_active):
                    n_changed += 1

            current[:] = new_active

            if n_changed == 0:
                return current, fixed_mc, iteration + 1

        return current, fixed_mc, self._max_iters

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def total_weight(self) -> float:
        """Sum of all recurrent weights (measure of learned structure)."""
        return float(self._weights.sum())

    @property
    def n_active_connections(self) -> int:
        """Number of recurrent connections with weight > 0."""
        return int((self._weights > 0).sum())

    def memory_bytes(self) -> int:
        """Estimated memory usage."""
        return self._targets.nbytes + self._weights.nbytes
