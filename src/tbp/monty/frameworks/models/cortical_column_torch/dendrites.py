# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sparse dendritic segments with a cached connected-synapse table.

Track 14 originally expressed dendritic prediction as sparse COO matmul,
but torch 1.13.1 on this macOS runtime can segfault while constructing CPU
sparse tensors after online segment growth. The cache below keeps the same
segment semantics using padded source/permanence tables:

    seg_overlap = sum(x[source] * permanence)   (total_segments,)
    seg_active  = sigmoid((overlap - threshold) / temp)
    cell_depol  = scatter_max(seg_active, seg_to_cell)  (n_cells,)

The cache is rebuilt lazily after mutations (grow, learn, prune) and keeps
the hot path vectorized on both CPU and CUDA without relying on sparse COO.
"""

from __future__ import annotations

import torch


class SparseDendrites:
    """Dendritic segments implemented with a cached connected-synapse table.

    Each cell can have multiple segments. Each segment connects to a subset
    of cells via synapses with permanence values. Prediction = weighted
    source gather followed by a per-cell max across segments.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    max_segments_per_cell : int
        Maximum segments any cell can grow.
    max_synapses_per_segment : int
        Maximum synapses per segment.
    activation_threshold : float
        Overlap threshold for segment activation.
    sigmoid_temp : float
        Temperature for soft sigmoid activation.
    connected_threshold : float
        Permanence threshold for a synapse to be "connected".
    permanence_increment : float
        Learning increment for active synapses.
    permanence_decrement : float
        Learning decrement for inactive synapses.
    initial_permanence : float
        Initial permanence for new synapses.
    seed : int
        Random seed.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        max_segments_per_cell: int = 32,
        max_synapses_per_segment: int = 24,
        activation_threshold: float = 0.3,
        sigmoid_temp: float = 0.1,
        connected_threshold: float = 0.3,
        permanence_increment: float = 0.05,
        permanence_decrement: float = 0.02,
        initial_permanence: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment
        self.activation_threshold = activation_threshold
        self.sigmoid_temp = sigmoid_temp
        self.connected_threshold = connected_threshold
        self.perm_inc = permanence_increment
        self.perm_dec = permanence_decrement
        self.initial_permanence = initial_permanence
        self.device = torch.device(device)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

        # Raw segment storage for mutations:
        # list of (parent_cell, {source_cell: permanence})
        self._segments: list[tuple[int, dict[int, float]]] = []
        self._segment_count_per_cell = torch.zeros(
            n_cells, dtype=torch.int32, device="cpu"
        )

        # Cached connected-synapse tables for fast prediction — rebuilt lazily.
        self._D: torch.Tensor | None = None
        self._segment_sources: torch.Tensor | None = None
        self._segment_permanences: torch.Tensor | None = None
        self._seg_to_cell: torch.Tensor | None = None  # (n_seg,)
        self._dirty = True

    @property
    def total_segments(self) -> int:
        return len(self._segments)

    def _invalidate_cache(self) -> None:
        """Mark the connected-synapse cache as stale after a mutation."""
        self._dirty = True

    def _rebuild_sparse(self) -> None:
        """Rebuild the connected-synapse cache from raw segments.

        Only synapses with permanence >= connected_threshold are included,
        mirroring biological synapse maturation. The historical method name
        stays in place because callers already use it during cache rebuilds.
        """
        if not self._dirty:
            return

        if not self._segments:
            self._D = None
            self._segment_sources = None
            self._segment_permanences = None
            self._seg_to_cell = None
            self._dirty = False
            return

        n_seg = len(self._segments)
        self._segment_sources = torch.zeros(
            (n_seg, self.max_synapses_per_segment),
            dtype=torch.long,
            device=self.device,
        )
        self._segment_permanences = torch.zeros(
            (n_seg, self.max_synapses_per_segment),
            dtype=torch.float32,
            device=self.device,
        )
        seg_to_cell: list[int] = []

        for seg_idx, (parent_cell, synapses) in enumerate(self._segments):
            seg_to_cell.append(parent_cell)
            connected_synapses = [
                (src, perm)
                for src, perm in synapses.items()
                if perm >= self.connected_threshold
            ]
            if not connected_synapses:
                continue

            for synapse_index, (src, perm) in enumerate(connected_synapses):
                self._segment_sources[seg_idx, synapse_index] = int(src)
                self._segment_permanences[seg_idx, synapse_index] = float(perm)

        self._seg_to_cell = torch.tensor(
            seg_to_cell, dtype=torch.long, device=self.device,
        )
        self._D = None
        self._dirty = False

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute graded depolarization for each cell via cached gathers.

        Pipeline:
            seg_overlap = sum(x[source] * permanence)   (total_segments,)
            seg_active  = sigmoid((overlap - threshold) / temp)
            cell_depol  = scatter_max(seg_active, seg_to_cell)  (n_cells,)

        Parameters
        ----------
        x : Tensor (n_cells,)
            Current activation pattern.

        Returns
        -------
        Tensor (n_cells,) of graded depolarization in [0, 1].
        """
        if not self._segments:
            return torch.zeros(
                self.n_cells, dtype=torch.float32, device=self.device,
            )

        self._rebuild_sparse()

        x_dev = x.to(self.device)

        source_activations = x_dev.index_select(
            0,
            self._segment_sources.reshape(-1),
        ).reshape_as(self._segment_sources)
        seg_overlap = torch.sum(
            source_activations * self._segment_permanences,
            dim=1,
        )

        # Graded sigmoid activation (not binary)
        seg_active = torch.sigmoid(
            (seg_overlap - self.activation_threshold)
            / max(self.sigmoid_temp, 1e-6)
        )

        # Vectorized scatter-max: sort activations ascending so that
        # scatter_ (which keeps the last write per index) retains the
        # maximum value for each parent cell. O(S log S) sort + O(S)
        # scatter — no Python loops, fully GPU-compatible.
        depol = torch.zeros(
            self.n_cells, dtype=torch.float32, device=self.device,
        )
        sorted_active, sort_idx = seg_active.sort()
        sorted_cells = self._seg_to_cell[sort_idx]
        depol.scatter_(0, sorted_cells, sorted_active)

        return depol

    def grow_segment(
        self,
        cell: int,
        source_cells: list[int],
    ) -> None:
        """Grow a new dendritic segment on *cell* connected to *source_cells*."""
        if int(self._segment_count_per_cell[cell]) >= self.max_segments_per_cell:
            return

        synapses = {}
        for src in source_cells[:self.max_synapses_per_segment]:
            synapses[src] = self.initial_permanence

        self._segments.append((cell, synapses))
        self._segment_count_per_cell[cell] += 1
        self._invalidate_cache()

    def learn(
        self,
        active_cells: torch.Tensor,
        prev_active_cells: torch.Tensor,
        learning_rate: float = 1.0,
    ) -> None:
        """Strengthen/weaken synapses on segments that were correctly predicted.

        For each segment whose parent cell is now active:
        - Strengthen synapses from previously-active cells
        - Weaken synapses from previously-inactive cells
        """
        active_set = set(torch.nonzero(active_cells, as_tuple=True)[0].tolist())
        prev_set = set(
            torch.nonzero(prev_active_cells, as_tuple=True)[0].tolist()
        )

        modified = False
        for parent_cell, synapses in self._segments:
            if parent_cell not in active_set:
                continue

            for src in list(synapses.keys()):
                if src in prev_set:
                    synapses[src] = min(
                        1.0, synapses[src] + self.perm_inc * learning_rate
                    )
                else:
                    synapses[src] = max(
                        0.0, synapses[src] - self.perm_dec * learning_rate
                    )
            modified = True

        if modified:
            self._invalidate_cache()

    def grow_for_unpredicted(
        self,
        active_cells: torch.Tensor,
        prev_active_cells: torch.Tensor,
        predicted_cells: torch.Tensor,
    ) -> None:
        """Grow new segments for active cells that were NOT predicted.

        These are "bursting" cells that need new temporal context connections.
        """
        active_idx = torch.nonzero(active_cells, as_tuple=True)[0].tolist()
        predicted_idx = set(
            torch.nonzero(predicted_cells, as_tuple=True)[0].tolist()
        )
        prev_idx = torch.nonzero(prev_active_cells, as_tuple=True)[0].tolist()

        if not prev_idx:
            return

        for cell in active_idx:
            if cell in predicted_idx:
                continue
            # Pick random subset of previous active cells as sources
            n = min(len(prev_idx), self.max_synapses_per_segment)
            perm = torch.randperm(len(prev_idx), generator=self._rng)[:n]
            sources = [prev_idx[i] for i in perm.tolist()]
            self.grow_segment(cell, sources)

    def prune(self, min_permanence: float = 0.01) -> int:
        """Remove segments with all permanences below threshold."""
        new_segments = []
        removed = 0
        self._segment_count_per_cell.zero_()

        for parent_cell, synapses in self._segments:
            # Remove dead synapses
            alive = {s: p for s, p in synapses.items() if p >= min_permanence}
            if alive:
                new_segments.append((parent_cell, alive))
                self._segment_count_per_cell[parent_cell] += 1
            else:
                removed += 1

        self._segments = new_segments
        if removed > 0:
            self._invalidate_cache()
        return removed

    def state_dict(self) -> dict:
        return {
            "segments": list(self._segments),
            "segment_count": self._segment_count_per_cell.clone(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self._segments = list(sd["segments"])
        self._segment_count_per_cell = sd["segment_count"].clone()
        self._invalidate_cache()
