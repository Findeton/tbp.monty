# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Vectorized dendritic segments for context-dependent prediction.

Each cell in the cortical column has multiple dendritic segments, each
independently capable of recognizing a pattern of previously active cells.

All segment/synapse data is stored in dense NumPy arrays for fast batch
computation. The key data structures:

- ``connections[cell, seg, syn]`` — presynaptic cell index (-1 = empty)
- ``permanences[cell, seg, syn]`` — synapse permanence in [0, 1]
- ``segment_count[cell]`` — how many segments a cell has grown

Prediction is a single vectorized pass: gather source activations via
fancy indexing, threshold on permanence, sum per segment, threshold on
activation count.
"""

from __future__ import annotations

import numpy as np


class DendriteSegments:
    """Vectorized dendritic segment machinery for the cortical column.

    Parameters
    ----------
    n_cells : int
        Total number of cells in the column.
    max_segments_per_cell : int
        Maximum dendritic segments per cell.
    max_synapses_per_segment : int
        Maximum synapses per segment.
    activation_threshold : int
        Minimum connected active synapses for a segment to activate.
    initial_permanence : float
        Permanence for newly created synapses.
    connected_threshold : float
        Minimum permanence to count as "connected".
    permanence_increment : float
        Amount to increase permanence on correct prediction.
    permanence_decrement : float
        Amount to decrease permanence on incorrect prediction.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        max_segments_per_cell: int = 32,
        max_synapses_per_segment: int = 24,
        activation_threshold: int = 12,
        initial_permanence: float = 0.5,
        connected_threshold: float = 0.3,
        permanence_increment: float = 0.05,
        permanence_decrement: float = 0.02,
        graded_activation: bool = False,
        graded_temperature: float = 2.0,
        pruning_threshold: float = 0.01,
        pruning_age: int = 50,
        seed: int = 42,
    ):
        self.n_cells = n_cells
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment
        self.activation_threshold = activation_threshold
        self.initial_permanence = initial_permanence
        self.connected_threshold = connected_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self._graded = graded_activation
        self._temperature = graded_temperature
        self._pruning_threshold = pruning_threshold
        self._pruning_age = pruning_age
        self._rng = np.random.RandomState(seed)

        # Dense 3D arrays for all synapse data.
        # connections[c, s, syn] = presynaptic cell index, -1 = empty slot
        self._connections = np.full(
            (n_cells, max_segments_per_cell, max_synapses_per_segment),
            -1,
            dtype=np.int32,
        )
        # permanences[c, s, syn] = synapse permanence ∈ [0, 1]
        self._permanences = np.zeros(
            (n_cells, max_segments_per_cell, max_synapses_per_segment),
            dtype=np.float32,
        )
        # How many segments each cell has grown so far
        self._segment_count = np.zeros(n_cells, dtype=np.int32)
        # How many synapses each segment has
        self._synapse_count = np.zeros(
            (n_cells, max_segments_per_cell), dtype=np.int32
        )
        # Steps since last segment activation (for pruning)
        self._segment_age = np.zeros(
            (n_cells, max_segments_per_cell), dtype=np.int32
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_segments(self) -> int:
        """Total number of segments across all cells."""
        return int(self._segment_count.sum())

    def n_segments(self, cell_idx: int) -> int:
        """Number of segments on a specific cell."""
        return int(self._segment_count[cell_idx])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _compute_segment_overlaps(
        self, active_mask_padded: np.ndarray
    ):
        """Core vectorized computation shared by prediction and active segments.

        Parameters
        ----------
        active_mask_padded : np.ndarray, shape (n_cells + 1,), bool
            Boolean mask of active cells, with a False sentinel at index -1.

        Returns
        -------
        cell_indices : np.ndarray (int)
            Indices of cells that have at least one segment.
        overlap : np.ndarray, shape (len(cell_indices), max_segs)
            Number of connected active synapses per segment.
        """
        has_segs = self._segment_count > 0
        if not np.any(has_segs):
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        cell_indices = np.where(has_segs)[0]

        conns = self._connections[cell_indices]   # (N, S, Y)
        perms = self._permanences[cell_indices]   # (N, S, Y)

        # A synapse contributes if: connected (perm >= threshold) AND source active
        connected = perms >= self.connected_threshold
        source_active = active_mask_padded[conns]  # -1 → last elem (False)

        overlap = (connected & source_active).sum(axis=2)  # (N, S)

        return cell_indices, overlap

    def _make_active_mask(self, active_cells) -> np.ndarray:
        """Convert set or array to padded bool mask (n_cells+1,).

        The extra element at index -1 is always False, so that
        ``mask[connections]`` returns False for empty synapse slots (-1).
        """
        mask = np.zeros(self.n_cells + 1, dtype=np.bool_)
        if isinstance(active_cells, set):
            if active_cells:
                idx = np.fromiter(active_cells, dtype=np.int32)
                mask[idx] = True
        elif isinstance(active_cells, np.ndarray):
            if active_cells.dtype == np.bool_:
                mask[:self.n_cells] = active_cells
            else:
                mask[active_cells] = True
        return mask

    def compute_predicted_cells(self, active_cells) -> set:
        """Determine which cells are predicted by the current active cells.

        Parameters
        ----------
        active_cells : set[int] or np.ndarray
            Currently active cell indices (set) or boolean mask (n_cells,).

        Returns
        -------
        set[int]
            Cell indices that are predicted (depolarized).
        """
        if isinstance(active_cells, set) and not active_cells:
            return set()

        active_mask = self._make_active_mask(active_cells)
        cell_indices, overlap = self._compute_segment_overlaps(active_mask)

        if len(cell_indices) == 0:
            return set()

        seg_active = overlap >= self.activation_threshold
        predicted = seg_active.any(axis=1)

        return set(cell_indices[predicted].tolist())

    def compute_predicted_cells_mask(self, active_mask_padded: np.ndarray) -> np.ndarray:
        """Vectorized prediction returning a boolean mask.

        Parameters
        ----------
        active_mask_padded : np.ndarray, shape (n_cells + 1,), bool
            Padded active mask (index -1 = False sentinel).

        Returns
        -------
        np.ndarray, shape (n_cells,), bool
            True for predicted cells.
        """
        result = np.zeros(self.n_cells, dtype=np.bool_)

        cell_indices, overlap = self._compute_segment_overlaps(active_mask_padded)
        if len(cell_indices) == 0:
            return result

        if self._graded:
            # Graded: sigmoid activation, threshold at 0.5
            x = (overlap.astype(np.float32) - self.activation_threshold) / max(
                self._temperature, 0.01
            )
            seg_activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
            predicted = seg_activation.max(axis=1) >= 0.5
        else:
            seg_active = overlap >= self.activation_threshold
            predicted = seg_active.any(axis=1)

        result[cell_indices[predicted]] = True
        return result

    def compute_graded_depolarization(
        self, active_mask_padded: np.ndarray
    ) -> np.ndarray:
        """Compute graded (continuous) depolarization per cell.

        Instead of binary threshold, uses a sigmoid function so sub-threshold
        segments contribute partial depolarization. Multiple partial activations
        can sum to produce a prediction.

        Parameters
        ----------
        active_mask_padded : np.ndarray, shape (n_cells + 1,), bool
            Padded active mask.

        Returns
        -------
        np.ndarray, shape (n_cells,), float32
            Depolarization level per cell in [0, 1].
        """
        result = np.zeros(self.n_cells, dtype=np.float32)

        cell_indices, overlap = self._compute_segment_overlaps(active_mask_padded)
        if len(cell_indices) == 0:
            return result

        # Sigmoid: 1 / (1 + exp(-(overlap - threshold) / temperature))
        x = (overlap.astype(np.float32) - self.activation_threshold) / max(
            self._temperature, 0.01
        )
        seg_activation = 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

        # Per-cell: max activation across segments
        cell_depol = seg_activation.max(axis=1)
        result[cell_indices] = cell_depol
        return result

    def get_active_segments(self, active_cells) -> dict:
        """Return which segments are active for which cells.

        Parameters
        ----------
        active_cells : set[int] or np.ndarray
            Currently active cell indices or boolean mask.

        Returns
        -------
        dict[int, list[int]]
            Maps cell_idx → list of active segment indices.
        """
        if isinstance(active_cells, set) and not active_cells:
            return {}

        active_mask = self._make_active_mask(active_cells)
        cell_indices, overlap = self._compute_segment_overlaps(active_mask)

        if len(cell_indices) == 0:
            return {}

        seg_active = overlap >= self.activation_threshold

        result = {}
        for i, cell_idx in enumerate(cell_indices):
            active_seg_indices = np.where(seg_active[i])[0]
            if len(active_seg_indices) > 0:
                result[int(cell_idx)] = active_seg_indices.tolist()

        return result

    def get_active_segments_from_mask(
        self, active_mask_padded: np.ndarray
    ) -> dict:
        """Same as get_active_segments but takes a pre-built padded mask."""
        cell_indices, overlap = self._compute_segment_overlaps(active_mask_padded)

        if len(cell_indices) == 0:
            return {}

        seg_active = overlap >= self.activation_threshold

        result = {}
        for i, cell_idx in enumerate(cell_indices):
            active_seg_indices = np.where(seg_active[i])[0]
            if len(active_seg_indices) > 0:
                result[int(cell_idx)] = active_seg_indices.tolist()

        return result

    # ------------------------------------------------------------------
    # Segment growth
    # ------------------------------------------------------------------

    def grow_segment(
        self,
        cell_idx: int,
        source_cells,
        n_synapses: int = None,
    ) -> bool:
        """Grow a new dendritic segment on a cell.

        Parameters
        ----------
        cell_idx : int
            Cell to grow the segment on.
        source_cells : set[int] or array-like
            Pool of cells to connect synapses to.
        n_synapses : int or None
            Number of synapses to create.

        Returns
        -------
        bool
            True if segment was created.
        """
        if n_synapses is None:
            n_synapses = self.max_synapses_per_segment

        seg_idx = int(self._segment_count[cell_idx])
        if seg_idx >= self.max_segments_per_cell:
            return False

        if isinstance(source_cells, set):
            source_list = sorted(source_cells - {cell_idx})
        else:
            source_list = [c for c in source_cells if c != cell_idx]

        if not source_list:
            return False

        n_syn = min(n_synapses, len(source_list))
        chosen = self._rng.choice(source_list, size=n_syn, replace=False)

        self._connections[cell_idx, seg_idx, :n_syn] = chosen
        self._permanences[cell_idx, seg_idx, :n_syn] = self.initial_permanence
        self._synapse_count[cell_idx, seg_idx] = n_syn
        self._segment_count[cell_idx] += 1

        return True

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def strengthen_segment(
        self,
        cell_idx: int,
        seg_idx: int,
        active_cells,
    ) -> None:
        """Strengthen synapses on an active segment (correct prediction).

        Increment permanence for synapses whose source was active.
        Decrement permanence for synapses whose source was inactive.
        """
        n_syn = int(self._synapse_count[cell_idx, seg_idx])
        if n_syn == 0:
            return

        conns = self._connections[cell_idx, seg_idx, :n_syn]
        perms = self._permanences[cell_idx, seg_idx, :n_syn]

        active_mask = self._make_active_mask(active_cells)
        is_active = active_mask[conns]

        perms = np.where(
            is_active,
            np.minimum(1.0, perms + self.permanence_increment),
            np.maximum(0.0, perms - self.permanence_decrement),
        )
        self._permanences[cell_idx, seg_idx, :n_syn] = perms

    def strengthen_segments_batch(
        self,
        cell_indices: np.ndarray,
        seg_indices: np.ndarray,
        active_mask_padded: np.ndarray,
    ) -> None:
        """Batch strengthen multiple segments with a pre-built active mask.

        Vectorized: processes all segments in one array operation instead of
        looping per segment. The padded mask's -1 sentinel handles empty
        synapse slots automatically.
        """
        if len(cell_indices) == 0:
            return

        conns = self._connections[cell_indices, seg_indices]   # (N, max_syn)
        perms = self._permanences[cell_indices, seg_indices]   # (N, max_syn)

        valid = conns >= 0                                     # (N, max_syn)
        is_active = active_mask_padded[conns]                  # -1 → False

        new_perms = np.where(
            is_active,
            np.minimum(1.0, perms + self.permanence_increment),
            np.maximum(0.0, perms - self.permanence_decrement),
        )
        # Only update valid synapses; leave padding unchanged
        self._permanences[cell_indices, seg_indices] = np.where(
            valid, new_perms, perms
        )

    def punish_segment(self, cell_idx: int, seg_idx: int) -> None:
        """Weaken all synapses on a segment (false prediction)."""
        n_syn = int(self._synapse_count[cell_idx, seg_idx])
        if n_syn == 0:
            return

        self._permanences[cell_idx, seg_idx, :n_syn] = np.maximum(
            0.0,
            self._permanences[cell_idx, seg_idx, :n_syn] - self.permanence_decrement,
        )

    def punish_segments_batch(
        self,
        cell_indices: np.ndarray,
        seg_indices: np.ndarray,
    ) -> None:
        """Batch punish multiple segments.

        Vectorized: processes all segments in one array operation.
        """
        if len(cell_indices) == 0:
            return

        conns = self._connections[cell_indices, seg_indices]
        perms = self._permanences[cell_indices, seg_indices]
        valid = conns >= 0

        new_perms = np.maximum(0.0, perms - self.permanence_decrement)
        self._permanences[cell_indices, seg_indices] = np.where(
            valid, new_perms, perms
        )

    def decay_all(self, rate: float = 0.001) -> None:
        """Apply global permanence decay. Only affects allocated synapses."""
        mask = self._connections >= 0
        self._permanences[mask] = np.maximum(
            0.0, self._permanences[mask] - rate
        )

    # ------------------------------------------------------------------
    # Structural Plasticity (Phase 6)
    # ------------------------------------------------------------------

    def add_synapses_to_segment(
        self, cell_idx: int, seg_idx: int, source_cells, max_new: int = 4,
    ) -> int:
        """Add new synapses to an existing segment from co-active cells.

        Only adds synapses to cells not already connected to this segment.

        Parameters
        ----------
        cell_idx : int
            Cell index.
        seg_idx : int
            Segment index.
        source_cells : set[int]
            Pool of candidate source cells.
        max_new : int
            Maximum synapses to add.

        Returns
        -------
        int
            Number of synapses actually added.
        """
        n_syn = int(self._synapse_count[cell_idx, seg_idx])
        available = self.max_synapses_per_segment - n_syn
        if available <= 0:
            return 0

        # Find cells not already connected
        existing = set(
            self._connections[cell_idx, seg_idx, :n_syn].tolist()
        )
        existing.discard(-1)
        existing.discard(cell_idx)
        candidates = sorted(source_cells - existing - {cell_idx})

        if not candidates:
            return 0

        n_add = min(max_new, available, len(candidates))
        chosen = self._rng.choice(candidates, size=n_add, replace=False)

        for i, src in enumerate(chosen):
            slot = n_syn + i
            self._connections[cell_idx, seg_idx, slot] = src
            self._permanences[cell_idx, seg_idx, slot] = self.initial_permanence

        self._synapse_count[cell_idx, seg_idx] += n_add
        return n_add

    def prune_dead_synapses(self, cell_idx: int, seg_idx: int) -> int:
        """Remove synapses with permanence at zero.

        Compacts the synapse array so empty slots are at the end.

        Returns
        -------
        int
            Number of synapses removed.
        """
        n_syn = int(self._synapse_count[cell_idx, seg_idx])
        if n_syn == 0:
            return 0

        conns = self._connections[cell_idx, seg_idx, :n_syn]
        perms = self._permanences[cell_idx, seg_idx, :n_syn]

        alive = perms > 0.0
        n_alive = int(alive.sum())
        n_removed = n_syn - n_alive

        if n_removed == 0:
            return 0

        # Compact: move alive synapses to front
        self._connections[cell_idx, seg_idx, :n_alive] = conns[alive]
        self._permanences[cell_idx, seg_idx, :n_alive] = perms[alive]
        # Clear remaining slots
        self._connections[cell_idx, seg_idx, n_alive:n_syn] = -1
        self._permanences[cell_idx, seg_idx, n_alive:n_syn] = 0.0
        self._synapse_count[cell_idx, seg_idx] = n_alive

        return n_removed

    def prune_old_segments(self) -> int:
        """Remove segments that haven't been active for pruning_age steps.

        A segment is pruned if:
        1. Its age exceeds pruning_age
        2. Its max permanence is below pruning_threshold

        Returns
        -------
        int
            Number of segments pruned.
        """
        n_pruned = 0
        for cell in range(self.n_cells):
            n_seg = int(self._segment_count[cell])
            if n_seg == 0:
                continue

            # Check each segment
            keep = []
            for seg in range(n_seg):
                age = int(self._segment_age[cell, seg])
                max_perm = float(self._permanences[cell, seg, :].max())
                if age >= self._pruning_age and max_perm < self._pruning_threshold:
                    n_pruned += 1
                else:
                    keep.append(seg)

            if len(keep) < n_seg:
                # Compact: shift kept segments to front
                for new_idx, old_idx in enumerate(keep):
                    if new_idx != old_idx:
                        self._connections[cell, new_idx] = self._connections[
                            cell, old_idx
                        ]
                        self._permanences[cell, new_idx] = self._permanences[
                            cell, old_idx
                        ]
                        self._synapse_count[cell, new_idx] = self._synapse_count[
                            cell, old_idx
                        ]
                        self._segment_age[cell, new_idx] = self._segment_age[
                            cell, old_idx
                        ]
                # Clear removed slots
                for idx in range(len(keep), n_seg):
                    self._connections[cell, idx] = -1
                    self._permanences[cell, idx] = 0.0
                    self._synapse_count[cell, idx] = 0
                    self._segment_age[cell, idx] = 0
                self._segment_count[cell] = len(keep)

        return n_pruned

    def age_segments(self, active_mask_padded: np.ndarray = None) -> None:
        """Increment age of all segments; reset active ones to zero.

        Parameters
        ----------
        active_mask_padded : np.ndarray or None
            If provided, segments that are currently active get their age
            reset. Otherwise all ages increment.
        """
        # Increment all
        has_segs = self._segment_count > 0
        if not has_segs.any():
            return
        # Only increment segments that actually exist
        for cell in np.where(has_segs)[0]:
            n_seg = int(self._segment_count[cell])
            self._segment_age[cell, :n_seg] += 1

        # Reset active segments
        if active_mask_padded is not None:
            cell_indices, overlap = self._compute_segment_overlaps(
                active_mask_padded
            )
            if len(cell_indices) > 0:
                seg_active = overlap >= self.activation_threshold
                for i, cell_idx in enumerate(cell_indices):
                    active_segs = np.where(seg_active[i])[0]
                    self._segment_age[cell_idx, active_segs] = 0
