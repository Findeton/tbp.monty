# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hetero-associative memory for biologically plausible object recognition.

Replaces the SDRObjectMemory lookup table with a fixed-size weight matrix that
associates cell activation patterns with object label SDRs via Hebbian learning.

In a real cortical column, synaptic weights ARE the memory. Learning a new object
modifies the same weights that store existing objects. The column has a fixed number
of neurons and synapses — capacity is bounded by how many distinct patterns fit
in the weight space, not by explicit storage allocation.

This module implements:

- **Label SDRs**: Each object gets a deterministic sparse label from a hash of its
  name. Labels are fixed-size (n_label_bits) with a fixed number of active bits.

- **Dual-pathway Hebbian association**:
  - Cell pathway: maps settled cell activations → labels (context-rich)
  - Feedforward pathway: maps input SDRs → labels (viewpoint-robust)
  - Combined readout score = cell_weight * cell_score + ff_weight * ff_score

- **Fixed memory**: weight matrices are (n_cells × n_labels) and (n_input × n_labels).
  Neither grows with the number of objects learned.

- **Capacity**: with sparse representations (~3% active), theoretical capacity is
  ~N/(a*log(1/a)) patterns. For N=16384 cells, that's ~150,000 patterns — enough
  for thousands of objects at 40 patterns/object.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class HeteroAssociativeMemory:
    """Fixed-size weight-based object memory using Hebbian association.

    Maps cell activation patterns (after attractor settling) and/or feedforward
    input SDRs to object label SDRs. Recognition is a matrix multiply + label
    overlap — O(n_cells * n_labels + n_objects * n_labels), independent of how
    many observations were stored.

    Parameters
    ----------
    n_cells : int
        Total cells in the column (for cell pathway).
    n_input : int
        Total bits in the feedforward input SDR (for ff pathway).
    n_label_bits : int
        Width of the label SDR space.
    n_active_label : int
        Number of active bits per label SDR (~sparsity of labels).
    learning_rate : float
        Hebbian learning rate for weight updates.
    decay_rate : float
        Per-step weight decay to prevent saturation.
    cell_weight : float
        Weight for cell-pathway readout in combined score.
    ff_weight : float
        Weight for feedforward-pathway readout in combined score.
    seed : int
        Random seed for label generation.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        n_input: int = 2816,
        n_label_bits: int = 256,
        n_active_label: int = 10,
        learning_rate: float = 0.01,
        decay_rate: float = 0.0001,
        cell_weight: float = 0.5,
        ff_weight: float = 0.5,
        seed: int = 42,
    ):
        self._n_cells = n_cells
        self._n_input = n_input
        self._n_label_bits = n_label_bits
        self._n_active_label = n_active_label
        self._lr = learning_rate
        self._decay = decay_rate
        self._cell_weight = cell_weight
        self._ff_weight = ff_weight
        self._seed = seed

        # Fixed-size weight matrices — THE memory
        self._cell_weights = np.zeros(
            (n_cells, n_label_bits), dtype=np.float32
        )
        if ff_weight > 0:
            self._ff_weights = np.zeros(
                (n_input, n_label_bits), dtype=np.float32
            )
        else:
            self._ff_weights = None

        # Label cache: object_name → label SDR (deterministic from hash)
        # This dict grows but each entry is tiny (n_label_bits float32 ≈ 1KB)
        # and could be regenerated on-the-fly from the hash.
        self._labels: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def _get_label(self, object_name: str) -> np.ndarray:
        """Get or create a deterministic label SDR for an object.

        The label is generated from a hash of the object name, making it
        reproducible without explicit storage. We cache for performance.
        """
        if object_name not in self._labels:
            # Deterministic RNG seeded from object name hash
            name_hash = hash(object_name) % (2**31)
            label_rng = np.random.RandomState(
                (self._seed + name_hash) % (2**31)
            )
            label = np.zeros(self._n_label_bits, dtype=np.float32)
            active_idx = label_rng.choice(
                self._n_label_bits, size=self._n_active_label, replace=False
            )
            label[active_idx] = 1.0
            self._labels[object_name] = label
        return self._labels[object_name]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        active_mask: np.ndarray,
        input_sdr: np.ndarray | None = None,
        object_name: str = "",
    ) -> None:
        """Hebbian association: active pattern → object label.

        Parameters
        ----------
        active_mask : np.ndarray, shape (n_cells,), bool
            Cell activation pattern (after settling).
        input_sdr : np.ndarray, shape (n_input,), float or None
            Feedforward input SDR. Required if ff_weight > 0.
        object_name : str
            Object identifier for label generation.
        """
        if not object_name:
            return

        label = self._get_label(object_name)
        cell_f = active_mask.astype(np.float32)

        # Cell pathway: outer product Hebbian
        # W += lr * cell ⊗ label
        # Only update rows where cell is active (sparse update)
        active_idx = np.where(cell_f > 0.5)[0]
        if len(active_idx) > 0:
            self._cell_weights[active_idx] += self._lr * label[np.newaxis, :]

        # Feedforward pathway
        if self._ff_weights is not None and input_sdr is not None:
            input_f = input_sdr.astype(np.float32)
            active_input = np.where(input_f > 0.5)[0]
            if len(active_input) > 0:
                self._ff_weights[active_input] += (
                    self._lr * label[np.newaxis, :]
                )

        # Clip to prevent unbounded growth
        np.clip(self._cell_weights, 0.0, 1.0, out=self._cell_weights)
        if self._ff_weights is not None:
            np.clip(self._ff_weights, 0.0, 1.0, out=self._ff_weights)

    def decay(self) -> None:
        """Global weight decay to prevent saturation."""
        if self._decay > 0:
            self._cell_weights = np.maximum(
                0.0, self._cell_weights - self._decay
            )
            if self._ff_weights is not None:
                self._ff_weights = np.maximum(
                    0.0, self._ff_weights - self._decay
                )

    # ------------------------------------------------------------------
    # Recall / Recognition
    # ------------------------------------------------------------------

    def recall(
        self,
        active_mask: np.ndarray,
        input_sdr: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Read out object evidence from current activation pattern.

        Projects the cell activation (and optionally feedforward SDR) through
        learned weight matrices into label space, then computes overlap with
        each known object's label pattern.

        Parameters
        ----------
        active_mask : np.ndarray, shape (n_cells,), bool
            Current cell activation (after settling).
        input_sdr : np.ndarray, shape (n_input,), float or None
            Feedforward input SDR. Used if ff_weight > 0.

        Returns
        -------
        dict[str, float]
            Object name → match score. Higher = better match.
        """
        if not self._labels:
            return {}

        scores = {}
        cell_f = active_mask.astype(np.float32)

        # Cell pathway readout
        cell_readout = cell_f @ self._cell_weights  # (n_label_bits,)

        # Feedforward pathway readout
        ff_readout = np.zeros(self._n_label_bits, dtype=np.float32)
        if (
            self._ff_weights is not None
            and input_sdr is not None
            and self._ff_weight > 0
        ):
            input_f = input_sdr.astype(np.float32)
            ff_readout = input_f @ self._ff_weights  # (n_label_bits,)

        # Combined readout
        combined = (
            self._cell_weight * cell_readout + self._ff_weight * ff_readout
        )

        # Score against each known label
        for obj_name, label in self._labels.items():
            overlap = float(np.dot(combined, label))
            # Normalize by label active count for cross-object comparability
            n_active = max(float(label.sum()), 1.0)
            scores[obj_name] = overlap / n_active

        return scores

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def known_objects(self) -> list:
        """List of known object names."""
        return list(self._labels.keys())

    def memory_bytes(self) -> int:
        """Total memory usage in bytes (fixed regardless of objects)."""
        total = self._cell_weights.nbytes
        if self._ff_weights is not None:
            total += self._ff_weights.nbytes
        # Labels are tiny and could be regenerated, but count them
        for label in self._labels.values():
            total += label.nbytes
        return total

    def weight_matrix_bytes(self) -> int:
        """Memory used by weight matrices only (the fixed part)."""
        total = self._cell_weights.nbytes
        if self._ff_weights is not None:
            total += self._ff_weights.nbytes
        return total

    @property
    def total_weight(self) -> float:
        """Sum of all weights (measure of learned structure)."""
        total = float(self._cell_weights.sum())
        if self._ff_weights is not None:
            total += float(self._ff_weights.sum())
        return total

    def clear(self) -> None:
        """Reset all learned associations."""
        self._cell_weights[:] = 0.0
        if self._ff_weights is not None:
            self._ff_weights[:] = 0.0
        self._labels.clear()

    def remove_object(self, object_name: str) -> None:
        """Remove a specific object's label (weights are shared, not removed).

        Note: this removes the label so the object won't appear in recall
        results, but the weights that were learned for this object remain
        in the matrix. This is biologically correct — you can't selectively
        erase synaptic modifications for one memory.
        """
        self._labels.pop(object_name, None)
