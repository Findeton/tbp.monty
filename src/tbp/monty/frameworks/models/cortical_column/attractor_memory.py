# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Attractor-based object memory for cortical column recognition.

Replaces SDRObjectMemory's brute-force snapshot matching with lightweight
prototype SDRs. After recurrent settling converges to an attractor, the
settled pattern is compared against stored prototypes — one per object —
giving O(K) recognition instead of O(K*N).

Prototypes are maintained as running averages: each bit stores the fraction
of training observations where that cell was active. During matching, the
prototype is thresholded to produce a binary pattern for overlap computation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class AttractorMemory:
    """Stores object prototypes for attractor-based recognition.

    Each object is represented by a single prototype vector: the average
    activation pattern across all training observations. Recognition is
    overlap between the current (settled) activation and each prototype.

    Parameters
    ----------
    n_cells : int
        Total cells in the column.
    prototype_threshold : float
        Fraction above which a prototype bit is considered "on".
        Bits active in >threshold of training observations are in the prototype.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        prototype_threshold: float = 0.3,
    ):
        self.n_cells = n_cells
        self._threshold = prototype_threshold

        # object_name -> {"sum": float32 array, "count": int}
        self._accumulators: Dict[str, dict] = {}
        # Cached binary prototypes (invalidated on update)
        self._prototypes: Dict[str, np.ndarray] = {}

    @property
    def known_objects(self) -> list:
        """List of known object names."""
        return list(self._accumulators.keys())

    def n_observations(self, object_name: str) -> int:
        """Number of observations stored for an object."""
        if object_name in self._accumulators:
            return self._accumulators[object_name]["count"]
        return 0

    def store(self, object_name: str, active_mask: np.ndarray) -> None:
        """Accumulate an observation into the object's prototype.

        Parameters
        ----------
        object_name : str
            Object identifier.
        active_mask : np.ndarray, shape (n_cells,), bool or float
            Cell activation pattern.
        """
        active_f = active_mask.astype(np.float32)

        if object_name not in self._accumulators:
            self._accumulators[object_name] = {
                "sum": np.zeros(self.n_cells, dtype=np.float32),
                "count": 0,
            }

        self._accumulators[object_name]["sum"] += active_f
        self._accumulators[object_name]["count"] += 1

        # Invalidate cached prototype
        self._prototypes.pop(object_name, None)

    def get_prototype(self, object_name: str) -> np.ndarray:
        """Get the binary prototype for an object.

        Returns
        -------
        np.ndarray, shape (n_cells,), bool
            Binary prototype (bits active in >threshold fraction of observations).
        """
        if object_name in self._prototypes:
            return self._prototypes[object_name]

        if object_name not in self._accumulators:
            return np.zeros(self.n_cells, dtype=np.bool_)

        acc = self._accumulators[object_name]
        freq = acc["sum"] / max(acc["count"], 1)
        proto = freq >= self._threshold
        self._prototypes[object_name] = proto
        return proto

    def match(self, active_mask: np.ndarray) -> Dict[str, float]:
        """Match current activation against all stored prototypes.

        Parameters
        ----------
        active_mask : np.ndarray, shape (n_cells,), bool
            Current cell activation (after settling).

        Returns
        -------
        dict[str, float]
            Object name -> overlap score (fraction of active bits that
            overlap with prototype).
        """
        scores = {}
        n_active = max(float(active_mask.sum()), 1.0)
        active_f = active_mask.astype(np.float32)

        for obj_name in self._accumulators:
            proto = self.get_prototype(obj_name)
            overlap = float(np.dot(active_f, proto.astype(np.float32)))
            scores[obj_name] = overlap / n_active

        return scores

    def match_location_feature(
        self,
        location_sdr: np.ndarray,
        feature_sdr: np.ndarray,
        location_prototypes: dict = None,
        feature_prototypes: dict = None,
    ) -> Dict[str, float]:
        """Match using separate location and feature SDRs.

        Falls back to simple overlap if prototypes aren't available.
        This provides backward compatibility with the eval harness which
        passes location/feature SDRs separately.

        Parameters
        ----------
        location_sdr : np.ndarray
            Current location encoding.
        feature_sdr : np.ndarray
            Current feature encoding.
        location_prototypes : dict or None
            Object -> location prototype arrays.
        feature_prototypes : dict or None
            Object -> feature prototype arrays.

        Returns
        -------
        dict[str, float]
            Object name -> match score.
        """
        if location_prototypes is None or feature_prototypes is None:
            # No separate prototypes — return zero (use cell-based matching)
            return {obj: 0.0 for obj in self._accumulators}

        scores = {}
        loc_n = max(float(location_sdr.sum()), 1.0)
        feat_n = max(float(feature_sdr.sum()), 1.0)

        for obj_name in self._accumulators:
            loc_proto = location_prototypes.get(obj_name)
            feat_proto = feature_prototypes.get(obj_name)
            if loc_proto is None or feat_proto is None:
                scores[obj_name] = 0.0
                continue

            loc_overlap = float(np.dot(location_sdr, loc_proto)) / loc_n
            feat_overlap = float(np.dot(feature_sdr, feat_proto)) / feat_n
            scores[obj_name] = loc_overlap * feat_overlap

        return scores

    def clear(self) -> None:
        """Remove all stored objects."""
        self._accumulators.clear()
        self._prototypes.clear()

    def remove_object(self, object_name: str) -> None:
        """Remove a specific object."""
        self._accumulators.pop(object_name, None)
        self._prototypes.pop(object_name, None)

    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        total = 0
        for acc in self._accumulators.values():
            total += acc["sum"].nbytes  # float32 accumulator
        for proto in self._prototypes.values():
            total += proto.nbytes  # bool prototype
        return total
