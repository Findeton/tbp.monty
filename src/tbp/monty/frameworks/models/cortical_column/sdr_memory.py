# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""SDR-based object memory for cortical column recognition.

Replaces the dense feature grid (GridObjectModel) with sparse distributed
representations. Each object is stored as a collection of SDR snapshots
taken during training — the set of cell activation patterns observed at
different locations on the object.

Recognition works by computing SDR overlap between the current observation
and stored patterns. High overlap with a stored pattern means "I've seen
this combination of features at this location before on object X."

Memory efficiency: ~50KB per object (N observations x ~500 bytes each)
vs ~200MB for a dense 50^3 x 16 feature grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StoredObservation:
    """A single stored observation from training.

    Attributes
    ----------
    location_sdr : np.ndarray
        Location encoding (grid cell SDR).
    feature_sdr : np.ndarray
        Sensory feature encoding.
    active_cells : np.ndarray
        Cell activation pattern (the full column state).
    """

    location_sdr: np.ndarray
    feature_sdr: np.ndarray
    active_cells: np.ndarray


class SDRObjectMemory:
    """Stores and matches object models using SDR representations.

    During training, the column observes an object from many viewpoints.
    At each viewpoint, the (location_sdr, feature_sdr, active_cells)
    triple is stored. During evaluation, the current observation is
    compared against all stored observations for all known objects.

    Parameters
    ----------
    location_overlap_threshold : float
        Minimum fraction of location SDR bits that must overlap to
        consider a stored observation as a location match. Range [0, 1].
    feature_overlap_threshold : float
        Minimum fraction of feature SDR bits that must overlap for a
        feature match. Range [0, 1].
    max_observations_per_object : int
        Maximum stored observations per object (oldest are dropped).
    """

    def __init__(
        self,
        location_overlap_threshold: float = 0.3,
        feature_overlap_threshold: float = 0.2,
        max_observations_per_object: int = 500,
    ):
        self._location_threshold = location_overlap_threshold
        self._feature_threshold = feature_overlap_threshold
        self._max_obs = max_observations_per_object

        # object_name -> list of StoredObservation
        self._memory: Dict[str, List[StoredObservation]] = {}

    @property
    def known_objects(self) -> list[str]:
        """Return list of known object IDs."""
        return list(self._memory.keys())

    def n_observations(self, object_name: str) -> int:
        """Number of stored observations for an object."""
        return len(self._memory.get(object_name, []))

    def store(
        self,
        object_name: str,
        location_sdr: np.ndarray,
        feature_sdr: np.ndarray,
        active_cells: np.ndarray,
    ) -> None:
        """Store an observation during training.

        Parameters
        ----------
        object_name : str
            Object identifier.
        location_sdr : np.ndarray
            Grid cell location encoding.
        feature_sdr : np.ndarray
            Feature encoding.
        active_cells : np.ndarray
            Full cell activation pattern.
        """
        if object_name not in self._memory:
            self._memory[object_name] = []

        obs = StoredObservation(
            location_sdr=location_sdr.astype(np.uint8),
            feature_sdr=feature_sdr.astype(np.uint8),
            active_cells=active_cells.astype(np.uint8),
        )
        self._memory[object_name].append(obs)

        # Trim if over limit
        if len(self._memory[object_name]) > self._max_obs:
            self._memory[object_name] = self._memory[object_name][-self._max_obs:]

    def match(
        self,
        location_sdr: np.ndarray,
        feature_sdr: np.ndarray,
    ) -> Dict[str, float]:
        """Match current observation against all stored objects.

        For each object, finds the best-matching stored observation
        (highest feature overlap at a similar location) and returns
        the match score.

        Parameters
        ----------
        location_sdr : np.ndarray
            Current location encoding.
        feature_sdr : np.ndarray
            Current feature encoding.

        Returns
        -------
        dict[str, float]
            Object name -> match score. Higher = better match.
            Score is the feature overlap fraction for the best location match.
        """
        scores = {}
        loc_n_active = max(float(np.sum(location_sdr)), 1.0)
        feat_n_active = max(float(np.sum(feature_sdr)), 1.0)

        for obj_name, observations in self._memory.items():
            best_score = 0.0

            for obs in observations:
                # Check location overlap
                loc_overlap = float(np.dot(location_sdr, obs.location_sdr))
                loc_frac = loc_overlap / loc_n_active

                if loc_frac < self._location_threshold:
                    continue

                # Check feature overlap
                feat_overlap = float(np.dot(feature_sdr, obs.feature_sdr))
                feat_frac = feat_overlap / feat_n_active

                # Combined score: location relevance * feature match
                score = loc_frac * feat_frac
                best_score = max(best_score, score)

            scores[obj_name] = best_score

        return scores

    def match_cells(
        self,
        active_cells: np.ndarray,
    ) -> Dict[str, float]:
        """Match current cell activation against stored cell patterns.

        This is a higher-level match that uses the full column state
        (which encodes both location and features contextually via
        dendritic predictions).

        Parameters
        ----------
        active_cells : np.ndarray
            Current cell activation pattern.

        Returns
        -------
        dict[str, float]
            Object name -> overlap score.
        """
        scores = {}
        n_active = max(float(np.sum(active_cells)), 1.0)

        for obj_name, observations in self._memory.items():
            best_overlap = 0.0
            for obs in observations:
                overlap = float(np.dot(active_cells, obs.active_cells))
                frac = overlap / n_active
                best_overlap = max(best_overlap, frac)
            scores[obj_name] = best_overlap

        return scores

    def clear(self) -> None:
        """Remove all stored objects."""
        self._memory.clear()

    def remove_object(self, object_name: str) -> None:
        """Remove a specific object from memory."""
        self._memory.pop(object_name, None)

    def memory_bytes(self) -> int:
        """Estimate total memory usage in bytes."""
        total = 0
        for observations in self._memory.values():
            for obs in observations:
                total += obs.location_sdr.nbytes
                total += obs.feature_sdr.nbytes
                total += obs.active_cells.nbytes
        return total
