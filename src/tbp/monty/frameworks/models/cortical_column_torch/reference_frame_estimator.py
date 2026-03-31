# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Reference frame estimation via Procrustes alignment.

During eval, the object may be at a different pose than during training.
Features (HSV, curvature) are rotation-invariant so feature-only queries
work without a reference frame.  To use location matching (which gives
sharper evidence), we need to estimate the world→object rotation.

Approach (mirrors EvidenceGraphLM's displacement-based hypothesis tracking):

1. **Bootstrap**: Feature-only queries produce per-object predicted
   object-space locations (attention-weighted average of stored training
   locations).  These give us (world_loc, predicted_obj_loc) pairs.

2. **Procrustes alignment**: With ≥ ``min_pairs`` observation pairs per
   object, estimate the best-fit rotation R (and translation t) via SVD
   that maps world locations to object-centric locations.

3. **Transform**: Once confident, transform new world locations into
   estimated object space, grid-cell encode them, and enable full
   (location + features) queries in the LocationFeatureMemory.
"""

from __future__ import annotations

import numpy as np


class ReferenceFrameEstimator:
    """Per-object rotation estimation via Procrustes SVD alignment.

    Collects (world_location, predicted_object_location) pairs from
    feature-weighted LFM retrieval, then estimates R, t such that::

        predicted_obj_loc ≈ R @ world_loc + t

    Parameters
    ----------
    min_pairs : int
        Minimum observation pairs before attempting rotation estimation.
        Need ≥ 3 non-collinear points for a well-constrained rotation.
    evidence_threshold : float
        Minimum per-object evidence to accept an observation pair.
        Low-evidence pairs are noisy and hurt the estimate.
    confidence_threshold : float
        Minimum alignment quality (1 - relative_residual) to consider
        a rotation estimate usable.
    max_pairs : int
        Maximum stored pairs per object (ring buffer).  Keeps memory
        bounded and lets the estimate adapt.
    """

    def __init__(
        self,
        min_pairs: int = 3,
        evidence_threshold: float = 0.15,
        confidence_threshold: float = 0.5,
        max_pairs: int = 30,
    ):
        self.min_pairs = min_pairs
        self.evidence_threshold = evidence_threshold
        self.confidence_threshold = confidence_threshold
        self.max_pairs = max_pairs

        # Per-object state
        self._world_locs: dict[str, list[np.ndarray]] = {}
        self._obj_locs: dict[str, list[np.ndarray]] = {}
        self._rotations: dict[str, np.ndarray] = {}      # (3, 3)
        self._translations: dict[str, np.ndarray] = {}   # (3,)
        self._confidence: dict[str, float] = {}

    def add_observation(
        self,
        object_id: str,
        world_loc: np.ndarray,
        predicted_obj_loc: np.ndarray,
        evidence: float,
    ) -> None:
        """Record a (world, predicted_object) location pair.

        Only pairs where the object has sufficient evidence are kept,
        to avoid polluting the estimate with noise from low-confidence
        objects.
        """
        if evidence < self.evidence_threshold:
            return

        w = np.asarray(world_loc, dtype=np.float64)
        o = np.asarray(predicted_obj_loc, dtype=np.float64)

        if object_id not in self._world_locs:
            self._world_locs[object_id] = []
            self._obj_locs[object_id] = []

        self._world_locs[object_id].append(w)
        self._obj_locs[object_id].append(o)

        # Ring buffer
        if len(self._world_locs[object_id]) > self.max_pairs:
            self._world_locs[object_id].pop(0)
            self._obj_locs[object_id].pop(0)

        if len(self._world_locs[object_id]) >= self.min_pairs:
            self._estimate_rotation(object_id)

    def _estimate_rotation(self, object_id: str) -> None:
        """Procrustes: find R, t minimizing ||R @ W + t - O||^2."""
        W = np.array(self._world_locs[object_id])  # (N, 3)
        O = np.array(self._obj_locs[object_id])     # (N, 3)

        # Center both point clouds
        w_mean = W.mean(axis=0)
        o_mean = O.mean(axis=0)
        W_c = W - w_mean
        O_c = O - o_mean

        # Check for degenerate configurations (collinear points)
        w_spread = np.linalg.norm(W_c, axis=1).max()
        if w_spread < 1e-8:
            return

        # SVD of cross-covariance: H = W_c^T @ O_c
        H = W_c.T @ O_c  # (3, 3)
        U, S, Vt = np.linalg.svd(H)

        # Ensure proper rotation (det = +1, not reflection)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T

        # Translation: t = o_mean - R @ w_mean
        t = o_mean - R @ w_mean

        # Confidence: 1 - (residual / spread)
        transformed = (R @ W.T).T + t
        residual = np.linalg.norm(transformed - O) / max(len(W), 1) ** 0.5
        spread = np.linalg.norm(O_c, axis=1).mean() + 1e-8
        confidence = max(0.0, 1.0 - residual / spread)

        self._rotations[object_id] = R
        self._translations[object_id] = t
        self._confidence[object_id] = confidence

    def has_rotation(self, object_id: str) -> bool:
        """Whether a sufficiently confident rotation exists for this object."""
        return (
            object_id in self._confidence
            and self._confidence[object_id] >= self.confidence_threshold
        )

    def get_confidence(self, object_id: str) -> float:
        """Current alignment confidence for an object (0 = none, 1 = perfect)."""
        return self._confidence.get(object_id, 0.0)

    def transform(
        self, object_id: str, world_loc: np.ndarray
    ) -> np.ndarray:
        """Transform a world-frame location to estimated object-centric frame.

        Raises KeyError if no rotation exists for this object.
        """
        R = self._rotations[object_id]
        t = self._translations[object_id]
        return R @ np.asarray(world_loc, dtype=np.float64) + t

    def reset(self) -> None:
        """Clear all accumulated observations and estimates."""
        self._world_locs.clear()
        self._obj_locs.clear()
        self._rotations.clear()
        self._translations.clear()
        self._confidence.clear()

    @property
    def estimated_objects(self) -> list[str]:
        """Objects with any rotation estimate (regardless of confidence)."""
        return list(self._rotations.keys())
