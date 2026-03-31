# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Biologically plausible predictive tracking for reference frames.

Implements the cortical anchor-track-predict-surprise loop:

1. **Anchor**: Feature-only query identifies the best-matching stored
   pattern.  Its training location becomes the believed object-space
   position (analogous to grid cells locking onto a location).

2. **Track**: As the sensor moves, world-frame displacement updates the
   believed object-space location.  This is an approximation — it
   assumes identity rotation (world ≈ object frame).  Wrong for
   rotated objects, but self-correcting via step 4.

3. **Predict**: Given the believed object-space location, the LFM
   predicts what features should be observed (``query_location``).

4. **Surprise**: Compare predicted features with actual features.
   Low surprise → the tracking hypothesis is good, add prediction
   bonus to evidence.  High surprise → drop the anchor and re-anchor
   from features on the next step.

This mirrors how a biological cortical column uses:
- Grid cells for path integration (track)
- Learned associations for prediction (predict)
- Prediction error for learning and resetting (surprise)
- Feature-driven pattern completion for anchoring (anchor)

Unlike EvidenceGraphLM's multi-hypothesis tracker or SVD-based rotation
estimation, this is a single-hypothesis, biologically plausible loop.
Rotation estimation requires multiple cortical columns with known
spatial relationships — a single column recognizes objects by appearance
and prediction consistency, not by solving for rotation.
"""

from __future__ import annotations

import numpy as np


class PredictiveTracker:
    """Per-object anchor-track-predict state manager.

    Maintains one tracked object-space location per object during an
    eval episode.  The column drives the loop: anchor, track, predict,
    evaluate surprise, re-anchor if needed.

    Parameters
    ----------
    surprise_threshold : float
        Cosine similarity below which the prediction is considered
        surprising and the anchor is dropped.  Higher = stricter.
    anchor_evidence_threshold : float
        Minimum feature-only evidence for an object before anchoring.
        Prevents anchoring to noise early in the episode.
    prediction_bonus_weight : float
        Weight of prediction confirmation bonus added to evidence.
        Scales the cosine similarity when prediction matches.
    max_consecutive_drops : int
        After this many consecutive drops (high surprise), stop
        attempting to anchor for this object for the rest of the
        episode.  Prevents oscillation on inherently ambiguous objects.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.3,
        anchor_evidence_threshold: float = 0.1,
        prediction_bonus_weight: float = 0.5,
        max_consecutive_drops: int = 5,
    ):
        self.surprise_threshold = surprise_threshold
        self.anchor_evidence_threshold = anchor_evidence_threshold
        self.prediction_bonus_weight = prediction_bonus_weight
        self.max_consecutive_drops = max_consecutive_drops

        # Per-object state
        self._anchored_locs: dict[str, np.ndarray] = {}
        self._n_confirmed: dict[str, int] = {}
        self._n_consecutive_drops: dict[str, int] = {}
        self._retired: set[str] = set()  # objects that gave up anchoring

    def anchor(self, object_id: str, obj_loc: np.ndarray) -> None:
        """Set an anchor for an object at a predicted object-space location.

        This is analogous to grid cells locking onto a specific location
        in the object's reference frame.  Does NOT reset the consecutive
        drop counter — only a successful confirmation resets it.
        """
        if object_id in self._retired:
            return
        self._anchored_locs[object_id] = np.array(obj_loc, dtype=np.float64)
        self._n_confirmed[object_id] = 0

    def track(self, object_id: str, world_displacement: np.ndarray) -> None:
        """Update tracked location by world-frame displacement.

        This approximates object-frame displacement as world-frame
        displacement (identity rotation assumption).  Incorrect for
        rotated objects, but self-correcting via surprise detection.

        Biologically: path integration via grid cell phase update.
        """
        if object_id in self._anchored_locs:
            self._anchored_locs[object_id] += np.asarray(
                world_displacement, dtype=np.float64
            )

    def is_anchored(self, object_id: str) -> bool:
        """Whether this object has an active anchor."""
        return object_id in self._anchored_locs

    def get_location(self, object_id: str) -> np.ndarray:
        """Get the current tracked object-space location."""
        return self._anchored_locs[object_id]

    def confirm(self, object_id: str) -> None:
        """Record that prediction matched observation (low surprise)."""
        self._n_confirmed[object_id] = (
            self._n_confirmed.get(object_id, 0) + 1
        )
        self._n_consecutive_drops[object_id] = 0

    def drop(self, object_id: str) -> None:
        """Drop anchor due to high surprise (prediction mismatch).

        If this object has been dropped too many times consecutively,
        retire it — stop trying to anchor for the rest of the episode.
        """
        self._anchored_locs.pop(object_id, None)
        self._n_confirmed.pop(object_id, None)
        drops = self._n_consecutive_drops.get(object_id, 0) + 1
        self._n_consecutive_drops[object_id] = drops
        if drops >= self.max_consecutive_drops:
            self._retired.add(object_id)

    def get_n_confirmed(self, object_id: str) -> int:
        """How many consecutive steps this object's prediction was confirmed."""
        return self._n_confirmed.get(object_id, 0)

    def is_retired(self, object_id: str) -> bool:
        """Whether this object has been retired from anchoring attempts."""
        return object_id in self._retired

    def reset(self) -> None:
        """Clear all state (called at start of each eval episode)."""
        self._anchored_locs.clear()
        self._n_confirmed.clear()
        self._n_consecutive_drops.clear()
        self._retired.clear()

    @property
    def anchored_objects(self) -> list[str]:
        """Objects currently anchored."""
        return list(self._anchored_locs.keys())
