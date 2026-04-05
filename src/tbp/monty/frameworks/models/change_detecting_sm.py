# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Change-detecting sensor module for behavior modeling.

Analogous to the magnocellular visual pathway, this SM detects local changes
(movement, feature changes) and provides the input to behavior-learning LMs.
Once primed by an initial observation, it can emit low-confidence steady-state
context even when no thresholded change is detected. This keeps the behavior
stream anchored to absolute sensory context rather than reducing it to sparse
change events only.

Key features:
- Computes animation-induced flow using point correspondence between frames
- Ego-motion (camera movement) is implicitly compensated: world-coordinate
  points on a rigid static surface have identical coordinates across frames,
  so matched-pair displacement is zero regardless of camera motion
- Detects feature changes (color, curvature) even without motion
- Outputs change-describing features in CMP format
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from skimage.color import rgb2hsv

from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.utils.sensor_processing import (
    log_sign,
    principal_curvatures,
    surface_normal_naive,
)

logger = logging.getLogger(__name__)


class ChangeDetectingSM(SensorModule):
    """Sensor module that detects and reports local changes.

    Uses point correspondence between consecutive frames to compute
    animation-induced flow while naturally canceling ego-motion artifacts.
    World-coordinate points on a rigid surface have identical coordinates
    regardless of camera position, so nearest-neighbor matching between
    frames isolates genuine object motion from camera-induced visibility
    changes.

    Parameters
    ----------
    sensor_module_id : str
        Unique identifier for this SM.
    flow_threshold : float
        Minimum local flow magnitude (meters) to trigger change detection.
    feature_change_thresholds : dict or None
        Per-feature thresholds for detecting feature changes. Keys are feature
        names, values are threshold magnitudes. If None, only flow is used.
    global_flow_suppression : bool
        Legacy parameter, kept for API compatibility. The point-correspondence
        approach inherently compensates for ego-motion.
    correspondence_threshold : float
        Maximum distance (meters) between points in consecutive frames to be
        considered the same physical surface location. Points farther apart
        are treated as newly visible / newly occluded.
    min_persistent_points : int
        Minimum number of persistent (matched) points required to compute
        correspondence-based flow. Falls back to centroid flow if fewer.
    include_absolute_features : bool
        If True, include current absolute appearance features such as HSV in the
        emitted state in addition to change descriptors.
    emit_low_confidence_state_on_no_change : bool
        If True, emit a low-confidence state on quiet frames after the first
        observation rather than suppressing the state entirely.
    """

    def __init__(
        self,
        sensor_module_id: str,
        flow_threshold: float = 0.01,
        feature_change_thresholds: Optional[Dict[str, float]] = None,
        global_flow_suppression: bool = True,
        correspondence_threshold: float = 0.05,
        min_persistent_points: int = 5,
        include_absolute_features: bool = True,
        emit_low_confidence_state_on_no_change: bool = True,
    ):
        self.sensor_module_id = sensor_module_id
        self._flow_threshold = flow_threshold
        self._feature_change_thresholds = feature_change_thresholds or {}
        self._global_flow_suppression = global_flow_suppression
        self._correspondence_threshold = correspondence_threshold
        self._min_persistent_points = min_persistent_points
        self._include_absolute_features = bool(include_absolute_features)
        self._emit_low_confidence_state_on_no_change = bool(
            emit_low_confidence_state_on_no_change
        )

        self._prev_observation: Optional[Dict[str, Any]] = None
        self._prev_location: Optional[np.ndarray] = None
        self._prev_features: Optional[Dict[str, np.ndarray]] = None
        self._prev_point_cloud: Optional[np.ndarray] = None

    def state_dict(self) -> dict:
        """Return serializable state."""
        return {
            "sensor_module_id": self.sensor_module_id,
            "flow_threshold": self._flow_threshold,
            "include_absolute_features": self._include_absolute_features,
            "emit_low_confidence_state_on_no_change": (
                self._emit_low_confidence_state_on_no_change
            ),
        }

    def update_state(self, agent) -> None:
        """Update sensor state from agent. No-op for now."""
        pass

    def pre_episode(self) -> None:
        """Reset state between episodes."""
        self._prev_observation = None
        self._prev_location = None
        self._prev_features = None
        self._prev_point_cloud = None

    def step(self, ctx, observation, motor_only_step=False):
        """Process observation and detect changes.

        Returns a State with mixed absolute-plus-change features when a
        significant local change is detected. After the first observation, it
        can also emit a low-confidence state on quiet frames so the behavior LM
        retains stable identity context instead of seeing only sparse events.

        The observation should contain "semantic_3d" (N, 4) array with
        [x, y, z, semantic_id] for each point.

        Args:
            ctx: RuntimeContext.
            observation: Dict with at least "semantic_3d".
            motor_only_step: If True, skip processing.

        Returns:
            State with change features, or State with use_state=False.
        """
        if motor_only_step:
            return self._make_no_change_state()

        # Extract current 3D location and point cloud
        current_location = self._extract_center_location(observation)
        current_point_cloud = self._extract_on_object_points(observation)
        current_features = self._extract_features(observation)

        if current_location is None:
            self._prev_location = None
            self._prev_features = None
            self._prev_point_cloud = None
            return self._make_no_change_state()

        if self._prev_location is None:
            # First observation — store and report no change
            self._prev_location = current_location.copy()
            self._prev_features = self._copy_features(current_features)
            self._prev_point_cloud = current_point_cloud
            return self._make_no_change_state()

        # Compute flow using point correspondence
        flow_direction, flow_magnitude = self._compute_correspondence_flow(
            current_point_cloud, current_location
        )

        # Check feature changes
        feature_deltas = self._compute_feature_deltas(current_features)
        feature_changed = self._check_feature_changes(feature_deltas)

        # Determine if change is significant
        change_detected = (
            flow_magnitude > self._flow_threshold or feature_changed
        )
        signal_strength = self._compute_change_signal_strength(
            flow_magnitude,
            feature_deltas,
        )

        # Update previous state
        self._prev_location = current_location.copy()
        self._prev_features = self._copy_features(current_features)
        self._prev_point_cloud = current_point_cloud

        if change_detected:
            return self._make_change_state(
                location=current_location,
                current_features=current_features,
                flow_direction=flow_direction,
                flow_magnitude=flow_magnitude,
                feature_deltas=feature_deltas,
                signal_strength=signal_strength,
            )
        if self._emit_low_confidence_state_on_no_change:
            return self._make_quiet_state(
                location=current_location,
                current_features=current_features,
                flow_direction=flow_direction,
                flow_magnitude=flow_magnitude,
                feature_deltas=feature_deltas,
                signal_strength=signal_strength,
            )
        return self._make_no_change_state()

    # ======================== Private ========================

    def _extract_on_object_points(
        self, observation
    ) -> Optional[np.ndarray]:
        """Extract all on-object 3D points from observation.

        Returns:
            (M, 3) array of world-coordinate points, or None.
        """
        if "semantic_3d" not in observation:
            return None
        semantic_3d = observation["semantic_3d"]
        if semantic_3d is None or len(semantic_3d) == 0:
            return None
        on_object = semantic_3d[:, 3] > 0
        if not np.any(on_object):
            return None
        return semantic_3d[on_object, :3].copy()

    def _extract_center_location(self, observation) -> Optional[np.ndarray]:
        """Extract 3D location of center pixel from observation."""
        if "semantic_3d" not in observation:
            return None

        semantic_3d = observation["semantic_3d"]
        if semantic_3d is None or len(semantic_3d) == 0:
            return None

        on_object = semantic_3d[:, 3] > 0
        if not np.any(on_object):
            return None

        on_object_points = semantic_3d[on_object, :3]
        return on_object_points.mean(axis=0)

    def _extract_features(self, observation) -> Dict[str, np.ndarray]:
        """Extract absolute features from the current observation."""
        features = {}
        if "rgba" in observation:
            rgba = observation["rgba"]
            if isinstance(rgba, np.ndarray) and rgba.size > 0:
                # Use center pixel or mean
                if rgba.ndim == 3:
                    h, w = rgba.shape[:2]
                    center_rgba = rgba[h // 2, w // 2]
                else:
                    center_rgba = np.asarray(rgba)
                features["rgba"] = np.asarray(center_rgba, dtype=float)

                rgb = np.asarray(center_rgba).reshape(-1)[:3]
                if rgb.size == 3:
                    features["hsv"] = np.asarray(rgb2hsv(rgb), dtype=float)

        semantic_3d = observation.get("semantic_3d")
        if isinstance(semantic_3d, np.ndarray):
            features.update(self._extract_center_patch_geometry(semantic_3d))

        return features

    def _extract_center_patch_geometry(
        self,
        semantic_3d: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract center-patch geometry features when a square patch is available."""
        if semantic_3d.ndim != 2 or semantic_3d.shape[1] < 4:
            return {}

        n_points = int(semantic_3d.shape[0])
        if n_points <= 0:
            return {}

        obs_dim = int(np.sqrt(n_points))
        if obs_dim * obs_dim != n_points:
            return {}

        center_id = (obs_dim // 2) + obs_dim * (obs_dim // 2)
        if semantic_3d[center_id, 3] <= 0:
            return {}

        try:
            surface_normal, valid_sn = surface_normal_naive(semantic_3d)
            if not valid_sn:
                return {}

            k1, k2, _dir1, _dir2, valid_pc = principal_curvatures(
                semantic_3d,
                center_id,
                surface_normal,
                weighted=True,
            )
        except (FloatingPointError, IndexError, ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            return {}

        if not valid_pc:
            return {}

        return {
            "principal_curvatures_log": log_sign(
                np.asarray([k1, k2], dtype=float)
            )
        }

    def _compute_correspondence_flow(
        self, current_points, current_location
    ) -> tuple:
        """Compute animation-induced flow via point correspondence.

        For each point in the current frame, find its nearest neighbor in the
        previous frame's point cloud. Points within correspondence_threshold
        are "persistent" — the same physical surface location visible in both
        frames.

        Uses RMS of per-point displacement magnitudes as the flow magnitude.
        This captures **non-rigid deformation** even when net displacement is
        near zero (e.g., symmetric walking: left leg forward, right leg
        backward → mean ≈ 0 but RMS > 0). For a static rigid object under
        camera motion, persistent points have identical world coordinates →
        RMS ≈ 0 regardless of camera movement.

        The flow direction comes from the mean displacement vector, which
        gives the dominant direction of motion when present.

        Falls back to centroid displacement if too few persistent points are
        found (e.g., very large camera movement or total occlusion change).

        Returns:
            (flow_direction, flow_magnitude) — direction is unit vector,
            magnitude is RMS of per-point displacements in meters.
        """
        if (self._prev_point_cloud is None
                or current_points is None
                or len(current_points) == 0
                or len(self._prev_point_cloud) == 0):
            # Fall back to centroid displacement
            return self._centroid_flow(current_location)

        # Subsample for efficiency if point clouds are large
        prev_pts = self._prev_point_cloud
        curr_pts = current_points
        max_points = 500
        if len(prev_pts) > max_points:
            idx = np.linspace(0, len(prev_pts) - 1, max_points, dtype=int)
            prev_pts = prev_pts[idx]
        if len(curr_pts) > max_points:
            idx = np.linspace(0, len(curr_pts) - 1, max_points, dtype=int)
            curr_pts = curr_pts[idx]

        # For each current point, find nearest previous point
        # Using broadcasting: (M, 1, 3) - (1, N, 3) → (M, N, 3)
        # This is O(M*N) but M, N ≤ max_points so it's fast
        diffs = curr_pts[:, np.newaxis, :] - prev_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)  # (M, N)
        nearest_idx = np.argmin(dists, axis=1)  # (M,)
        nearest_dist = dists[np.arange(len(curr_pts)), nearest_idx]  # (M,)

        # Filter to persistent points (within correspondence threshold)
        persistent_mask = nearest_dist < self._correspondence_threshold
        n_persistent = np.sum(persistent_mask)

        if n_persistent < self._min_persistent_points:
            # Too few correspondences — fall back to centroid
            return self._centroid_flow(current_location)

        # Compute displacement of persistent point pairs
        matched_curr = curr_pts[persistent_mask]
        matched_prev = prev_pts[nearest_idx[persistent_mask]]
        displacements = matched_curr - matched_prev  # (K, 3)

        # Flow direction: mean displacement (dominant motion direction)
        mean_disp = np.mean(displacements, axis=0)
        mean_norm = float(np.linalg.norm(mean_disp))
        if mean_norm > 1e-10:
            direction = mean_disp / mean_norm
        else:
            direction = np.zeros(3)

        # Flow magnitude: RMS of per-point displacement magnitudes
        # Captures non-rigid deformation even when net displacement ≈ 0
        per_point_mags = np.linalg.norm(displacements, axis=1)  # (K,)
        magnitude = float(np.sqrt(np.mean(per_point_mags ** 2)))

        return direction, magnitude

    def _centroid_flow(self, current_location) -> tuple:
        """Fallback: compute flow from centroid displacement."""
        raw_displacement = current_location - self._prev_location
        magnitude = float(np.linalg.norm(raw_displacement))
        if magnitude > 1e-10:
            direction = raw_displacement / magnitude
        else:
            direction = np.zeros(3)
        return direction, magnitude

    def _compute_feature_deltas(
        self, current_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute per-feature deltas between current and previous."""
        deltas = {}
        if self._prev_features is None:
            return deltas

        for key in current_features:
            if key in self._prev_features:
                delta = current_features[key] - self._prev_features[key]
                deltas[key] = delta

        return deltas

    def _check_feature_changes(
        self, feature_deltas: Dict[str, np.ndarray]
    ) -> bool:
        """Check if any feature delta exceeds its threshold."""
        for key, delta in feature_deltas.items():
            if key in self._feature_change_thresholds:
                threshold = self._feature_change_thresholds[key]
                if np.linalg.norm(delta) > threshold:
                    return True
        return False

    def _copy_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Deep copy feature dict."""
        return {k: v.copy() for k, v in features.items()}

    def _compute_change_signal_strength(
        self,
        flow_magnitude: float,
        feature_deltas: Dict[str, np.ndarray],
    ) -> float:
        """Return a threshold-relative salience score for the current step."""
        scores = []
        if self._flow_threshold > 0:
            scores.append(float(flow_magnitude) / float(self._flow_threshold))

        for key, delta in feature_deltas.items():
            threshold = self._feature_change_thresholds.get(key)
            if threshold is None or threshold <= 0:
                continue
            scores.append(float(np.linalg.norm(delta)) / float(threshold))

        return max(scores, default=0.0)

    def _build_non_morph_features(
        self,
        current_features: Dict[str, np.ndarray],
        flow_direction: np.ndarray,
        flow_magnitude: float,
        feature_deltas: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Build the mixed absolute-plus-change feature payload."""
        non_morph = {
            "flow_direction": np.asarray(flow_direction, dtype=float),
            "flow_magnitude": np.array([flow_magnitude], dtype=float),
        }
        if self._include_absolute_features:
            for key, value in current_features.items():
                non_morph[key] = np.asarray(value, dtype=float).copy()
        for key, delta in feature_deltas.items():
            non_morph[f"delta_{key}"] = np.asarray(delta, dtype=float).copy()
        return non_morph

    def _make_change_state(
        self,
        location: np.ndarray,
        current_features: Dict[str, np.ndarray],
        flow_direction: np.ndarray,
        flow_magnitude: float,
        feature_deltas: Dict[str, np.ndarray],
        signal_strength: float,
    ) -> State:
        """Create a State representing detected change."""
        # Use flow direction as the primary pose vector (direction of movement)
        # Construct orthonormal basis from flow direction
        pose_vectors = self._flow_to_pose_vectors(flow_direction)
        non_morph = self._build_non_morph_features(
            current_features,
            flow_direction,
            flow_magnitude,
            feature_deltas,
        )
        pose_fully_defined = bool(np.linalg.norm(flow_direction) > 1e-10)

        return State(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": pose_fully_defined,
            },
            non_morphological_features=non_morph,
            confidence=min(max(signal_strength, 0.0), 1.0),
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    def _make_quiet_state(
        self,
        location: np.ndarray,
        current_features: Dict[str, np.ndarray],
        flow_direction: np.ndarray,
        flow_magnitude: float,
        feature_deltas: Dict[str, np.ndarray],
        signal_strength: float,
    ) -> State:
        """Create a low-confidence steady-state context observation."""
        pose_vectors = self._flow_to_pose_vectors(flow_direction)
        pose_fully_defined = bool(np.linalg.norm(flow_direction) > 1e-10)
        quiet_confidence = 0.05 + 0.2 * min(max(signal_strength, 0.0), 1.0)

        return State(
            location=location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": pose_fully_defined,
            },
            non_morphological_features=self._build_non_morph_features(
                current_features,
                flow_direction,
                flow_magnitude,
                feature_deltas,
            ),
            confidence=min(quiet_confidence, 0.25),
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    def _make_no_change_state(self) -> State:
        """Create a State indicating no significant change."""
        return State(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
            },
            non_morphological_features=None,
            confidence=0.0,
            use_state=False,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

    @staticmethod
    def _flow_to_pose_vectors(flow_direction: np.ndarray) -> np.ndarray:
        """Convert a flow direction vector to a 3x3 pose vectors matrix.

        Creates an orthonormal basis where the first vector is the flow
        direction, analogous to how CameraSM uses surface normal as the
        first pose vector.
        """
        v1 = flow_direction.copy()
        norm = np.linalg.norm(v1)
        if norm < 1e-10:
            return np.eye(3)

        v1 = v1 / norm

        # Find a non-parallel vector for cross product
        if abs(v1[0]) < 0.9:
            aux = np.array([1.0, 0.0, 0.0])
        else:
            aux = np.array([0.0, 1.0, 0.0])

        v2 = np.cross(v1, aux)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)

        return np.array([v1, v2, v3])
