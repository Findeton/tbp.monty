# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""PyTorch encoders: State → sparse continuous input tensor.

Same mathematical operations as the numpy encoders in
``cortical_column.encoders``, but using PyTorch tensors for GPU compatibility.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch


class TorchGridCellEncoder:
    """Multi-scale periodic location encoder producing float32 activations.

    Each module has a different scale.  For each module, location is projected
    onto ``n_phases`` random directions and passed through cos/sin, giving a
    ``2 * n_phases``-dimensional activation per module.

    Total output size = ``n_modules * 2 * n_phases``.
    """

    def __init__(
        self,
        n_modules: int = 20,
        n_phases: int = 8,
        scales: list[float] | None = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_modules = n_modules
        self.n_phases = n_phases
        self.total_bits = n_modules * 2 * n_phases
        self.device = torch.device(device)

        if scales is None:
            scales = [2.0 ** i for i in range(n_modules)]
        self._scales = torch.tensor(scales[:n_modules], dtype=torch.float32,
                                    device=self.device)

        rng = np.random.RandomState(seed)
        dirs = rng.randn(n_modules, n_phases, 3).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8
        dirs /= norms
        self._directions = torch.from_numpy(dirs).to(self.device)

    def encode(self, location: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a 3D location into grid cell activations.

        Returns a tensor of shape ``(total_bits,)`` with values in [-1, 1].
        """
        if isinstance(location, np.ndarray):
            loc = torch.from_numpy(location.astype(np.float32)).to(self.device)
        else:
            loc = location.to(self.device).float()

        # (n_modules, n_phases) = directions @ loc
        proj = torch.matmul(self._directions, loc)  # (n_modules, n_phases)
        scaled = proj * self._scales.unsqueeze(1)    # (n_modules, n_phases)
        cos_part = torch.cos(scaled)
        sin_part = torch.sin(scaled)
        # Interleave: (n_modules, 2 * n_phases)
        result = torch.stack([cos_part, sin_part], dim=-1)
        return result.reshape(-1)


class TorchScalarEncoder:
    """Encode a scalar value into a distributed float activation pattern."""

    def __init__(
        self,
        n_bits: int = 64,
        min_val: float = 0.0,
        max_val: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_bits = n_bits
        self.min_val = min_val
        self.max_val = max_val
        self.device = torch.device(device)

        rng = np.random.RandomState(seed)
        self._centers = torch.from_numpy(
            np.sort(rng.uniform(min_val, max_val, n_bits)).astype(np.float32)
        ).to(self.device)
        self._width = (max_val - min_val) / n_bits * 2.0

    def encode(self, value: float) -> torch.Tensor:
        """Encode a scalar. Returns ``(n_bits,)`` tensor with Gaussian bumps."""
        v = torch.tensor(value, dtype=torch.float32, device=self.device)
        return torch.exp(-0.5 * ((self._centers - v) / self._width) ** 2)


class TorchFeatureEncoder:
    """Encode a Monty State into a single sparse continuous input tensor.

    Combines grid-cell location encoding with scalar feature encoding.
    """

    def __init__(
        self,
        location_bits: int = 320,
        feature_bits_per_dim: int = 32,
        n_feature_dims: int = 10,
        observation_packet_location_mode: str = "state_location",
        ignore_pose_vectors_for_observation_packets: bool = False,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self._observation_packet_location_mode = str(
            observation_packet_location_mode
        )
        if self._observation_packet_location_mode not in {
            "state_location",
            "sensor_frame_centroid",
        }:
            raise ValueError(
                "observation_packet_location_mode must be one of "
                "{'state_location', 'sensor_frame_centroid'}"
            )
        self._ignore_pose_vectors_for_observation_packets = bool(
            ignore_pose_vectors_for_observation_packets
        )

        n_modules = max(1, location_bits // 16)
        n_phases = 8
        actual_loc_bits = n_modules * 2 * n_phases
        self._location_encoder = TorchGridCellEncoder(
            n_modules=n_modules, n_phases=n_phases, seed=seed, device=device,
        )
        self._location_bits = actual_loc_bits

        self._feature_bits_per_dim = feature_bits_per_dim
        self._n_feature_dims = n_feature_dims
        self._feature_encoders = [
            TorchScalarEncoder(
                n_bits=feature_bits_per_dim,
                min_val=-1.0,
                max_val=1.0,
                seed=seed + i + 100,
                device=device,
            )
            for i in range(n_feature_dims)
        ]

        self.total_bits = actual_loc_bits + feature_bits_per_dim * n_feature_dims

    def encode(self, state) -> torch.Tensor:
        """Encode a Monty State into a continuous input vector.

        Extracts location and up to ``n_feature_dims`` scalar features from
        the State's morphological and non-morphological features.
        """
        parts = []

        # Location
        loc = self.extract_location(state)
        parts.append(self._location_encoder.encode(loc))

        # Features: extract scalars from morphological + non-morphological
        feature_vals = self._extract_features(
            state,
            ignore_pose_vectors_with_packet=(
                self._ignore_pose_vectors_for_observation_packets
            ),
        )
        for i, enc in enumerate(self._feature_encoders):
            val = feature_vals[i] if i < len(feature_vals) else 0.0
            parts.append(enc.encode(val))

        return torch.cat(parts)

    def encode_location(self, location: np.ndarray) -> torch.Tensor:
        """Encode just the location component."""
        return self._location_encoder.encode(location)

    def extract_location(self, state) -> np.ndarray:
        loc = np.asarray(
            getattr(state, "location", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        ).reshape(-1)
        if loc.size < 3:
            loc = np.pad(loc, (0, 3 - loc.size), mode="constant")
        loc = loc[:3]

        if self._observation_packet_location_mode != "sensor_frame_centroid":
            return loc

        non_morph = getattr(state, "non_morphological_features", None) or {}
        packet = self._get_observation_packet(non_morph)
        sensor_frame_loc = self._extract_sensor_frame_centroid(packet)
        if sensor_frame_loc is not None:
            return sensor_frame_loc

        if str(getattr(state, "sender_type", "SM")) == "SM":
            return np.zeros(3, dtype=np.float32)
        return loc

    @staticmethod
    def _get_observation_packet(
        non_morphological_features: dict[str, Any],
    ) -> dict[str, Any] | None:
        for packet_key in ("observation_packet_v2", "detail_packet"):
            packet = non_morphological_features.get(packet_key)
            if isinstance(packet, dict):
                return packet
        return None

    @staticmethod
    def _extract_sensor_frame_centroid(
        packet: dict[str, Any] | None,
    ) -> np.ndarray | None:
        if not isinstance(packet, dict):
            return None

        weighted_sum = np.zeros(3, dtype=np.float32)
        total_weight = 0.0
        for cell in packet.get("cells", []) or []:
            sensor_patch = np.asarray(
                cell.get("sensor_frame_patch", np.zeros((0, 0, 3))),
                dtype=np.float32,
            )
            if sensor_patch.ndim != 3 or sensor_patch.size == 0 or sensor_patch.shape[-1] < 3:
                continue

            points = sensor_patch[..., :3].reshape(-1, 3)
            weights = np.ones(points.shape[0], dtype=np.float32)
            support_patch = np.asarray(
                cell.get("support_patch", np.zeros((0, 0))),
                dtype=np.float32,
            )
            if support_patch.ndim >= 2 and support_patch.shape[:2] == sensor_patch.shape[:2]:
                weights = np.clip(support_patch.reshape(-1), 0.0, 1.0)

            valid = np.isfinite(points).all(axis=1)
            weights = weights * valid.astype(np.float32)
            if float(np.sum(weights)) <= 1e-6:
                continue

            weighted_sum += np.sum(points * weights[:, None], axis=0)
            total_weight += float(np.sum(weights))

        if total_weight <= 1e-6:
            return None
        return (weighted_sum / total_weight).astype(np.float32)

    @staticmethod
    def _clip_unit(value: float) -> float:
        return float(np.clip(float(value), -1.0, 1.0))

    @staticmethod
    def _center_unit_interval(value: float) -> float:
        return TorchFeatureEncoder._clip_unit((2.0 * float(value)) - 1.0)

    @staticmethod
    def _squash_feature(value: float, scale: float = 1.0) -> float:
        return float(np.tanh(float(scale) * float(value)))

    @staticmethod
    def _mean_abs_gradient(values: np.ndarray, axis: int) -> float:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0 or array.shape[axis] <= 1:
            return 0.0
        return float(np.mean(np.abs(np.diff(array, axis=axis))))

    @staticmethod
    def _extract_packet_features(state, packet: dict[str, Any]) -> list[float]:
        cells = packet.get("cells")
        if not isinstance(cells, list) or len(cells) == 0:
            return []

        rgb_means: list[np.ndarray] = []
        rgb_gradients: list[np.ndarray] = []
        sensor_z_means: list[float] = []

        for cell in cells:
            rgb_patch = np.asarray(
                cell.get("rgb_patch", np.zeros((0, 0, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            if rgb_patch.ndim == 3 and rgb_patch.size > 0:
                rgb_means.append(np.mean(rgb_patch, axis=(0, 1)))
                rgb_gradients.append(
                    np.asarray(
                        [
                            TorchFeatureEncoder._mean_abs_gradient(rgb_patch, axis=0),
                            TorchFeatureEncoder._mean_abs_gradient(rgb_patch, axis=1),
                        ],
                        dtype=np.float32,
                    )
                )

            sensor_patch = np.asarray(
                cell.get("sensor_frame_patch", np.zeros((0, 0, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            if sensor_patch.ndim == 3 and sensor_patch.size > 0 and sensor_patch.shape[-1] >= 3:
                sensor_z_means.append(float(np.mean(sensor_patch[..., 2])))

        rgb_mean = (
            np.mean(np.stack(rgb_means, axis=0), axis=0)
            if rgb_means
            else np.zeros(3, dtype=np.float32)
        )
        rgb_gradient = (
            np.mean(np.stack(rgb_gradients, axis=0), axis=0)
            if rgb_gradients
            else np.zeros(2, dtype=np.float32)
        )
        sensor_z_mean = (
            float(np.mean(sensor_z_means)) if sensor_z_means else 0.0
        )
        luminance = float(np.dot(rgb_mean[:3], np.array([0.299, 0.587, 0.114], dtype=np.float32)))
        red_green = float(rgb_mean[0] - rgb_mean[1]) if rgb_mean.size >= 2 else 0.0
        texture_x = TorchFeatureEncoder._squash_feature(rgb_gradient[0], scale=2.0)
        texture_y = TorchFeatureEncoder._squash_feature(rgb_gradient[1], scale=2.0)
        texture_energy = TorchFeatureEncoder._squash_feature(
            float(np.linalg.norm(rgb_gradient)),
            scale=1.5,
        )

        packet_type = str(packet.get("packet_type") or "visual_observation_packet_v2")
        is_change_packet = packet_type == "change_observation_packet_v2"
        if not is_change_packet:
            return [
                TorchFeatureEncoder._center_unit_interval(luminance),
                TorchFeatureEncoder._clip_unit(red_green),
                texture_x,
                texture_y,
            ]

        temporal_context = packet.get("temporal_context") or {}
        delta_norms = []
        feature_deltas = temporal_context.get("feature_deltas") or {}
        if isinstance(feature_deltas, dict):
            for value in feature_deltas.values():
                delta = np.asarray(value, dtype=np.float32).reshape(-1)
                if delta.size > 0:
                    delta_norms.append(float(np.linalg.norm(delta)))

        delta_mean = float(np.mean(delta_norms)) if delta_norms else 0.0
        delta_spread = float(np.std(delta_norms)) if delta_norms else 0.0
        return [
            TorchFeatureEncoder._squash_feature(delta_mean, scale=2.0),
            TorchFeatureEncoder._squash_feature(delta_spread, scale=2.0),
            texture_energy,
        ]

    @staticmethod
    def _extract_features(
        state,
        *,
        ignore_pose_vectors_with_packet: bool = False,
    ) -> list[float]:
        """Pull scalar features from a State for encoding."""
        vals = []

        morph = getattr(state, "morphological_features", None) or {}
        non_morph = getattr(state, "non_morphological_features", None) or {}
        sender_type = str(getattr(state, "sender_type", "SM"))
        packet_features: list[float] = []

        if sender_type == "LM":
            graph_id = non_morph.get("graph_id")
            if graph_id not in (None, "", "unknown", "no_observations_yet"):
                vals.extend(
                    TorchFeatureEncoder._hash_text_features(str(graph_id), 4)
                )

            sender_id = getattr(state, "sender_id", None)
            if sender_id not in (None, ""):
                vals.extend(
                    TorchFeatureEncoder._hash_text_features(str(sender_id), 1)
                )

            evidence = non_morph.get("evidence")
            if evidence is not None:
                vals.append(float(np.tanh(float(evidence))))

            surprise = non_morph.get("surprise")
            if surprise is not None:
                vals.append(float(np.clip(float(surprise), -1.0, 1.0)))

        packet = TorchFeatureEncoder._get_observation_packet(non_morph)
        if packet is not None:
            packet_features = TorchFeatureEncoder._extract_packet_features(state, packet)

        packet_temporal_context = packet.get("temporal_context") if packet else None

        # HSV color
        hsv = non_morph.get("hsv")
        if hsv is not None:
            if hasattr(hsv, "__len__"):
                vals.extend(float(v) for v in hsv[:3])
            else:
                vals.append(float(hsv))

        # Flow direction and magnitude
        flow_direction = None
        if isinstance(packet_temporal_context, dict):
            flow_direction = packet_temporal_context.get("flow_direction")
        if flow_direction is None:
            flow_direction = non_morph.get("flow_direction")
        if flow_direction is not None:
            vals.extend(float(v) for v in np.asarray(flow_direction).ravel()[:3])

        flow_magnitude = None
        if isinstance(packet_temporal_context, dict):
            flow_magnitude = packet_temporal_context.get("flow_magnitude")
        if flow_magnitude is None:
            flow_magnitude = non_morph.get("flow_magnitude")
        if flow_magnitude is not None:
            if hasattr(flow_magnitude, "__len__"):
                vals.append(float(np.asarray(flow_magnitude).ravel()[0]))
            else:
                vals.append(float(flow_magnitude))

        # Curvatures
        for key in ("principal_curvatures", "principal_curvatures_log",
                     "gaussian_curvature", "mean_curvature"):
            val = non_morph.get(key, morph.get(key))
            if val is not None:
                if hasattr(val, "__len__"):
                    vals.extend(float(v) for v in np.asarray(val).ravel()[:4])
                else:
                    vals.append(float(val))

        # Pose vectors — flatten the 3x3 matrix and take first few values
        pose = morph.get("pose_vectors")
        if not (
            pose is not None
            and packet is not None
            and ignore_pose_vectors_with_packet
            and sender_type == "SM"
        ) and pose is not None:
            flat = np.asarray(pose).ravel()
            vals.extend(float(v) for v in flat[:3])

        vals.extend(packet_features)

        return vals

    @staticmethod
    def _hash_text_features(text: str, n_vals: int) -> list[float]:
        """Map text to deterministic float features in [-1, 1]."""
        digest = hashlib.sha256(text.encode()).digest()
        vals = []
        for idx in range(max(0, int(n_vals))):
            start = (2 * idx) % len(digest)
            chunk = digest[start:start + 2]
            if len(chunk) < 2:
                chunk = (chunk + digest[:2])[:2]
            raw = int.from_bytes(chunk, byteorder="big", signed=False)
            vals.append((2.0 * (raw / 65535.0)) - 1.0)
        return vals

    @staticmethod
    def hash_label(name: str, n_bits: int = 256) -> torch.Tensor:
        """Generate a deterministic binary label SDR from an object name."""
        h = hashlib.sha256(name.encode()).digest()
        bits = torch.zeros(n_bits, dtype=torch.float32)
        for i in range(min(n_bits, len(h) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if h[byte_idx] & (1 << bit_idx):
                bits[i] = 1.0
        return bits
