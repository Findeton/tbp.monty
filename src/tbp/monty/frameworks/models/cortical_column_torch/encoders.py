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
        seed: int = 42,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

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
        loc = np.asarray(getattr(state, "location", [0, 0, 0]), dtype=np.float32)
        parts.append(self._location_encoder.encode(loc))

        # Features: extract scalars from morphological + non-morphological
        feature_vals = self._extract_features(state)
        for i, enc in enumerate(self._feature_encoders):
            val = feature_vals[i] if i < len(feature_vals) else 0.0
            parts.append(enc.encode(val))

        return torch.cat(parts)

    def encode_location(self, location: np.ndarray) -> torch.Tensor:
        """Encode just the location component."""
        return self._location_encoder.encode(location)

    @staticmethod
    def _extract_features(state) -> list[float]:
        """Pull scalar features from a State for encoding."""
        vals = []

        morph = getattr(state, "morphological_features", None) or {}
        non_morph = getattr(state, "non_morphological_features", None) or {}

        # HSV color
        hsv = non_morph.get("hsv")
        if hsv is not None:
            if hasattr(hsv, "__len__"):
                vals.extend(float(v) for v in hsv[:3])
            else:
                vals.append(float(hsv))

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
        if pose is not None:
            flat = np.asarray(pose).ravel()
            vals.extend(float(v) for v in flat[:3])

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
