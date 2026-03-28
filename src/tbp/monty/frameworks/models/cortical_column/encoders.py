# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""SDR encoders for cortical column input.

Converts continuous sensory values into sparse distributed representations
using biologically motivated encoding schemes:

- :class:`GridCellEncoder` — multi-scale periodic encoding for spatial location,
  analogous to grid cells in entorhinal cortex. Different modules at different
  spatial scales provide coarse-to-fine location discrimination.

- :class:`ScalarEncoder` — overlapping bucket encoding for continuous features
  (color, curvature, etc.). Adjacent values share active bits, preserving
  similarity in SDR space.

- :func:`encode_state` — combines encoders to produce a single SDR from a
  Monty State object.
"""

from __future__ import annotations

import math

import numpy as np


class GridCellEncoder:
    """Multi-scale periodic location encoder analogous to grid cells.

    Each "module" encodes location at a different spatial scale. Within a
    module, the 3D location is projected onto random directions and the
    phase (location modulo scale) determines which cells are active.

    The result is a multi-resolution SDR where nearby locations share
    active bits (within each scale) but distant locations do not.

    Parameters
    ----------
    n_modules : int
        Number of grid cell modules (different spatial scales).
    cells_per_module : int
        Number of cells in each module.
    n_active_per_module : int
        Number of active cells per module.
    min_scale : float
        Smallest spatial period (finest resolution).
    max_scale : float
        Largest spatial period (coarsest resolution).
    seed : int
        Random seed for projection directions.
    """

    def __init__(
        self,
        n_modules: int = 8,
        cells_per_module: int = 256,
        n_active_per_module: int = 8,
        min_scale: float = 0.02,
        max_scale: float = 1.0,
        seed: int = 42,
    ):
        self.n_modules = n_modules
        self.cells_per_module = cells_per_module
        self.n_active_per_module = n_active_per_module
        self.total_bits = n_modules * cells_per_module
        self.n_active = n_modules * n_active_per_module

        # Geometric spacing of scales (like biological grid cell modules)
        self.scales = np.geomspace(min_scale, max_scale, n_modules)

        # Random projection directions per module (3D -> 1D phase per axis)
        rng = np.random.RandomState(seed)
        # Each module has 3 random unit vectors for projecting 3D location
        self.projections = []
        for _ in range(n_modules):
            vecs = rng.randn(3, 3)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            self.projections.append(vecs)

        # Random phase offsets per cell within each module
        # Each cell has a preferred phase in [0, 1)^3
        self.cell_phases = []
        for _ in range(n_modules):
            phases = rng.uniform(0, 1, (cells_per_module, 3))
            self.cell_phases.append(phases)

    def encode(self, location: np.ndarray) -> np.ndarray:
        """Encode a 3D location as an SDR.

        Parameters
        ----------
        location : array-like, shape (3,)
            3D spatial coordinates.

        Returns
        -------
        np.ndarray, shape (total_bits,)
            Binary SDR with n_active bits set to 1.
        """
        location = np.asarray(location, dtype=np.float64)
        sdr = np.zeros(self.total_bits, dtype=np.float64)

        for m in range(self.n_modules):
            scale = self.scales[m]
            proj = self.projections[m]  # (3, 3)
            cell_ph = self.cell_phases[m]  # (cells_per_module, 3)

            # Project location onto module's random directions
            # Result is 3 phase values (one per direction)
            phase = (proj @ location) / scale  # (3,)
            phase = phase % 1.0  # wrap to [0, 1)

            # Compute distance of each cell's preferred phase to current phase
            # Using circular distance (wrap-around)
            diff = cell_ph - phase[np.newaxis, :]  # (cells_per_module, 3)
            # Circular distance: min(|d|, 1-|d|) for each dimension
            diff = np.abs(diff)
            diff = np.minimum(diff, 1.0 - diff)
            # Combined distance across 3 phase dimensions
            dist = np.sum(diff ** 2, axis=1)  # (cells_per_module,)

            # Winner-take-all: closest cells to current phase
            active_idx = np.argsort(dist)[:self.n_active_per_module]

            offset = m * self.cells_per_module
            sdr[offset + active_idx] = 1.0

        return sdr


class ScalarEncoder:
    """Overlapping bucket encoder for continuous scalar features.

    Maps a continuous value in [min_val, max_val] to a contiguous block
    of active bits in a 1D SDR. Adjacent values share active bits,
    preserving similarity. Values at the extremes wrap (periodic) or
    clip (bounded), controlled by the ``periodic`` parameter.

    This is the standard HTM scalar encoder.

    Parameters
    ----------
    n_bits : int
        Total number of bits in the output SDR.
    n_active : int
        Number of active (1) bits for each encoded value.
    min_val : float
        Minimum expected value.
    max_val : float
        Maximum expected value.
    periodic : bool
        If True, the encoding wraps around (good for angles, hue).
    """

    def __init__(
        self,
        n_bits: int = 128,
        n_active: int = 10,
        min_val: float = 0.0,
        max_val: float = 1.0,
        periodic: bool = False,
    ):
        if n_active > n_bits:
            raise ValueError("n_active must be <= n_bits")
        self.n_bits = n_bits
        self.n_active = n_active
        self.min_val = min_val
        self.max_val = max_val
        self.periodic = periodic

        if periodic:
            self._range = n_bits
        else:
            self._range = n_bits - n_active

    def encode(self, value: float) -> np.ndarray:
        """Encode a scalar value as an SDR.

        Parameters
        ----------
        value : float
            Scalar value to encode.

        Returns
        -------
        np.ndarray, shape (n_bits,)
            Binary SDR with n_active bits set to 1.
        """
        sdr = np.zeros(self.n_bits, dtype=np.float64)

        # Normalize to [0, 1]
        if self.max_val == self.min_val:
            t = 0.5
        else:
            t = (value - self.min_val) / (self.max_val - self.min_val)

        if not self.periodic:
            t = np.clip(t, 0.0, 1.0)
        else:
            t = t % 1.0

        # Map to starting bit position
        start = int(round(t * self._range)) % self.n_bits

        # Set n_active contiguous bits (with wrap-around if periodic)
        for i in range(self.n_active):
            idx = (start + i) % self.n_bits
            sdr[idx] = 1.0

        return sdr


class FeatureSDREncoder:
    """Composite encoder that converts a Monty State into a single SDR.

    Combines GridCellEncoder for location and ScalarEncoders for each
    sensory feature (HSV, curvatures) into a unified sparse representation.

    Parameters
    ----------
    grid_cell_kwargs : dict or None
        Arguments for GridCellEncoder.
    features : list[str]
        Feature names to encode. Supported: "hsv", "principal_curvatures_log".
    seed : int
        Random seed.
    """

    def __init__(
        self,
        grid_cell_kwargs: dict | None = None,
        features: list[str] | None = None,
        seed: int = 42,
    ):
        if features is None:
            features = ["hsv", "principal_curvatures_log"]
        self._features = features

        # Location encoder
        gc_kwargs = dict(
            n_modules=8,
            cells_per_module=256,
            n_active_per_module=8,
            min_scale=0.02,
            max_scale=1.0,
            seed=seed,
        )
        if grid_cell_kwargs:
            gc_kwargs.update(grid_cell_kwargs)
        self.location_encoder = GridCellEncoder(**gc_kwargs)

        # Feature encoders
        self._feature_encoders = {}
        self._feature_dims = {}

        if "hsv" in features:
            # Hue is periodic [0, 1], saturation and value are [0, 1]
            self._feature_encoders["h"] = ScalarEncoder(
                n_bits=256, n_active=11, min_val=0, max_val=1, periodic=True
            )
            self._feature_encoders["s"] = ScalarEncoder(
                n_bits=128, n_active=7, min_val=0, max_val=1
            )
            self._feature_encoders["v"] = ScalarEncoder(
                n_bits=128, n_active=7, min_val=0, max_val=1
            )

        if "principal_curvatures_log" in features:
            # Log curvatures typically in [-5, 5]
            self._feature_encoders["k1"] = ScalarEncoder(
                n_bits=128, n_active=7, min_val=-5, max_val=5
            )
            self._feature_encoders["k2"] = ScalarEncoder(
                n_bits=128, n_active=7, min_val=-5, max_val=5
            )

        # Compute total dimensions
        self._location_bits = self.location_encoder.total_bits
        self._feature_bits = sum(e.n_bits for e in self._feature_encoders.values())
        self.total_bits = self._location_bits + self._feature_bits
        self.n_active_location = self.location_encoder.n_active
        self.n_active_features = sum(
            e.n_active for e in self._feature_encoders.values()
        )
        self.n_active = self.n_active_location + self.n_active_features

    def encode_location(self, location: np.ndarray) -> np.ndarray:
        """Encode just the location as an SDR."""
        return self.location_encoder.encode(location)

    def encode_features(self, state) -> np.ndarray:
        """Encode sensory features from a State object as an SDR."""
        feature_sdrs = []

        if "hsv" in self._features and hasattr(state, "non_morphological_features"):
            nmf = state.non_morphological_features
            if nmf is not None and "hsv" in nmf:
                hsv = nmf["hsv"]
                feature_sdrs.append(self._feature_encoders["h"].encode(hsv[0]))
                feature_sdrs.append(self._feature_encoders["s"].encode(hsv[1]))
                feature_sdrs.append(self._feature_encoders["v"].encode(hsv[2]))
            else:
                for key in ["h", "s", "v"]:
                    feature_sdrs.append(np.zeros(self._feature_encoders[key].n_bits))

        if "principal_curvatures_log" in self._features and hasattr(
            state, "non_morphological_features"
        ):
            nmf = state.non_morphological_features
            if nmf is not None and "principal_curvatures_log" in nmf:
                curv = nmf["principal_curvatures_log"]
                feature_sdrs.append(self._feature_encoders["k1"].encode(curv[0]))
                feature_sdrs.append(self._feature_encoders["k2"].encode(curv[1]))
            else:
                for key in ["k1", "k2"]:
                    feature_sdrs.append(np.zeros(self._feature_encoders[key].n_bits))

        if feature_sdrs:
            return np.concatenate(feature_sdrs)
        return np.zeros(self._feature_bits)

    def encode(self, state) -> np.ndarray:
        """Encode a full State as a combined location + feature SDR.

        Parameters
        ----------
        state : State
            Monty State object with location and features.

        Returns
        -------
        np.ndarray, shape (total_bits,)
            Combined SDR.
        """
        loc_sdr = self.encode_location(np.asarray(state.location))
        feat_sdr = self.encode_features(state)
        return np.concatenate([loc_sdr, feat_sdr])
