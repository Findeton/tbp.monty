# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Motor-conditional sensorimotor prediction for cortical columns.

Learns to predict the next location given the current location and a motor
displacement (efference copy). Uses a simple Hebbian associative weight
matrix between concatenated (location_sdr, motor_sdr) and next_location_sdr.

This enables:
- Distinguishing "I moved" (predicted) from "world changed" (unpredicted)
- Active inference: propose motor actions that maximally disambiguate hypotheses
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.models.cortical_column.encoders import (
    GridCellEncoder,
    ScalarEncoder,
)


class MotorEncoder:
    """Encode a 3D motor displacement as an SDR.

    Uses three independent ScalarEncoders (one per axis) concatenated.

    Parameters
    ----------
    bits_per_axis : int
        SDR bits for each axis.
    active_per_axis : int
        Active bits per axis.
    max_displacement : float
        Maximum expected displacement magnitude per axis.
    """

    def __init__(
        self,
        bits_per_axis: int = 64,
        active_per_axis: int = 5,
        max_displacement: float = 0.2,
    ):
        self.bits_per_axis = bits_per_axis
        self.active_per_axis = active_per_axis
        self.total_bits = bits_per_axis * 3
        self.n_active = active_per_axis * 3

        self._encoders = [
            ScalarEncoder(
                n_bits=bits_per_axis,
                n_active=active_per_axis,
                min_val=-max_displacement,
                max_val=max_displacement,
            )
            for _ in range(3)
        ]

    def encode(self, displacement: np.ndarray) -> np.ndarray:
        """Encode a 3D displacement vector.

        Parameters
        ----------
        displacement : np.ndarray, shape (3,)
            Motor displacement (dx, dy, dz).

        Returns
        -------
        np.ndarray, shape (total_bits,)
            SDR encoding of the displacement.
        """
        parts = []
        for i, enc in enumerate(self._encoders):
            parts.append(enc.encode(float(displacement[i])))
        return np.concatenate(parts)


class MotorPrediction:
    """Learns motor-conditional location predictions via Hebbian association.

    During training, records (location_t, displacement, location_{t+1}) and
    learns a weight matrix W such that:
        predicted_next_loc ≈ W @ concat(location_sdr, motor_sdr)

    The prediction error (difference between predicted and actual next
    location SDR) serves as a signal for distinguishing self-motion from
    world changes.

    Parameters
    ----------
    location_bits : int
        Size of the location SDR (from GridCellEncoder).
    motor_encoder_kwargs : dict or None
        Arguments for MotorEncoder.
    learning_rate : float
        Hebbian learning rate.
    decay_rate : float
        Per-step weight decay to prevent saturation.
    """

    def __init__(
        self,
        location_bits: int = 2048,
        motor_encoder_kwargs: dict = None,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
    ):
        self._location_bits = location_bits
        self._lr = learning_rate
        self._decay = decay_rate

        me_kwargs = motor_encoder_kwargs or {}
        self._motor_encoder = MotorEncoder(**me_kwargs)

        # Input = concat(location_sdr, motor_sdr)
        self._input_bits = location_bits + self._motor_encoder.total_bits
        # Output = location_sdr (predicted next location)
        self._output_bits = location_bits

        # Hebbian weight matrix: output x input (float32 to save memory)
        self._weights = np.zeros(
            (self._output_bits, self._input_bits), dtype=np.float32
        )

        # State from last step (for learning on next step)
        self._prev_location_sdr = None
        self._prev_motor_sdr = None
        self._prev_input = None

    @property
    def motor_encoder(self) -> MotorEncoder:
        return self._motor_encoder

    def reset(self) -> None:
        """Reset episode state (not weights)."""
        self._prev_location_sdr = None
        self._prev_motor_sdr = None
        self._prev_input = None

    def predict(
        self, location_sdr: np.ndarray, motor_sdr: np.ndarray,
    ) -> np.ndarray:
        """Predict the next location SDR from current location + motor command.

        Parameters
        ----------
        location_sdr : np.ndarray, shape (location_bits,)
            Current location encoding.
        motor_sdr : np.ndarray, shape (motor_total_bits,)
            Motor displacement encoding.

        Returns
        -------
        np.ndarray, shape (location_bits,), float32
            Predicted next location activation (continuous, not thresholded).
        """
        input_vec = np.concatenate([
            location_sdr.astype(np.float32),
            motor_sdr.astype(np.float32),
        ])
        return self._weights @ input_vec

    def step(
        self,
        location_sdr: np.ndarray,
        displacement: np.ndarray,
    ) -> dict:
        """Process one sensorimotor step.

        Call this each step with the current location and the motor
        displacement that was just executed. On the second+ call, it:
        1. Predicts current location from (prev_location, prev_motor)
        2. Computes prediction error against actual current location
        3. Learns the association

        Parameters
        ----------
        location_sdr : np.ndarray, shape (location_bits,)
            Current (actual) location SDR.
        displacement : np.ndarray, shape (3,)
            Motor displacement that was just executed to arrive here.

        Returns
        -------
        dict with keys:
            prediction_error : float
                Normalized prediction error (0 = perfect, 1 = no overlap).
            predicted_location : np.ndarray or None
                The predicted location activation (None on first step).
        """
        motor_sdr = self._motor_encoder.encode(displacement)
        result = {"prediction_error": 1.0, "predicted_location": None}

        if self._prev_input is not None:
            # Predict current location from previous state + motor
            predicted = self._weights @ self._prev_input

            # Prediction error: 1 - normalized overlap
            loc_f = location_sdr.astype(np.float32)
            n_active = max(float(loc_f.sum()), 1.0)

            # Threshold predicted to get binary, then compute overlap
            pred_threshold = np.sort(predicted)[-max(1, int(n_active))]
            pred_binary = predicted >= pred_threshold
            overlap = float(np.dot(loc_f, pred_binary.astype(np.float32)))
            pred_error = 1.0 - (overlap / n_active)

            result["prediction_error"] = max(0.0, min(1.0, pred_error))
            result["predicted_location"] = predicted

            # Learn: Hebbian update W += lr * (target - predicted) @ input^T
            error = loc_f - predicted
            self._weights += self._lr * np.outer(error, self._prev_input)

            # Decay
            if self._decay > 0:
                self._weights *= (1.0 - self._decay)

            # Clip to prevent explosion
            np.clip(self._weights, -1.0, 1.0, out=self._weights)

        # Save for next step
        self._prev_location_sdr = location_sdr.copy()
        self._prev_motor_sdr = motor_sdr.copy()
        self._prev_input = np.concatenate([
            location_sdr.astype(np.float32),
            motor_sdr.astype(np.float32),
        ])

        return result

    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        return self._weights.nbytes
