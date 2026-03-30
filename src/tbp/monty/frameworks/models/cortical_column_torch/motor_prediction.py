# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Contrastive Hebbian motor prediction.

Replaces the delta rule with biologically plausible learning:
1. Free phase: predict next location from W @ concat(loc, motor)
2. Clamped phase: clamp to actual observed location
3. Weight update: ΔW = lr * (clamped_coactivations - free_coactivations)

No explicit error computation, no backprop, no autograd.
"""

from __future__ import annotations

import torch


class ContrastiveMotorPrediction:
    """Contrastive Hebbian learning for sensorimotor prediction.

    Parameters
    ----------
    location_dim : int
        Dimensionality of location encoding.
    motor_dim : int
        Dimensionality of motor command (displacement vector).
    hidden_dim : int
        Size of the prediction layer.
    learning_rate : float
        Contrastive Hebbian learning rate.
    n_settle_iters : int
        Settling iterations for free/clamped phases.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        location_dim: int = 320,
        motor_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        n_settle_iters: int = 3,
        device: str = "cpu",
    ):
        self.location_dim = location_dim
        self.motor_dim = motor_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_settle_iters = n_settle_iters
        self.device = torch.device(device)

        input_dim = location_dim + motor_dim

        # Prediction weights: input → hidden
        self._W_ih = torch.zeros(
            hidden_dim, input_dim, dtype=torch.float32, device=self.device
        )
        # Hidden → output (location prediction)
        self._W_ho = torch.zeros(
            location_dim, hidden_dim, dtype=torch.float32, device=self.device
        )
        # Initialize with small random values
        torch.nn.init.normal_(self._W_ih, std=0.01)
        torch.nn.init.normal_(self._W_ho, std=0.01)

        self._prev_location: torch.Tensor | None = None
        self._prediction_error = 0.0

    @property
    def prediction_error(self) -> float:
        return self._prediction_error

    def step(
        self,
        location: torch.Tensor,
        displacement: torch.Tensor,
        learn: bool = True,
    ) -> dict:
        """Process one motor step.

        Parameters
        ----------
        location : Tensor (location_dim,)
            Current location encoding.
        displacement : Tensor (motor_dim,)
            Motor command / displacement vector.
        learn : bool
            Whether to update weights.

        Returns
        -------
        Dict with prediction_error and predicted_location.
        """
        with torch.no_grad():
            loc = location.detach().to(self.device).float()
            disp = displacement.detach().to(self.device).float()

            if self._prev_location is None:
                self._prev_location = loc.clone()
                self._prediction_error = 0.0
                return {"prediction_error": 0.0, "predicted_location": loc}

            # Input: concat(prev_location, displacement)
            inp = torch.cat([self._prev_location, disp])

            # Free phase: predict
            hidden_free = torch.relu(self._W_ih @ inp)
            predicted = self._W_ho @ hidden_free

            # Prediction error
            error = float((loc[:len(predicted)] - predicted).norm())
            self._prediction_error = error

            if learn:
                # Clamped phase: clamp the output to the actual target location
                # and let the hidden units settle via feedback from W_ho.
                # This gives hidden_clamped != hidden_free, which is the core
                # of contrastive Hebbian learning (Movellan 1991).
                target = loc[:self.location_dim]

                # Settle hidden units with output clamped to target:
                # hidden receives both feedforward (from input) and feedback
                # (from clamped output via W_ho transpose).
                hidden_clamped = hidden_free.clone()
                for _ in range(self.n_settle_iters):
                    # Feedback from clamped output
                    feedback = self._W_ho.t() @ target
                    # Combine feedforward and feedback
                    hidden_clamped = torch.relu(
                        self._W_ih @ inp + feedback
                    )

                # Contrastive Hebbian: ΔW = lr * (clamped - free) coactivations
                # W_ho: output × hidden coactivation difference
                clamped_outer = torch.outer(target, hidden_clamped)
                free_outer = torch.outer(predicted, hidden_free)
                self._W_ho += self.learning_rate * (clamped_outer - free_outer)

                # W_ih: hidden × input coactivation difference
                clamped_ih_outer = torch.outer(hidden_clamped, inp)
                free_ih_outer = torch.outer(hidden_free, inp)
                self._W_ih += self.learning_rate * (
                    clamped_ih_outer - free_ih_outer
                )

            self._prev_location = loc.clone()

            return {
                "prediction_error": error,
                "predicted_location": predicted,
            }

    def reset(self) -> None:
        self._prev_location = None
        self._prediction_error = 0.0

    def state_dict(self) -> dict:
        return {
            "W_ih": self._W_ih.cpu(),
            "W_ho": self._W_ho.cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self._W_ih = sd["W_ih"].to(self.device)
        self._W_ho = sd["W_ho"].to(self.device)
