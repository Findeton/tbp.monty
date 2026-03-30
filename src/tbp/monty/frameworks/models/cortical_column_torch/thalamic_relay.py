# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Thalamocortical loop gating.

The thalamus is not a passive relay. It actively gates what information
reaches L4 and maintains persistent activity for working memory.

Circuit:
    Sensory input → Thalamic relay → L4
                         ↑
                    L6 (feedback/modulation)

- L6 feedback → thalamic gate: sigmoid(W @ L6_activation + bias)
- Low surprise → L6 partially closes gate → attenuates redundant input
- High surprise → L6 opens gate → full input reaches L4
- This implements predictive coding at the thalamic level
"""

from __future__ import annotations

import torch


class ThalamicRelay:
    """Thalamic relay unit with L6-modulated gating.

    Parameters
    ----------
    n_input : int
        Dimensionality of sensory input.
    n_l6_cells : int
        Number of L6 cells providing feedback.
    learning_rate : float
        Learning rate for gate weight adaptation.
    initial_gate_bias : float
        Initial gate bias. Positive = gate starts open.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_input: int = 640,
        n_l6_cells: int = 2048,
        learning_rate: float = 0.005,
        initial_gate_bias: float = 2.0,
        device: str = "cpu",
    ):
        self.n_input = n_input
        self.n_l6_cells = n_l6_cells
        self._lr = learning_rate
        self.device = torch.device(device)

        # Gate weights: L6 activation → per-input gate
        self._W_gate = torch.zeros(
            n_input, n_l6_cells, dtype=torch.float32, device=self.device
        )
        torch.nn.init.normal_(self._W_gate, std=0.01)

        # Bias starts positive → gate initially open
        self._bias = torch.full(
            (n_input,), initial_gate_bias,
            dtype=torch.float32, device=self.device
        )

        self._gate = torch.ones(
            n_input, dtype=torch.float32, device=self.device
        )

    @property
    def gate(self) -> torch.Tensor:
        """Current gate values in [0, 1]."""
        return self._gate

    def forward(
        self,
        sensory_input: torch.Tensor,
        l6_activation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gate sensory input through thalamic relay.

        Parameters
        ----------
        sensory_input : Tensor (n_input,)
            Raw encoded sensory input.
        l6_activation : Tensor (n_l6_cells,) or None
            L6 feedback. If None, gate is fully open.

        Returns
        -------
        Gated input: sensory_input * gate
        """
        if l6_activation is not None:
            self._gate = torch.sigmoid(
                self._W_gate @ l6_activation + self._bias
            )
        else:
            self._gate = torch.ones(
                self.n_input, dtype=torch.float32, device=self.device
            )

        return sensory_input * self._gate

    def learn(
        self,
        surprise: float,
        l6_activation: torch.Tensor | None = None,
        lr: float = 1.0,
    ) -> None:
        """Adapt gate weights based on surprise.

        High surprise → the gate should have been more open
        → adjust weights to open gate when this L6 pattern occurs.

        Low surprise → gate correctly attenuated predictable input
        → strengthen current gating.
        """
        if l6_activation is None or l6_activation.abs().sum() < 1e-8:
            return

        # Error signal: how wrong was the gate?
        # High surprise → gate was too closed → positive signal → open more
        signal = (surprise - 0.5) * self._lr * lr

        # Gradient: d(gate)/d(W) ∝ gate * (1 - gate) * l6
        gate_grad = self._gate * (1.0 - self._gate)
        self._W_gate += signal * torch.outer(gate_grad, l6_activation)
        self._bias += signal * gate_grad

    def reset(self) -> None:
        """Reset gate to fully open."""
        self._gate = torch.ones(
            self.n_input, dtype=torch.float32, device=self.device
        )
