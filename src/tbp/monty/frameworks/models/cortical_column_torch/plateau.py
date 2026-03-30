# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Plateau potential working memory for L5 cells.

When L5 cells receive coincident basal (from L2/3) and apical (from L1)
input, they generate dendritic calcium spikes — plateau potentials lasting
200-500ms. This sustained depolarization maintains a working memory trace.

The plateau state enriches the Hopfield retrieval query:
    query = current_activation + gamma * plateau_state
    x_new = Ξᵀ softmax(β · Ξ · query)

References:
- Major & Bhatt (2018) — plateau potentials as cortical working memory
- Takahashi et al. (2020) — plateau-dependent persistent activity
"""

from __future__ import annotations

import torch


class PlateauPotentialMemory:
    """Decaying plateau state for L5 working memory.

    Parameters
    ----------
    n_cells : int
        Number of L5 cells.
    tau : float
        Decay time constant (in steps). Higher = longer context window.
        Plateau decays as: state *= exp(-1/tau) each step.
    plateau_threshold : float
        Minimum combined basal+apical drive to trigger plateau.
    gamma : float
        Weight of plateau state when mixed into retrieval query.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 2048,
        tau: float = 10.0,
        plateau_threshold: float = 0.3,
        gamma: float = 0.3,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self._tau = tau
        self._decay = float(torch.exp(torch.tensor(-1.0 / max(tau, 1e-6))))
        self._threshold = plateau_threshold
        self.gamma = gamma
        self.device = torch.device(device)

        self._state = torch.zeros(
            n_cells, dtype=torch.float32, device=self.device
        )

    @property
    def tau(self) -> float:
        return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        """Update tau (allows neuromodulatory control of context window)."""
        self._tau = value
        self._decay = float(torch.exp(torch.tensor(-1.0 / max(value, 1e-6))))

    @property
    def state(self) -> torch.Tensor:
        return self._state

    def update(
        self,
        l5_activation: torch.Tensor,
        has_basal_and_apical: torch.Tensor,
    ) -> torch.Tensor:
        """Update plateau state.

        Parameters
        ----------
        l5_activation : Tensor (n_cells,)
            Current L5 activation values.
        has_basal_and_apical : Tensor (n_cells,) bool
            Which L5 cells had coincident basal+apical input.

        Returns
        -------
        Updated plateau state tensor.
        """
        # Decay existing plateau
        self._state *= self._decay

        # Trigger new plateaus where basal+apical coincide above threshold
        strong_drive = l5_activation > self._threshold
        trigger = has_basal_and_apical & strong_drive

        # Set plateau to maximum where triggered
        self._state = torch.where(
            trigger,
            torch.clamp(l5_activation, min=self._state),
            self._state,
        )

        return self._state.clone()

    def enrich_query(self, current_activation: torch.Tensor) -> torch.Tensor:
        """Mix plateau state into retrieval query.

        query = current_activation + gamma * plateau_state
        """
        return current_activation + self.gamma * self._state

    def reset(self) -> None:
        self._state.zero_()

    def state_dict(self) -> dict:
        return {"state": self._state.cpu(), "tau": self._tau}

    def load_state_dict(self, sd: dict) -> None:
        self._state = sd["state"].to(self.device)
        self.tau = sd["tau"]
