# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Laminar layer modules for CorticalColumnTorch.

Each cortical layer has distinct cell populations, connectivity patterns,
and computational roles:

- **L4** (input): Spiny stellate cells. Feedforward spatial pooling. No
  long-range projections.
- **L2/3** (representation): Pyramidal cells. Hopfield recurrent settling.
  Lateral voting projections.
- **L5** (output): Large pyramidal cells with apical dendrites in L1.
  Integrates bottom-up (L2/3) and top-down (L1 context). Produces
  column output for hierarchy and motor.
- **L6** (feedback): Projects to thalamic relay, modulating feedforward
  gate. Sends feedback to L4.

L1 has no cell bodies — it is a dendritic zone where L5 apical tufts
receive top-down input. Represented implicitly via L5's apical dendrites.
"""

from __future__ import annotations

import torch


class LaminarConfig:
    """Configuration for laminar cell allocation per minicolumn.

    Parameters
    ----------
    n_L4 : int
        Cells per minicolumn in L4 (input layer).
    n_L23 : int
        Cells per minicolumn in L2/3 (representation layer).
    n_L5 : int
        Cells per minicolumn in L5 (output layer).
    n_L6 : int
        Cells per minicolumn in L6 (feedback layer).
    """

    def __init__(
        self,
        n_L4: int = 2,
        n_L23: int = 4,
        n_L5: int = 1,
        n_L6: int = 1,
    ):
        self.n_L4 = n_L4
        self.n_L23 = n_L23
        self.n_L5 = n_L5
        self.n_L6 = n_L6

    @property
    def cells_per_minicolumn(self) -> int:
        return self.n_L4 + self.n_L23 + self.n_L5 + self.n_L6

    def layer_offsets(self, mc_idx: int) -> dict[str, tuple[int, int]]:
        """Return (start, end) cell indices for each layer within a minicolumn.

        Parameters
        ----------
        mc_idx : int
            Minicolumn index.

        Returns
        -------
        Dict mapping layer name to (start_cell, end_cell) global indices.
        """
        base = mc_idx * self.cells_per_minicolumn
        offsets = {}
        cursor = base
        for name, count in [
            ("L4", self.n_L4),
            ("L23", self.n_L23),
            ("L5", self.n_L5),
            ("L6", self.n_L6),
        ]:
            offsets[name] = (cursor, cursor + count)
            cursor += count
        return offsets

    def layer_slice(
        self, n_minicolumns: int, layer: str
    ) -> tuple[torch.Tensor, int]:
        """Return indices of all cells in a given layer across all minicolumns.

        Parameters
        ----------
        n_minicolumns : int
            Total number of minicolumns.
        layer : str
            One of "L4", "L23", "L5", "L6".

        Returns
        -------
        (indices_tensor, n_cells_in_layer) — global cell indices for the layer.
        """
        cpm = self.cells_per_minicolumn
        layer_map = {"L4": 0, "L23": self.n_L4, "L5": self.n_L4 + self.n_L23,
                     "L6": self.n_L4 + self.n_L23 + self.n_L5}
        count_map = {"L4": self.n_L4, "L23": self.n_L23,
                     "L5": self.n_L5, "L6": self.n_L6}
        offset = layer_map[layer]
        count = count_map[layer]

        indices = []
        for mc in range(n_minicolumns):
            base = mc * cpm + offset
            for c in range(count):
                indices.append(base + c)

        return torch.tensor(indices, dtype=torch.long), count * n_minicolumns


class L4Layer:
    """Layer 4: Feedforward input processing via spatial pooling.

    Receives encoded sensory input (or thalamic relay output). Computes
    overlap with proximal dendrite permanences and applies top-k inhibition.
    Output feeds L2/3.

    This layer wraps the existing spatial pooling logic but operates only
    on L4 cells within each minicolumn.
    """

    def __init__(
        self,
        n_minicolumns: int,
        n_input: int,
        n_cells_per_mc: int = 2,
        sparsity: float = 0.03,
        connected_threshold: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_mc = n_cells_per_mc
        self.n_cells = n_minicolumns * n_cells_per_mc
        self._sparsity = sparsity
        self._n_active_mc = max(1, int(sparsity * n_minicolumns))
        self._connected_threshold = connected_threshold
        self.device = torch.device(device)

        import numpy as np
        rng = np.random.RandomState(seed + 2000)
        n_potential = max(1, n_input // 10)

        self._ff_potential = torch.zeros(
            n_minicolumns, n_input, dtype=torch.bool, device=self.device
        )
        self._ff_permanences = torch.zeros(
            n_minicolumns, n_input, dtype=torch.float32, device=self.device
        )
        self._sp_increment = 0.05
        self._sp_decrement = 0.02

        for mc in range(n_minicolumns):
            indices = rng.choice(n_input, size=n_potential, replace=False)
            self._ff_potential[mc, indices] = True
            perms = rng.normal(connected_threshold, 0.05, n_potential)
            perms = np.clip(perms, 0.0, 1.0).astype(np.float32)
            self._ff_permanences[mc, indices] = torch.from_numpy(perms)

        # Boosting
        self._boost_factors = torch.ones(
            n_minicolumns, dtype=torch.float32, device=self.device
        )
        self._mc_duty_cycle = torch.full(
            (n_minicolumns,), sparsity, dtype=torch.float32, device=self.device
        )
        self._boost_strength = 3.0

        # Active minicolumns mask
        self._active_mc = torch.zeros(
            n_minicolumns, dtype=torch.bool, device=self.device
        )
        # L4 cell activations: one value per L4 cell
        self._activation = torch.zeros(
            self.n_cells, dtype=torch.float32, device=self.device
        )

    def forward(
        self,
        input_vec: torch.Tensor,
        n_active_override: int | None = None,
        thalamic_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute L4 spatial pooling.

        Parameters
        ----------
        input_vec : Tensor (n_input,)
            Encoded sensory input.
        n_active_override : int or None
            Override number of active minicolumns (for neuromodulation).
        thalamic_gate : Tensor (n_input,) or None
            Per-input gating from thalamic relay (L6 feedback).

        Returns
        -------
        (active_mc, l4_activation) — active minicolumn mask and L4 cell values.
        """
        gated = input_vec
        if thalamic_gate is not None:
            gated = input_vec * thalamic_gate

        active_input = (gated.abs() > 0.1).float()
        connected = (
            (self._ff_permanences >= self._connected_threshold) &
            self._ff_potential
        ).float()
        overlap = (connected * active_input.unsqueeze(0)).sum(dim=1)
        boosted = overlap * self._boost_factors

        k = n_active_override if n_active_override else self._n_active_mc
        k = min(k, self.n_minicolumns)

        if k >= self.n_minicolumns:
            winners = torch.arange(self.n_minicolumns, device=self.device)
        else:
            _, winners = torch.topk(boosted, k)

        self._active_mc.zero_()
        self._active_mc[winners] = True

        # L4 activation: overlap score distributed to cells in winning MCs
        self._activation.zero_()
        act_2d = self._activation.reshape(self.n_minicolumns, self.n_cells_per_mc)
        for mc_idx in winners:
            mc = mc_idx.item()
            # Uniform activation for L4 cells in winning minicolumn
            act_2d[mc] = overlap[mc] / (overlap[mc].abs() + 1e-8)

        return self._active_mc.clone(), self._activation.clone()

    def learn(self, input_vec: torch.Tensor, lr: float = 1.0) -> None:
        """Hebbian SP learning on L4 proximal dendrites."""
        active_input = (input_vec.abs() > 0.1)
        winner_idx = torch.nonzero(self._active_mc, as_tuple=True)[0]
        if len(winner_idx) == 0:
            return

        inc = self._sp_increment * lr
        dec = self._sp_decrement * lr

        w_pot = self._ff_potential[winner_idx]
        active_pot = active_input.unsqueeze(0) & w_pot
        inactive_pot = (~active_input).unsqueeze(0) & w_pot

        self._ff_permanences[winner_idx] = torch.clamp(
            self._ff_permanences[winner_idx]
            + inc * active_pot.float()
            - dec * inactive_pot.float(),
            0.0, 1.0,
        )

        alpha = 0.01
        self._mc_duty_cycle *= (1.0 - alpha)
        self._mc_duty_cycle[self._active_mc] += alpha
        if self._boost_strength > 0:
            self._boost_factors = torch.exp(
                self._boost_strength * (self._sparsity - self._mc_duty_cycle)
            )


class L23Layer:
    """Layer 2/3: Representation layer with Hopfield recurrent settling.

    Receives feedforward input from L4. Performs recurrent Hopfield
    settling to converge to stored attractor patterns. Lateral projections
    carry voting signals to L2/3 of other columns.
    """

    def __init__(
        self,
        n_minicolumns: int,
        n_cells_per_mc: int = 4,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_mc = n_cells_per_mc
        self.n_cells = n_minicolumns * n_cells_per_mc
        self.device = torch.device(device)

        self._activation = torch.zeros(
            self.n_cells, dtype=torch.float32, device=self.device
        )

    def forward(
        self,
        l4_active_mc: torch.Tensor,
        l4_activation: torch.Tensor,
        basal_prediction: torch.Tensor,
        n_L4_cells_per_mc: int,
    ) -> torch.Tensor:
        """Compute L2/3 activation from L4 input + basal prediction.

        Parameters
        ----------
        l4_active_mc : Tensor (n_minicolumns,) bool
            Which minicolumns are active from L4.
        l4_activation : Tensor (n_L4_cells,)
            L4 cell activations.
        basal_prediction : Tensor (n_L23_cells,)
            Basal dendritic prediction for L2/3 cells.
        n_L4_cells_per_mc : int
            Number of L4 cells per minicolumn (for indexing).

        Returns
        -------
        L2/3 activation tensor.
        """
        self._activation.zero_()
        act_2d = self._activation.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )
        pred_2d = basal_prediction.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )

        active_mcs = torch.nonzero(l4_active_mc, as_tuple=True)[0]

        for mc_idx in active_mcs:
            mc = mc_idx.item()
            pred = pred_2d[mc]
            has_pred = pred.max() > 0.3

            if has_pred:
                act_2d[mc] = pred
            else:
                # Burst: all cells equally active
                act_2d[mc] = 1.0 / self.n_cells_per_mc

        return self._activation.clone()


class L5Layer:
    """Layer 5: Output layer with apical dendrite integration.

    Integrates bottom-up input from L2/3 (basal) and top-down context
    from L1/higher areas (apical). Produces the column's output signal.
    When both basal and apical inputs coincide → plateau potential.
    """

    def __init__(
        self,
        n_minicolumns: int,
        n_cells_per_mc: int = 1,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_mc = n_cells_per_mc
        self.n_cells = n_minicolumns * n_cells_per_mc
        self.device = torch.device(device)

        self._activation = torch.zeros(
            self.n_cells, dtype=torch.float32, device=self.device
        )
        # Flags for plateau potential detection
        self._has_basal_and_apical = torch.zeros(
            self.n_cells, dtype=torch.bool, device=self.device
        )

    def forward(
        self,
        l23_activation: torch.Tensor,
        apical_prediction: torch.Tensor,
        active_mc: torch.Tensor,
        n_L23_cells_per_mc: int,
    ) -> torch.Tensor:
        """Compute L5 output by integrating L2/3 (basal) and apical.

        Parameters
        ----------
        l23_activation : Tensor (n_L23_cells,)
            L2/3 settled activation.
        apical_prediction : Tensor (n_L5_cells,)
            Apical dendritic prediction from top-down context.
        active_mc : Tensor (n_minicolumns,) bool
            Active minicolumns.
        n_L23_cells_per_mc : int
            L2/3 cells per minicolumn.

        Returns
        -------
        L5 activation tensor.
        """
        self._activation.zero_()
        self._has_basal_and_apical.zero_()

        l23_2d = l23_activation.reshape(self.n_minicolumns, n_L23_cells_per_mc)
        act_2d = self._activation.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )
        apical_2d = apical_prediction.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )
        ba_2d = self._has_basal_and_apical.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )

        active_mcs = torch.nonzero(active_mc, as_tuple=True)[0]

        for mc_idx in active_mcs:
            mc = mc_idx.item()
            # Basal drive: mean of L2/3 activation in this minicolumn
            basal_drive = l23_2d[mc].mean()
            apical_drive = apical_2d[mc]

            has_basal = basal_drive > 0.1
            has_apical = apical_drive.max() > 0.1

            if has_basal and has_apical:
                # Both: strong activation + plateau potential trigger
                act_2d[mc] = 0.7 * basal_drive + 0.3 * apical_drive
                ba_2d[mc] = True
            elif has_basal:
                act_2d[mc] = basal_drive
            elif has_apical:
                act_2d[mc] = apical_drive * 0.3
            # else: no drive, stays zero

        return self._activation.clone()


class L6Layer:
    """Layer 6: Feedback/modulation layer.

    Projects to thalamic relay, modulating the gate that controls L4 input.
    Also sends feedback to L4 for representational refinement. The feedback
    signal is computed from the column's current state (L2/3 + L5) and
    applied on the *next* step (one-step delay).
    """

    def __init__(
        self,
        n_minicolumns: int,
        n_input: int,
        n_cells_per_mc: int = 1,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.n_cells_per_mc = n_cells_per_mc
        self.n_cells = n_minicolumns * n_cells_per_mc
        self.n_input = n_input
        self._lr = learning_rate
        self.device = torch.device(device)

        self._activation = torch.zeros(
            self.n_cells, dtype=torch.float32, device=self.device
        )
        # L6 → thalamic gate weights: maps L6 cells to gate over input
        # Initialized as identity-like: each L6 cell gates a portion of input
        self._W_gate = torch.zeros(
            n_input, self.n_cells, dtype=torch.float32, device=self.device
        )
        torch.nn.init.normal_(self._W_gate, std=0.01)

        self._bias_gate = torch.zeros(
            n_input, dtype=torch.float32, device=self.device
        )

    def forward(
        self,
        l23_activation: torch.Tensor,
        l5_activation: torch.Tensor,
        active_mc: torch.Tensor,
        n_L23_cells_per_mc: int,
        n_L5_cells_per_mc: int,
    ) -> torch.Tensor:
        """Compute L6 activation from L2/3 and L5 state.

        Parameters
        ----------
        l23_activation : Tensor
            L2/3 activation.
        l5_activation : Tensor
            L5 activation.
        active_mc : Tensor (n_minicolumns,) bool
            Active minicolumns.

        Returns
        -------
        L6 activation tensor.
        """
        self._activation.zero_()
        act_2d = self._activation.reshape(
            self.n_minicolumns, self.n_cells_per_mc
        )
        l23_2d = l23_activation.reshape(self.n_minicolumns, n_L23_cells_per_mc)
        l5_2d = l5_activation.reshape(self.n_minicolumns, n_L5_cells_per_mc)

        active_mcs = torch.nonzero(active_mc, as_tuple=True)[0]
        for mc_idx in active_mcs:
            mc = mc_idx.item()
            # L6 activation = combination of L2/3 and L5
            drive = 0.5 * l23_2d[mc].mean() + 0.5 * l5_2d[mc].mean()
            act_2d[mc] = drive

        return self._activation.clone()

    def compute_thalamic_gate(self) -> torch.Tensor:
        """Compute thalamic gate from current L6 activation.

        Returns gate in [0, 1] per input channel. Higher = more input passes.
        """
        gate = torch.sigmoid(
            self._W_gate @ self._activation + self._bias_gate
        )
        return gate

    def learn(self, surprise: float, lr: float = 1.0) -> None:
        """Adjust gate weights based on surprise.

        High surprise → open gate more (reduce attenuation).
        Low surprise → close gate (attenuate predictable input).
        """
        if self._activation.abs().sum() < 1e-8:
            return

        # Gradient signal: surprise - 0.5 (centered)
        signal = surprise - 0.5
        # Move gate weights to reduce surprise
        delta = self._lr * lr * signal * torch.outer(
            torch.ones(self.n_input, device=self.device),
            self._activation,
        )
        self._W_gate += delta
