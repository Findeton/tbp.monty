# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Inhibitory interneuron populations.

Three biologically distinct inhibitory cell types:

- **PV+ (basket cells)**: Fast perisomatic inhibition → layer-specific
  winner-take-all (top-k) competition.
- **SST+ (Martinotti cells)**: Dendritic inhibition → selectively gate
  multi-head dendritic branches based on learned context.
- **VIP+ cells**: Inhibit SST+ (disinhibition) → driven by top-down
  attention and ACh/novelty, unmasking branches for novel input.

Circuit: VIP+ ⊣ SST+ ⊣ dendritic branches

References:
- Tremblay et al. (2016) — "GABAergic Interneurons in the Neocortex"
- Karnani et al. (2016) — VIP-SST disinhibition
"""

from __future__ import annotations

import torch


class PVInhibition:
    """PV+ basket cell perisomatic inhibition.

    Implements layer-specific top-k competition. Each layer can have
    different sparsity levels.

    Parameters
    ----------
    n_minicolumns : int
        Number of minicolumns.
    layer_sparsities : dict
        Maps layer name to fraction of active minicolumns.
        E.g. {"L4": 0.03, "L23": 0.05, "L5": 0.1}
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_minicolumns: int = 2048,
        layer_sparsities: dict[str, float] | None = None,
        device: str = "cpu",
    ):
        self.n_minicolumns = n_minicolumns
        self.device = torch.device(device)
        self._sparsities = layer_sparsities or {
            "L4": 0.03,
            "L23": 0.05,
            "L5": 0.1,
            "L6": 0.1,
        }

    def inhibit(
        self,
        overlap: torch.Tensor,
        layer: str,
        n_active_override: int | None = None,
    ) -> torch.Tensor:
        """Apply top-k inhibition for a specific layer.

        Parameters
        ----------
        overlap : Tensor (n_minicolumns,)
            Activation overlap scores per minicolumn.
        layer : str
            Layer name (L4, L23, L5, L6).
        n_active_override : int or None
            Override for neuromodulatory sparsity control.

        Returns
        -------
        Bool tensor (n_minicolumns,) indicating winning minicolumns.
        """
        sparsity = self._sparsities.get(layer, 0.05)
        k = n_active_override or max(1, int(sparsity * self.n_minicolumns))
        k = min(k, self.n_minicolumns)

        if k >= self.n_minicolumns:
            return torch.ones(
                self.n_minicolumns, dtype=torch.bool, device=self.device
            )

        winners = torch.zeros(
            self.n_minicolumns, dtype=torch.bool, device=self.device
        )
        _, top_idx = torch.topk(overlap, k)
        winners[top_idx] = True
        return winners


class SSTInhibition:
    """SST+ Martinotti cell dendritic branch inhibition.

    Learns which dendritic branches to gate based on context. Projects to
    dendrites of pyramidal cells, selectively suppressing specific
    multi-head compartments.

    Parameters
    ----------
    n_cells : int
        Number of pyramidal cells being gated.
    n_heads : int
        Number of dendritic heads per cell.
    learning_rate : float
        Hebbian learning rate for SST gating weights.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        n_heads: int = 4,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.n_heads = n_heads
        self._lr = learning_rate
        self.device = torch.device(device)

        # SST activation per head: learned context-dependent gating
        # W_gate[h] maps cell activations → gating for head h
        # Initialized to small values → initially all branches pass
        self._W_gate = torch.zeros(
            n_heads, n_cells, dtype=torch.float32, device=self.device
        )
        torch.nn.init.normal_(self._W_gate, std=0.01)
        self._bias = torch.zeros(
            n_heads, dtype=torch.float32, device=self.device
        )

        # Current gating state
        self._activation = torch.zeros(
            n_heads, dtype=torch.float32, device=self.device
        )

    def compute_gate(
        self,
        cell_activation: torch.Tensor,
        vip_disinhibition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute branch gating from current cell state.

        Parameters
        ----------
        cell_activation : Tensor (n_cells,)
            Current cell activation pattern.
        vip_disinhibition : Tensor (n_heads,) or None
            VIP+ disinhibition signal. Higher = SST more inhibited
            → branches more open.

        Returns
        -------
        Tensor (n_heads,) in [0, 1]. Gate values: 1 = branch open, 0 = gated.
        """
        # SST activation: how much each head should be suppressed
        raw_sst = torch.sigmoid(
            torch.mv(self._W_gate, cell_activation) + self._bias
        )

        if vip_disinhibition is not None:
            # VIP inhibits SST: effective_sst = sst * (1 - vip)
            effective_sst = raw_sst * (1.0 - vip_disinhibition.clamp(0, 1))
        else:
            effective_sst = raw_sst

        self._activation = effective_sst

        # Gate = 1 - SST (high SST = branch suppressed)
        return 1.0 - effective_sst

    def learn(
        self,
        cell_activation: torch.Tensor,
        surprise: float,
        lr: float = 1.0,
    ) -> None:
        """Adjust gating weights.

        Low surprise → strengthen current gating (it was useful).
        High surprise → weaken gating (we need more branches open).
        """
        if cell_activation.abs().sum() < 1e-8:
            return

        # Signal: low surprise → positive (strengthen gate),
        #         high surprise → negative (weaken gate)
        signal = 0.5 - surprise
        delta = self._lr * lr * signal
        self._W_gate += delta * torch.outer(
            self._activation, cell_activation
        )


class VIPDisinhibition:
    """VIP+ cell disinhibition circuit.

    VIP+ fires in response to top-down attention and novelty (ACh),
    inhibiting SST+ cells and thereby unmasking dendritic branches.

    Parameters
    ----------
    n_heads : int
        Number of dendritic heads to disinhibit.
    novelty_weight : float
        Weight of novelty (ACh) signal on VIP activation.
    attention_weight : float
        Weight of top-down attention on VIP activation.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_heads: int = 4,
        novelty_weight: float = 1.0,
        attention_weight: float = 0.5,
        device: str = "cpu",
    ):
        self.n_heads = n_heads
        self.novelty_weight = novelty_weight
        self.attention_weight = attention_weight
        self.device = torch.device(device)

        self._activation = torch.zeros(
            n_heads, dtype=torch.float32, device=self.device
        )

    def compute(
        self,
        novelty: float = 0.5,
        top_down_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute VIP+ activation.

        Parameters
        ----------
        novelty : float
            ACh/novelty signal in [0, 1].
        top_down_attention : Tensor (n_heads,) or None
            Per-head attention signal from higher areas.

        Returns
        -------
        Tensor (n_heads,) in [0, 1]. VIP activation level.
        """
        vip = self.novelty_weight * novelty * torch.ones(
            self.n_heads, dtype=torch.float32, device=self.device
        )

        if top_down_attention is not None:
            vip += self.attention_weight * top_down_attention.to(self.device)

        self._activation = torch.sigmoid(vip - 0.5)
        return self._activation.clone()

    def reset(self) -> None:
        self._activation.zero_()


class InterneuronCircuit:
    """Combined PV+/SST+/VIP+ interneuron circuit.

    Convenience wrapper that combines all three interneuron types and
    provides a single interface for the column.

    Parameters
    ----------
    n_minicolumns : int
        Number of minicolumns.
    n_cells : int
        Total cells (for SST gating).
    n_heads : int
        Dendritic heads per cell.
    layer_sparsities : dict or None
        Per-layer sparsity for PV+ inhibition.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_minicolumns: int = 2048,
        n_cells: int = 16384,
        n_heads: int = 4,
        layer_sparsities: dict[str, float] | None = None,
        sst_learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        self.pv = PVInhibition(
            n_minicolumns=n_minicolumns,
            layer_sparsities=layer_sparsities,
            device=device,
        )
        self.sst = SSTInhibition(
            n_cells=n_cells,
            n_heads=n_heads,
            learning_rate=sst_learning_rate,
            device=device,
        )
        self.vip = VIPDisinhibition(
            n_heads=n_heads,
            device=device,
        )

    def compute_branch_gate(
        self,
        cell_activation: torch.Tensor,
        novelty: float = 0.5,
        top_down_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the full VIP → SST → branch gate.

        Returns
        -------
        Tensor (n_heads,) branch gate values in [0, 1].
        """
        vip_signal = self.vip.compute(
            novelty=novelty, top_down_attention=top_down_attention
        )
        gate = self.sst.compute_gate(
            cell_activation=cell_activation,
            vip_disinhibition=vip_signal,
        )
        return gate

    def learn(
        self,
        cell_activation: torch.Tensor,
        surprise: float,
        lr: float = 1.0,
    ) -> None:
        """Learn SST gating weights."""
        self.sst.learn(cell_activation, surprise, lr)

    def reset(self) -> None:
        self.vip.reset()
