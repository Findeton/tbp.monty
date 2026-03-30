# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Spike-Timing-Dependent Plasticity (STDP) and eligibility traces.

Replaces symmetric Hebbian (W += lr * pre * post) with temporally
asymmetric learning that captures *direction* of causation:

    trace_pre += pre_active;   trace_pre *= decay
    trace_post += post_active; trace_post *= decay

    ΔW_LTP = lr_LTP * post_active * trace_pre   (causal: pre→post)
    ΔW_LTD = -lr_LTD * pre_active * trace_post  (acausal: post→pre)

Eligibility traces bridge STDP to neuromodulatory consolidation:

    eligibility += ΔW_proposed;  eligibility *= decay_elig
    ΔW_actual = eligibility * neuromodulator_signal

References:
- Gerstner et al. (2018) — eligibility traces on behavioral timescales
- Izhikevich (2007) — STDP + dopamine for distal reward
"""

from __future__ import annotations

import torch


class STDPRule:
    """Trace-based STDP for continuous-rate networks.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    lr_ltp : float
        Learning rate for LTP (causal: pre before post).
    lr_ltd : float
        Learning rate for LTD (acausal: post before pre).
    trace_decay : float
        Per-step decay for pre/post traces (τ ≈ 5-10 steps).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        lr_ltp: float = 0.01,
        lr_ltd: float = 0.008,
        trace_decay: float = 0.8,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.lr_ltp = lr_ltp
        self.lr_ltd = lr_ltd
        self.trace_decay = trace_decay
        self.device = torch.device(device)

        self._trace_pre = torch.zeros(
            n_cells, dtype=torch.float32, device=self.device
        )
        self._trace_post = torch.zeros(
            n_cells, dtype=torch.float32, device=self.device
        )

    def update_traces(
        self,
        pre_active: torch.Tensor,
        post_active: torch.Tensor,
    ) -> None:
        """Update pre and post traces with current activation.

        Parameters
        ----------
        pre_active : Tensor (n_cells,)
            Pre-synaptic activation (previous step's active cells).
        post_active : Tensor (n_cells,)
            Post-synaptic activation (current step's active cells).
        """
        self._trace_pre = self._trace_pre * self.trace_decay + pre_active
        self._trace_post = self._trace_post * self.trace_decay + post_active

    def compute_weight_update(
        self,
        pre_active: torch.Tensor,
        post_active: torch.Tensor,
    ) -> torch.Tensor:
        """Compute STDP weight change matrix.

        Parameters
        ----------
        pre_active : Tensor (n_cells,)
            Pre-synaptic activation.
        post_active : Tensor (n_cells,)
            Post-synaptic activation.

        Returns
        -------
        Tensor (n_cells, n_cells) of proposed weight changes.
        ΔW[i,j] = change for synapse from j (pre) to i (post).
        """
        # LTP: post fires, trace_pre was active → strengthen pre→post
        # ΔW[i,j] for synapse j→i: post_active[i] * trace_pre[j]
        ltp = self.lr_ltp * torch.outer(post_active, self._trace_pre)

        # LTD: pre fires, trace_post was active → weaken synapse
        # ΔW[i,j] for synapse j→i: -trace_post[i] * pre_active[j]
        ltd = -self.lr_ltd * torch.outer(self._trace_post, pre_active)

        return ltp + ltd

    def compute_sparse_update(
        self,
        pre_active: torch.Tensor,
        post_active: torch.Tensor,
        connectivity: list[tuple[int, dict[int, float]]],
    ) -> dict[tuple[int, int], float]:
        """Compute STDP weight changes only for existing synapses.

        More memory-efficient than full matrix for sparse dendrites.

        Parameters
        ----------
        pre_active : Tensor (n_cells,)
        post_active : Tensor (n_cells,)
        connectivity : list of (parent_cell, {source: permanence})
            Existing dendritic segments.

        Returns
        -------
        Dict mapping (parent_cell, source_cell) → ΔW.
        """
        updates = {}
        pre_cpu = pre_active.detach().cpu()
        post_cpu = post_active.detach().cpu()

        for parent_cell, synapses in connectivity:
            post_val = float(post_cpu[parent_cell])
            for src in synapses:
                pre_val = float(pre_cpu[src])

                # LTP: post fires now, pre was recently active
                ltp = self.lr_ltp * post_val * float(self._trace_pre[src])
                # LTD: pre fires now, post was recently active
                ltd = -self.lr_ltd * pre_val * float(
                    self._trace_post[parent_cell]
                )

                delta = ltp + ltd
                if abs(delta) > 1e-8:
                    updates[(parent_cell, src)] = delta

        return updates

    def apply_sparse_update(
        self,
        connectivity: list[tuple[int, dict[int, float]]],
        updates: dict[tuple[int, int], float],
    ) -> None:
        """Apply STDP updates to dendritic segment permanences."""
        for parent_cell, synapses in connectivity:
            for src in list(synapses.keys()):
                key = (parent_cell, src)
                if key in updates:
                    synapses[src] = max(0.0, min(1.0,
                        synapses[src] + updates[key]
                    ))

    def reset(self) -> None:
        self._trace_pre.zero_()
        self._trace_post.zero_()


class EligibilityTrace:
    """Eligibility trace bridging STDP to neuromodulatory consolidation.

    Proposed weight changes from STDP accumulate as eligibility traces
    that decay slowly. Actual weight changes only occur when a
    neuromodulatory signal (reward/novelty) arrives.

    Parameters
    ----------
    decay : float
        Per-step decay for eligibility traces (τ ≈ 50-100 steps).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        decay: float = 0.99,
        device: str = "cpu",
    ):
        self.decay = decay
        self.device = torch.device(device)
        self._traces: dict[tuple[int, int], float] = {}

    def accumulate(self, updates: dict[tuple[int, int], float]) -> None:
        """Add proposed STDP changes to eligibility traces.

        Parameters
        ----------
        updates : dict
            Mapping (post_cell, pre_cell) → ΔW from STDP.
        """
        # Decay existing
        for key in list(self._traces.keys()):
            self._traces[key] *= self.decay
            if abs(self._traces[key]) < 1e-8:
                del self._traces[key]

        # Accumulate new
        for key, delta in updates.items():
            self._traces[key] = self._traces.get(key, 0.0) + delta

    def consolidate(
        self,
        neuromod_signal: float,
        connectivity: list[tuple[int, dict[int, float]]],
    ) -> int:
        """Apply eligible weight changes scaled by neuromodulator.

        Parameters
        ----------
        neuromod_signal : float
            Combined neuromodulatory signal. Positive → consolidate,
            negative → decay, zero → no change.
        connectivity : list of (parent_cell, {source: permanence})
            Dendritic segments to modify.

        Returns
        -------
        Number of synapses modified.
        """
        if abs(neuromod_signal) < 1e-8 or not self._traces:
            return 0

        n_modified = 0
        for parent_cell, synapses in connectivity:
            for src in list(synapses.keys()):
                key = (parent_cell, src)
                if key in self._traces:
                    delta = self._traces[key] * neuromod_signal
                    synapses[src] = max(0.0, min(1.0, synapses[src] + delta))
                    n_modified += 1

        return n_modified

    def clear(self) -> None:
        self._traces.clear()

    @property
    def n_eligible(self) -> int:
        return len(self._traces)
