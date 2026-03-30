# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Multi-head dendritic attention.

Each cell has ``n_heads`` independent dendritic compartments (branches).
Each branch produces its own graded activation via sparse matmul. The
cell's total depolarization is a **nonlinear multiplicative integration**
of branch activations — matching supralinear dendritic integration
observed experimentally (Larkum 2013).

    branch_acts[h] = sigmoid(D_h @ x)
    cell_depol = product(1 + alpha * branch_acts[h]) - 1

Two moderately active branches produce a much stronger response than
one strongly active branch.
"""

from __future__ import annotations

import torch


class MultiHeadDendrites:
    """Multi-head dendritic segments with nonlinear integration.

    Parameters
    ----------
    n_cells : int
        Total number of cells.
    n_heads : int
        Number of independent dendritic compartments per cell.
    max_segments_per_head : int
        Maximum segments per head per cell.
    max_synapses_per_segment : int
        Maximum synapses per segment.
    activation_threshold : float
        Overlap threshold for segment activation.
    sigmoid_temp : float
        Temperature for soft sigmoid activation.
    connected_threshold : float
        Permanence threshold for connected synapse.
    permanence_increment : float
        Learning increment for active synapses.
    permanence_decrement : float
        Learning decrement for inactive synapses.
    initial_permanence : float
        Initial permanence for new synapses.
    alpha : float
        Multiplicative integration gain. Higher = more supralinear.
    seed : int
        Random seed.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        n_heads: int = 4,
        max_segments_per_head: int = 8,
        max_synapses_per_segment: int = 24,
        activation_threshold: float = 0.3,
        sigmoid_temp: float = 0.1,
        connected_threshold: float = 0.3,
        permanence_increment: float = 0.05,
        permanence_decrement: float = 0.02,
        initial_permanence: float = 0.5,
        alpha: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.n_heads = n_heads
        self.max_segments_per_head = max_segments_per_head
        self.max_synapses_per_segment = max_synapses_per_segment
        self.activation_threshold = activation_threshold
        self.sigmoid_temp = sigmoid_temp
        self.connected_threshold = connected_threshold
        self.perm_inc = permanence_increment
        self.perm_dec = permanence_decrement
        self.initial_permanence = initial_permanence
        self.alpha = alpha
        self.device = torch.device(device)

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)

        # Per-head segment storage:
        # _heads[h] = list of (parent_cell, {source_cell: permanence})
        self._heads: list[list[tuple[int, dict[int, float]]]] = [
            [] for _ in range(n_heads)
        ]
        self._segment_count_per_cell_per_head = torch.zeros(
            n_heads, n_cells, dtype=torch.int32, device="cpu"
        )

    @property
    def total_segments(self) -> int:
        return sum(len(h) for h in self._heads)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Compute graded depolarization with multi-head integration.

        Each head produces an independent branch activation per cell.
        Cell depolarization = product(1 + alpha * branch_act) - 1.

        Returns
        -------
        Tensor (n_cells,) of graded depolarization in [0, 1].
        """
        if self.total_segments == 0:
            return torch.zeros(
                self.n_cells, dtype=torch.float32, device=self.device
            )

        x_cpu = x.detach().cpu()
        # Per-head, per-cell max activation
        branch_acts = torch.zeros(
            self.n_heads, self.n_cells, dtype=torch.float32
        )

        for h, segments in enumerate(self._heads):
            for parent_cell, synapses in segments:
                overlap = 0.0
                for src, perm in synapses.items():
                    if perm >= self.connected_threshold:
                        overlap += float(x_cpu[src])

                activation = torch.sigmoid(
                    torch.tensor(
                        (overlap - self.activation_threshold)
                        / max(self.sigmoid_temp, 1e-6)
                    )
                ).item()

                if activation > branch_acts[h, parent_cell]:
                    branch_acts[h, parent_cell] = activation

        # Multiplicative integration across heads
        # cell_depol = product(1 + alpha * branch_act[h]) - 1
        product = torch.ones(self.n_cells, dtype=torch.float32)
        for h in range(self.n_heads):
            product *= (1.0 + self.alpha * branch_acts[h])
        depol = (product - 1.0).clamp(0.0, 1.0)

        return depol.to(self.device)

    def predict_per_head(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-head activations without integration.

        Returns
        -------
        Tensor (n_heads, n_cells) of per-head branch activations.
        """
        if self.total_segments == 0:
            return torch.zeros(
                self.n_heads, self.n_cells, dtype=torch.float32,
                device=self.device,
            )

        x_cpu = x.detach().cpu()
        branch_acts = torch.zeros(
            self.n_heads, self.n_cells, dtype=torch.float32
        )

        for h, segments in enumerate(self._heads):
            for parent_cell, synapses in segments:
                overlap = 0.0
                for src, perm in synapses.items():
                    if perm >= self.connected_threshold:
                        overlap += float(x_cpu[src])

                activation = torch.sigmoid(
                    torch.tensor(
                        (overlap - self.activation_threshold)
                        / max(self.sigmoid_temp, 1e-6)
                    )
                ).item()

                if activation > branch_acts[h, parent_cell]:
                    branch_acts[h, parent_cell] = activation

        return branch_acts.to(self.device)

    def grow_segment(
        self,
        cell: int,
        source_cells: list[int],
        head: int = 0,
    ) -> None:
        """Grow a new segment on a specific head of a cell."""
        if head >= self.n_heads:
            return
        if (int(self._segment_count_per_cell_per_head[head, cell])
                >= self.max_segments_per_head):
            return

        synapses = {}
        for src in source_cells[:self.max_synapses_per_segment]:
            synapses[src] = self.initial_permanence

        self._heads[head].append((cell, synapses))
        self._segment_count_per_cell_per_head[head, cell] += 1

    def learn(
        self,
        active_cells: torch.Tensor,
        prev_active_cells: torch.Tensor,
        learning_rate: float = 1.0,
        head: int | None = None,
    ) -> None:
        """Hebbian learning on segments.

        If ``head`` is specified, only learn on that head. Otherwise learn
        on all heads.
        """
        active_set = set(
            torch.nonzero(active_cells, as_tuple=True)[0].tolist()
        )
        prev_set = set(
            torch.nonzero(prev_active_cells, as_tuple=True)[0].tolist()
        )

        heads_to_learn = [head] if head is not None else range(self.n_heads)

        for h in heads_to_learn:
            for parent_cell, synapses in self._heads[h]:
                if parent_cell not in active_set:
                    continue
                for src in list(synapses.keys()):
                    if src in prev_set:
                        synapses[src] = min(
                            1.0, synapses[src] + self.perm_inc * learning_rate
                        )
                    else:
                        synapses[src] = max(
                            0.0, synapses[src] - self.perm_dec * learning_rate
                        )

    def grow_for_unpredicted(
        self,
        active_cells: torch.Tensor,
        prev_active_cells: torch.Tensor,
        predicted_cells: torch.Tensor,
    ) -> None:
        """Grow segments for unpredicted cells, distributing across heads."""
        active_idx = torch.nonzero(active_cells, as_tuple=True)[0].tolist()
        predicted_idx = set(
            torch.nonzero(predicted_cells, as_tuple=True)[0].tolist()
        )
        prev_idx = torch.nonzero(prev_active_cells, as_tuple=True)[0].tolist()

        if not prev_idx:
            return

        for cell in active_idx:
            if cell in predicted_idx:
                continue
            # Pick the head with fewest segments for this cell
            counts = self._segment_count_per_cell_per_head[:, cell]
            best_head = int(counts.argmin())

            n = min(len(prev_idx), self.max_synapses_per_segment)
            perm = torch.randperm(len(prev_idx), generator=self._rng)[:n]
            sources = [prev_idx[i] for i in perm.tolist()]
            self.grow_segment(cell, sources, head=best_head)

    def apply_gate(
        self, branch_gate: torch.Tensor
    ) -> None:
        """Store a per-head gating mask from SST+ interneurons.

        This gate is applied during ``predict()`` — gated branches
        have their activation suppressed.

        Parameters
        ----------
        branch_gate : Tensor (n_heads,) or (n_heads, n_cells)
            Gate values in [0, 1]. 0 = fully suppressed, 1 = fully active.
        """
        self._branch_gate = branch_gate.to(self.device)

    def predict_gated(self, x: torch.Tensor) -> torch.Tensor:
        """Like predict() but applies SST+ branch gating."""
        branch_acts = self.predict_per_head(x)

        if hasattr(self, "_branch_gate") and self._branch_gate is not None:
            gate = self._branch_gate
            if gate.dim() == 1:
                gate = gate.unsqueeze(1)  # (n_heads, 1)
            branch_acts = branch_acts * gate

        # Multiplicative integration
        product = torch.ones(self.n_cells, dtype=torch.float32,
                             device=self.device)
        for h in range(self.n_heads):
            product *= (1.0 + self.alpha * branch_acts[h])
        return (product - 1.0).clamp(0.0, 1.0)

    def prune(self, min_permanence: float = 0.01) -> int:
        """Remove segments with all permanences below threshold."""
        removed = 0
        self._segment_count_per_cell_per_head.zero_()

        for h in range(self.n_heads):
            new_segs = []
            for parent_cell, synapses in self._heads[h]:
                alive = {s: p for s, p in synapses.items()
                         if p >= min_permanence}
                if alive:
                    new_segs.append((parent_cell, alive))
                    self._segment_count_per_cell_per_head[h, parent_cell] += 1
                else:
                    removed += 1
            self._heads[h] = new_segs

        return removed
