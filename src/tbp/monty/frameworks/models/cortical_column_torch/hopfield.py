# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Modern Hopfield network memory (Ramsauer et al. 2020).

Implements energy-based attractor dynamics with exponential storage capacity
via the log-sum-exp energy function and softmax update rule.  All operations
are forward-only — no autograd, no backprop.

Energy:  E(x) = -log Σ exp(β ⟨ξᵢ, x⟩) + ½β‖x‖²
Update:  x_new = Ξᵀ softmax(β Ξ x)
"""

from __future__ import annotations

import torch


class ModernHopfieldMemory:
    """Modern Hopfield network with sparse continuous patterns.

    Stores patterns via Hebbian accumulation and retrieves via the modern
    Hopfield update rule (softmax attention over stored patterns).

    Parameters
    ----------
    n_cells : int
        Dimensionality of patterns.
    beta : float
        Inverse temperature.  Higher → sharper retrieval, lower → broader.
    max_stored : int
        Maximum number of stored patterns (ring buffer).
    max_settle_iters : int
        Maximum settling iterations.
    convergence_threshold : float
        L2 change threshold for convergence.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        beta: float = 8.0,
        max_stored: int = 1000,
        max_settle_iters: int = 10,
        convergence_threshold: float = 1e-4,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.beta = beta
        self.max_stored = max_stored
        self.max_settle_iters = max_settle_iters
        self.convergence_threshold = convergence_threshold
        self.device = torch.device(device)

        # Stored patterns: (n_stored, n_cells)
        self._patterns = torch.zeros(
            0, n_cells, dtype=torch.float32, device=self.device
        )
        self._n_stored = 0

    @property
    def n_stored(self) -> int:
        return self._n_stored

    def store(self, pattern: torch.Tensor) -> None:
        """Store a pattern (Hebbian accumulation).

        Normalizes the pattern before storing.
        """
        with torch.no_grad():
            p = pattern.detach().to(self.device).float()
            norm = p.norm() + 1e-8
            p = p / norm

            if self._n_stored < self.max_stored:
                self._patterns = torch.cat(
                    [self._patterns, p.unsqueeze(0)], dim=0
                )
                self._n_stored += 1
            else:
                # Ring buffer: overwrite oldest
                idx = self._n_stored % self.max_stored
                self._patterns[idx] = p
                self._n_stored += 1

    def settle(
        self,
        x: torch.Tensor,
        beta: float | None = None,
        max_iters: int | None = None,
        sparsity_fn=None,
    ) -> tuple[torch.Tensor, int]:
        """Run modern Hopfield settling to convergence.

        Parameters
        ----------
        x : Tensor (n_cells,)
            Initial activation pattern.
        beta : float or None
            Override inverse temperature.
        max_iters : int or None
            Override max iterations.
        sparsity_fn : callable or None
            Applied after each iteration to enforce sparsity.

        Returns
        -------
        (converged_x, n_iterations)
        """
        if self._n_stored == 0:
            return x, 0

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            iters = max_iters if max_iters is not None else self.max_settle_iters
            n_patterns = min(self._n_stored, self.max_stored)
            xi = self._patterns[:n_patterns]  # (N, n_cells)

            x_cur = x.clone().to(self.device)

            for i in range(iters):
                # Modern Hopfield update: x_new = Ξᵀ softmax(β Ξ x)
                similarities = b * torch.mv(xi, x_cur)  # (N,)
                # Numerical stability
                similarities = similarities - similarities.max()
                weights = torch.softmax(similarities, dim=0)  # (N,)
                x_new = torch.mv(xi.t(), weights)  # (n_cells,)

                if sparsity_fn is not None:
                    x_new = sparsity_fn(x_new)

                # Convergence check
                delta = (x_new - x_cur).norm()
                x_cur = x_new
                if delta < self.convergence_threshold:
                    return x_cur, i + 1

            return x_cur, iters

    def energy(self, x: torch.Tensor, beta: float | None = None) -> torch.Tensor:
        """Compute the modern Hopfield energy.

        E(x) = -log Σ exp(β ⟨ξᵢ, x⟩) + ½β‖x‖²
        """
        if self._n_stored == 0:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            n_patterns = min(self._n_stored, self.max_stored)
            xi = self._patterns[:n_patterns]

            similarities = b * torch.mv(xi, x)
            lse = torch.logsumexp(similarities, dim=0)
            regularizer = 0.5 * b * x.dot(x)
            return -lse + regularizer

    def retrieve(
        self,
        query: torch.Tensor,
        beta: float | None = None,
    ) -> torch.Tensor:
        """Single-step retrieval (no settling loop).

        Returns Ξᵀ softmax(β Ξ query).
        """
        if self._n_stored == 0:
            return query

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            n_patterns = min(self._n_stored, self.max_stored)
            xi = self._patterns[:n_patterns]

            similarities = b * torch.mv(xi, query)
            similarities = similarities - similarities.max()
            weights = torch.softmax(similarities, dim=0)
            return torch.mv(xi.t(), weights)

    def clear(self) -> None:
        """Remove all stored patterns."""
        self._patterns = torch.zeros(
            0, self.n_cells, dtype=torch.float32, device=self.device
        )
        self._n_stored = 0

    def state_dict(self) -> dict:
        return {
            "patterns": self._patterns.cpu(),
            "n_stored": self._n_stored,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._patterns = sd["patterns"].to(self.device)
        self._n_stored = sd["n_stored"]
