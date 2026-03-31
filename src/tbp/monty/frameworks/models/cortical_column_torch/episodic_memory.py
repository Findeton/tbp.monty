# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hippocampal-like episodic memory: one-shot storage with novelty gating.

Biologically, this corresponds to the hippocampal formation (HPC):
- Rapid one-shot encoding of individual experiences (DG/CA3)
- Novelty detection via pattern separation (DG)
- Pattern completion for retrieval (CA3 recurrent connections)
- Replay for offline consolidation back to cortex (sharp-wave ripples)

This module lives OUTSIDE the cortical column.  The column sends its
activation patterns here for episodic storage; the episodic memory can
later replay stored episodes for cortical consolidation.

Contrast with the cortical column's Hopfield memory, which stores a
small number of slowly-learned object-level attractors (L2/3 recurrence).
"""

from __future__ import annotations

import torch


class EpisodicMemory:
    """Hippocampal episodic memory with one-shot storage and novelty gating.

    Parameters
    ----------
    n_cells : int
        Dimensionality of stored patterns.
    max_episodes : int
        Maximum stored episodes (ring buffer).
    novelty_threshold : float or None
        Cosine similarity threshold for novelty gating.  A new pattern
        is only stored if its max cosine similarity to all existing
        patterns is below this threshold (pattern separation).
        None disables gating.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        max_episodes: int = 2000,
        novelty_threshold: float | None = 0.7,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.max_episodes = max_episodes
        self._novelty_threshold = novelty_threshold
        self.device = torch.device(device)

        # Stored patterns: (n_stored, n_cells)
        self._patterns = torch.zeros(
            0, n_cells, dtype=torch.float32, device=self.device
        )
        # Labels parallel to patterns (None = unlabeled)
        self._labels: list[str | None] = []
        self._n_stored = 0

    @property
    def n_stored(self) -> int:
        return self._n_stored

    def store(
        self,
        pattern: torch.Tensor,
        label: str | None = None,
        novelty_threshold: float | None = None,
    ) -> bool:
        """Store an episode if sufficiently novel (pattern separation).

        Stores the raw pattern to preserve magnitude information.
        Returns True if the pattern was stored, False if rejected.
        """
        with torch.no_grad():
            p = pattern.detach().to(self.device).float()

            thresh = (novelty_threshold if novelty_threshold is not None
                      else self._novelty_threshold)

            # Novelty gate: cosine similarity check
            if thresh is not None and self._n_stored > 0:
                n = min(self._n_stored, self.max_episodes)
                p_unit = p / (p.norm() + 1e-8)
                stored_norms = (
                    self._patterns[:n].norm(dim=1, keepdim=True) + 1e-8
                )
                stored_unit = self._patterns[:n] / stored_norms
                sims = torch.mv(stored_unit, p_unit)
                if sims.max().item() >= thresh:
                    return False

            if self._n_stored < self.max_episodes:
                self._patterns = torch.cat(
                    [self._patterns, p.unsqueeze(0)], dim=0
                )
                self._labels.append(label)
                self._n_stored += 1
            else:
                # Ring buffer: overwrite oldest
                idx = self._n_stored % self.max_episodes
                self._patterns[idx] = p
                self._labels[idx] = label
                self._n_stored += 1

            return True

    def retrieve_by_label(self, label: str) -> torch.Tensor | None:
        """Get all stored patterns with a given label.

        Returns (k, n_cells) tensor or None if no patterns match.
        """
        n = min(self._n_stored, self.max_episodes)
        indices = [
            i for i in range(n) if self._labels[i] == label
        ]
        if not indices:
            return None
        return self._patterns[indices]

    def replay_batch(self, batch_size: int = 32) -> list[tuple[torch.Tensor, str | None]]:
        """Sample a random batch for replay (sharp-wave ripple analog).

        Returns list of (pattern, label) tuples.
        """
        n = min(self._n_stored, self.max_episodes)
        if n == 0:
            return []

        k = min(batch_size, n)
        indices = torch.randperm(n)[:k]
        return [
            (self._patterns[i].clone(), self._labels[i])
            for i in indices
        ]

    @property
    def labels(self) -> list[str | None]:
        n = min(self._n_stored, self.max_episodes)
        return self._labels[:n]

    @property
    def known_labels(self) -> set[str]:
        n = min(self._n_stored, self.max_episodes)
        return {lbl for lbl in self._labels[:n] if lbl is not None}

    def clear(self) -> None:
        """Remove all stored episodes."""
        self._patterns = torch.zeros(
            0, self.n_cells, dtype=torch.float32, device=self.device
        )
        self._labels = []
        self._n_stored = 0

    def state_dict(self) -> dict:
        return {
            "patterns": self._patterns.cpu(),
            "labels": list(self._labels),
            "n_stored": self._n_stored,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._patterns = sd["patterns"].to(self.device)
        self._labels = sd["labels"]
        self._n_stored = sd["n_stored"]
