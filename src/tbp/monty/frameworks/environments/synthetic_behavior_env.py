# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Synthetic environment for behavior testbed.

Replays pre-computed State sequences as observations, enabling end-to-end
testing of behavior learning and recognition without GPU or Habitat.
Each State represents a single observation from a sensor module.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from tbp.monty.frameworks.models.states import State

logger = logging.getLogger(__name__)


class SyntheticBehaviorEnvironment:
    """Replays pre-computed State sequences as observations.

    This environment holds a collection of named behavior sequences. Each
    sequence is a list of States. On each step, the environment returns the
    next State in the current sequence. Supports multiple input channels
    (e.g., morphology SM + behavior SM).

    Parameters
    ----------
    sequences : dict[str, list[State]]
        Named behavior sequences. Keys are behavior IDs, values are lists
        of States.
    active_sequence : str or None
        Which sequence to replay. Set via set_active_sequence().
    """

    def __init__(
        self,
        sequences: Optional[Dict[str, List[State]]] = None,
    ):
        self._sequences = sequences or {}
        self._active_sequence: Optional[str] = None
        self._step_index = 0

    def add_sequence(self, name: str, states: List[State]) -> None:
        """Add a behavior sequence.

        Args:
            name: Behavior identifier.
            states: Ordered list of States.
        """
        self._sequences[name] = states

    def set_active_sequence(self, name: str) -> None:
        """Set which sequence to replay."""
        if name not in self._sequences:
            raise KeyError(f"Unknown sequence: {name}")
        self._active_sequence = name
        self._step_index = 0

    def reset(self) -> None:
        """Reset to the beginning of the active sequence."""
        self._step_index = 0

    def step(self) -> Optional[State]:
        """Return the next observation in the active sequence.

        Returns None if the sequence is exhausted.
        """
        if self._active_sequence is None:
            return None

        seq = self._sequences[self._active_sequence]
        if self._step_index >= len(seq):
            return None

        state = seq[self._step_index]
        self._step_index += 1
        return state

    def get_remaining_steps(self) -> int:
        """Return number of remaining steps in active sequence."""
        if self._active_sequence is None:
            return 0
        return max(0, len(self._sequences[self._active_sequence]) - self._step_index)

    def get_sequence_length(self, name: Optional[str] = None) -> int:
        """Return length of a sequence (active by default)."""
        name = name or self._active_sequence
        if name is None or name not in self._sequences:
            return 0
        return len(self._sequences[name])

    def get_sequence_names(self) -> List[str]:
        """Return names of all registered sequences."""
        return list(self._sequences.keys())

    @property
    def is_done(self) -> bool:
        """Whether the active sequence has been fully replayed."""
        return self.get_remaining_steps() == 0


def make_behavior_sequences(
    behaviors: Dict[str, callable],
    n_steps: int = 40,
    **kwargs,
) -> Dict[str, List[State]]:
    """Generate behavior sequences from behavior generator functions.

    Args:
        behaviors: Dict mapping behavior name to generator function.
            Each function should accept n_steps as first arg and return
            a list of States.
        n_steps: Number of steps per behavior.
        **kwargs: Additional kwargs passed to each generator.

    Returns:
        Dict mapping behavior name to list of States.
    """
    sequences = {}
    for name, gen_fn in behaviors.items():
        sequences[name] = gen_fn(n_steps=n_steps, **kwargs)
    return sequences
