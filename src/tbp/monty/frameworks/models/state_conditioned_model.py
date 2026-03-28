# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""State-conditioned object model for behavioral and morphological states.

A StateConditionedModel wraps multiple GridObjectModel instances, one per
discrete state (e.g., stapler-open=0, stapler-closed=1). Each sub-model is
a standard spatial graph; the state dimension indexes which sub-graph to use.

This follows the TBP theory: state is a sub-object-ID, not a continuous
4th dimension. Different states may have completely different feature
distributions at different locations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tbp.monty.frameworks.models.object_model import GridObjectModel

logger = logging.getLogger(__name__)


class StateConditionedModel:
    """Object model with multiple discrete states, each a GridObjectModel.

    Parameters
    ----------
    object_id : str
        Identifier for the object.
    max_nodes : int
        Maximum nodes per sub-graph (passed to GridObjectModel).
    max_size : float
        Maximum object size in meters (passed to GridObjectModel).
    num_voxels_per_dim : int
        Voxel resolution per dimension (passed to GridObjectModel).
    """

    is_state_conditioned = True

    def __init__(
        self,
        object_id: str,
        max_nodes: int,
        max_size: float,
        num_voxels_per_dim: int,
    ):
        self.object_id = object_id
        self._max_nodes = max_nodes
        self._max_size = max_size
        self._num_voxels_per_dim = num_voxels_per_dim

        # state_id (int) -> GridObjectModel
        self._state_models: Dict[int, GridObjectModel] = {}

        # Ordered sequence of (from_state, to_state, interval_duration)
        self._transitions: List[Tuple[int, int, float]] = []

    # ======================== Public API ========================

    def build_model(
        self,
        locations: np.ndarray,
        features: dict[str, Any],
        state_id: int = 0,
    ) -> None:
        """Build a sub-graph for the given state.

        Args:
            locations: (N, 3) array of 3D locations.
            features: Dict of feature arrays.
            state_id: Integer state identifier.
        """
        model = self._get_or_create_model(state_id)
        model.build_model(locations=locations, features=features)
        logger.info(
            f"Built state-conditioned sub-graph for state {state_id} "
            f"of object {self.object_id}"
        )

    def update_model(
        self,
        locations: np.ndarray,
        features: dict[str, Any],
        state_id: int = 0,
        location_rel_model: Optional[np.ndarray] = None,
        object_location_rel_body: Optional[np.ndarray] = None,
        object_rotation: Optional[np.ndarray] = None,
    ) -> None:
        """Add observations to the sub-graph for the given state.

        Args:
            locations: (N, 3) array of 3D locations.
            features: Dict of feature arrays.
            state_id: Integer state identifier.
            location_rel_model: Location in model reference frame.
            object_location_rel_body: Object location relative to body.
            object_rotation: Object rotation matrix.
        """
        model = self._get_or_create_model(state_id)
        if location_rel_model is not None:
            model.update_model(
                locations=locations,
                features=features,
                location_rel_model=location_rel_model,
                object_location_rel_body=object_location_rel_body,
                object_rotation=object_rotation,
            )
        else:
            model.build_model(locations=locations, features=features)

    def get_states(self) -> List[int]:
        """Return sorted list of state IDs."""
        return sorted(self._state_models.keys())

    def get_num_states(self) -> int:
        """Return number of states."""
        return len(self._state_models)

    def get_model_for_state(self, state_id: int) -> GridObjectModel:
        """Return the GridObjectModel for a given state.

        Raises:
            KeyError: If state_id not found.
        """
        if state_id not in self._state_models:
            raise KeyError(
                f"State {state_id} not found. "
                f"Available states: {self.get_states()}"
            )
        return self._state_models[state_id]

    def has_state(self, state_id: int) -> bool:
        """Check if a state exists in this model."""
        return state_id in self._state_models

    # ---- Transition sequence ----

    def add_transition(
        self,
        from_state: int,
        to_state: int,
        interval_duration: float = 1.0,
    ) -> None:
        """Add a state transition to the ordered sequence.

        Args:
            from_state: Source state ID.
            to_state: Destination state ID.
            interval_duration: Duration (in timer ticks) between states.
        """
        self._transitions.append((from_state, to_state, interval_duration))

    def get_transitions(self) -> List[Tuple[int, int, float]]:
        """Return the ordered list of state transitions."""
        return list(self._transitions)

    def get_transition_sequence(self) -> List[Tuple[int, int, float]]:
        """Return the ordered list of state transitions.

        Alias for ``get_transitions()``, used by ``_adjust_timer_speed()``
        in EvidenceGraphLM.
        """
        return self.get_transitions()

    def get_next_state(self, current_state: int) -> Optional[int]:
        """Get the next state in the transition sequence.

        Returns None if current_state has no outgoing transition.
        """
        for from_s, to_s, _ in self._transitions:
            if from_s == current_state:
                return to_s
        return None

    # ---- Delegating to sub-models ----

    def find_nearest_neighbors(
        self,
        search_locations: np.ndarray,
        num_neighbors: int,
        state_id: int = 0,
        return_distance: bool = False,
    ):
        """Find nearest neighbors in the sub-graph for a given state.

        Delegates to GridObjectModel.find_nearest_neighbors().
        """
        model = self.get_model_for_state(state_id)
        return model.find_nearest_neighbors(
            search_locations=search_locations,
            num_neighbors=num_neighbors,
            return_distance=return_distance,
        )

    @property
    def _graph(self):
        """Return graph of first state (for backward compat with code
        that accesses model._graph directly)."""
        if not self._state_models:
            return None
        first_state = self.get_states()[0]
        return self._state_models[first_state]._graph

    def __repr__(self):
        states = self.get_states()
        n_nodes = {}
        for sid in states:
            m = self._state_models[sid]
            if m._graph is not None and hasattr(m._graph, "pos"):
                n_nodes[sid] = len(m._graph.pos)
            elif m._graph is not None and hasattr(m._graph, "num_nodes"):
                n_nodes[sid] = m._graph.num_nodes
            else:
                n_nodes[sid] = 0
        return (
            f"StateConditionedModel(object_id={self.object_id}, "
            f"states={states}, nodes_per_state={n_nodes}, "
            f"transitions={len(self._transitions)})"
        )

    # ======================== Private ========================

    def _get_or_create_model(self, state_id: int) -> GridObjectModel:
        """Get existing sub-model or create a new one for the state."""
        if state_id not in self._state_models:
            self._state_models[state_id] = GridObjectModel(
                object_id=f"{self.object_id}_state_{state_id}",
                max_nodes=self._max_nodes,
                max_size=self._max_size,
                num_voxels_per_dim=self._num_voxels_per_dim,
            )
        return self._state_models[state_id]
