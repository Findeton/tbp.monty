# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""LearningModule adapter for CorticalColumn.

Wraps :class:`CorticalColumn` in the :class:`LearningModule` interface so it
can be used as a drop-in replacement for :class:`EvidenceGraphLM` inside
:class:`MontyBase`.

Usage::

    from tbp.monty.frameworks.models.cortical_column.learning_module import (
        CorticalColumnLM,
    )

    lm = CorticalColumnLM(column_kwargs=dict(use_weight_memory=True))
    lm.set_experiment_mode(ExperimentMode.TRAIN)
    lm.pre_episode()
    lm.matching_step(ctx, observations)
    lm.post_episode()
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn
from tbp.monty.frameworks.models.states import GoalState, State

logger = logging.getLogger(__name__)


class CorticalColumnLM(LearningModule):
    """LearningModule adapter that wraps a CorticalColumn.

    This adapter implements all 12 abstract methods of :class:`LearningModule`
    by delegating to a composed :class:`CorticalColumn` instance. The inner
    column remains a clean, framework-independent implementation.

    Parameters
    ----------
    column_kwargs : dict or None
        Keyword arguments forwarded to :class:`CorticalColumn`.
    learning_module_id : str
        Unique identifier for this LM within a Monty model.
    vote_evidence_threshold : float
        Minimum scaled evidence for a hypothesis to be included in votes.
    output_evidence_threshold : float
        Minimum evidence for get_output to report a confident hypothesis.
    surprise_gated_output : bool
        If True, get_output returns use_state=False when surprise is low.
    output_surprise_threshold : float
        Surprise below this value triggers gated output.
    """

    def __init__(
        self,
        column_kwargs: dict | None = None,
        learning_module_id: str = "CorticalColumnLM_0",
        vote_evidence_threshold: float = 0.5,
        output_evidence_threshold: float = 1.0,
        surprise_gated_output: bool = False,
        output_surprise_threshold: float = 0.3,
    ):
        self.learning_module_id = learning_module_id
        self._vote_evidence_threshold = vote_evidence_threshold
        self._output_evidence_threshold = output_evidence_threshold
        self._surprise_gated_output = surprise_gated_output
        self._output_surprise_threshold = output_surprise_threshold

        # Inner column — all SDR/dendrite/attractor logic lives here.
        self._column = CorticalColumn(**(column_kwargs or {}))

        # Experiment state
        self._mode: ExperimentMode | None = None
        self._stepped = False
        self._last_result: dict = {}
        self._last_input_state: State | None = None
        self._primary_target: str | None = None
        self._primary_target_object: str | None = None
        self._primary_target_state: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle (LearningModule interface)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset before a new training/eval run."""
        self._stepped = False
        self._last_result = {}
        self._last_input_state = None

    def pre_episode(self, primary_target=None, **kwargs) -> None:
        """Reset for a new episode."""
        self._stepped = False
        self._last_result = {}
        self._last_input_state = None
        self._primary_target = primary_target

        # Determine object name (and optional state) for training
        object_name = None
        state_id = None
        if self._mode is ExperimentMode.TRAIN and primary_target is not None:
            if isinstance(primary_target, dict):
                object_name = primary_target.get("object")
                state_id = primary_target.get("state")
            elif isinstance(primary_target, str):
                object_name = primary_target
            self._primary_target_object = object_name
            self._primary_target_state = state_id

        # State-conditioned: composite key so each (object, state) pair gets
        # a distinct label SDR in associative memory.
        composite_name = object_name
        if object_name and state_id is not None:
            composite_name = f"{object_name}:{state_id}"

        mode_str = "train" if self._mode is ExperimentMode.TRAIN else "eval"
        self._column.pre_episode(mode=mode_str, object_name=composite_name)

    def post_episode(self) -> None:
        """Finalize episode."""
        self._column.post_episode()

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        """Set train or eval mode."""
        self._mode = mode

    # ------------------------------------------------------------------
    # Core algorithm (LearningModule interface)
    # ------------------------------------------------------------------

    def matching_step(self, ctx, observations=None) -> None:
        """Inference step — process observations and update evidence.

        Parameters
        ----------
        ctx : RuntimeContext
        observations : list[State] or None
            Observations from connected sensor modules and/or parent LMs.
        """
        state = self._extract_usable_state(observations)
        if state is None:
            self._last_result = self._column._empty_result()
            return

        self._last_input_state = state
        self._last_result = self._column.step(state)
        self._stepped = True

    def exploratory_step(self, ctx, observations=None) -> None:
        """Training step — same as matching_step (column learns based on mode).

        CorticalColumn's learning is controlled by its internal mode (set in
        pre_episode), not by which step method is called. Both matching and
        exploratory steps pass observations through the same column.step().
        """
        state = self._extract_usable_state(observations)
        if state is None:
            self._last_result = self._column._empty_result()
            return

        self._last_input_state = state
        self._last_result = self._column.step(state)
        self._stepped = True

    # ------------------------------------------------------------------
    # Voting (LearningModule interface)
    # ------------------------------------------------------------------

    def send_out_vote(self) -> dict | None:
        """Return vote data for other LMs.

        Returns the same format as EvidenceGraphLM: a dict with
        ``possible_states`` mapping object IDs to lists of State objects,
        and ``sensed_pose_rel_body``.
        """
        if not self._stepped:
            return None

        evidence = self._last_result.get("evidence", {})
        if not evidence:
            return None

        # Scale evidence to [-1, 1]
        values = list(evidence.values())
        max_ev = max(values) if values else 1.0
        min_ev = min(values) if values else 0.0
        ev_range = max_ev - min_ev if max_ev != min_ev else 1.0

        possible_states = {}
        for obj_id, ev in evidence.items():
            scaled = (2.0 * (ev - min_ev) / ev_range - 1.0) if ev_range > 0 else 0.0
            if scaled < self._vote_evidence_threshold:
                continue

            # Build a State representing this hypothesis
            location = (
                np.asarray(self._last_input_state.location, dtype=np.float64)
                if self._last_input_state is not None
                else np.zeros(3)
            )
            pose = (
                self._last_input_state.morphological_features.get(
                    "pose_vectors", np.eye(3)
                )
                if self._last_input_state is not None
                else np.eye(3)
            )

            vote_state = State(
                location=location,
                morphological_features={
                    "pose_vectors": pose,
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=scaled,
                use_state=True,
                sender_id=self.learning_module_id,
                sender_type="LM",
            )
            possible_states[obj_id] = [vote_state]

        if not possible_states:
            return None

        sensed_pose = (
            np.asarray(self._last_input_state.location, dtype=np.float64)
            if self._last_input_state is not None
            else np.zeros(3)
        )

        return {
            "possible_states": possible_states,
            "sensed_pose_rel_body": sensed_pose,
        }

    def receive_votes(self, votes) -> None:
        """Process votes from other LMs to adjust evidence.

        Parameters
        ----------
        votes : list[dict | None]
            Each element is the output of another LM's send_out_vote().
        """
        if votes is None:
            return

        for vote_data in votes:
            if vote_data is None:
                continue
            possible_states = vote_data.get("possible_states", {})
            for obj_id, state_list in possible_states.items():
                if obj_id not in self._column._evidence:
                    continue
                # Sum confidence from all voting states for this object
                vote_boost = sum(
                    s.confidence for s in state_list if hasattr(s, "confidence")
                )
                self._column._evidence[obj_id] += vote_boost

    # ------------------------------------------------------------------
    # Hierarchy output (LearningModule interface)
    # ------------------------------------------------------------------

    def get_output(self) -> State | None:
        """Return this LM's recognition result as a State for parent LMs.

        When surprise_gated_output is enabled, returns use_state=False if
        the column's surprise is below threshold (prediction confirmed).
        """
        if not self._stepped:
            return None

        mlh = self._column.get_current_mlh()
        surprise = self._column.surprise

        location = (
            np.asarray(self._last_input_state.location, dtype=np.float64)
            if self._last_input_state is not None
            else np.zeros(3)
        )
        pose = (
            self._last_input_state.morphological_features.get(
                "pose_vectors", np.eye(3)
            )
            if self._last_input_state is not None
            else np.eye(3)
        )

        # Surprise-gated: confirmed predictions send minimal output
        if (
            self._surprise_gated_output
            and surprise < self._output_surprise_threshold
        ):
            return State(
                location=location,
                morphological_features={
                    "pose_vectors": pose,
                    "pose_fully_defined": True,
                },
                non_morphological_features={
                    "confirmed": True,
                    "surprise": surprise,
                    "graph_id": mlh.get("graph_id"),
                },
                confidence=1.0,
                use_state=False,
                sender_id=self.learning_module_id,
                sender_type="LM",
            )

        # Normal output: full hypothesis
        ev = mlh.get("evidence", 0.0)
        confident = ev > self._output_evidence_threshold

        # Parse composite key back into object + state
        graph_id = mlh.get("graph_id")
        inferred_obj = graph_id
        inferred_state = None
        if graph_id and ":" in graph_id:
            parts = graph_id.rsplit(":", 1)
            inferred_obj = parts[0]
            try:
                inferred_state = int(parts[1])
            except ValueError:
                inferred_state = None

        return State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
            },
            non_morphological_features={
                "graph_id": inferred_obj,
                "evidence": ev,
                "surprise": surprise,
            },
            confidence=min(ev / max(self._output_evidence_threshold, 1e-6), 1.0),
            use_state=confident,
            sender_id=self.learning_module_id,
            sender_type="LM",
            inferred_state=inferred_state,
        )

    def propose_goal_states(self) -> list:
        """Return goal states for motor policy. Currently empty."""
        return []

    # ------------------------------------------------------------------
    # Context (LearningModule interface)
    # ------------------------------------------------------------------

    def receive_context(self, **context_signal) -> None:
        """Forward context signal to inner column's apical dendrites."""
        self._column.receive_context(**context_signal)

    def get_context_signal(self) -> dict | None:
        """Return context signal for child LMs."""
        if not self._stepped:
            return None
        return self._column.get_context_signal()

    # ------------------------------------------------------------------
    # Persistence (LearningModule interface)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize column state for saving."""
        # CorticalColumn doesn't have state_dict yet — serialize key arrays
        sd = {
            "learning_module_id": self.learning_module_id,
            "evidence": dict(self._column._evidence),
        }
        if self._column._use_weight_memory and self._column._associative_memory:
            am = self._column._associative_memory
            sd["associative_memory"] = {
                "cell_weights": am._cell_weights.copy(),
                "ff_weights": (
                    am._ff_weights.copy() if am._ff_weights is not None else None
                ),
                "labels": dict(am._labels),
            }
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore column state from saved dict."""
        self.learning_module_id = state_dict.get(
            "learning_module_id", self.learning_module_id
        )
        ev = state_dict.get("evidence", {})
        self._column._evidence = dict(ev)

        am_state = state_dict.get("associative_memory")
        if am_state and self._column._associative_memory:
            am = self._column._associative_memory
            if am_state.get("cell_weights") is not None:
                am._cell_weights[:] = am_state["cell_weights"]
            if am_state.get("ff_weights") is not None and am._ff_weights is not None:
                am._ff_weights[:] = am_state["ff_weights"]
            am._labels = dict(am_state.get("labels", {}))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def column(self) -> CorticalColumn:
        """The wrapped CorticalColumn instance."""
        return self._column

    @staticmethod
    def _parse_composite_key(key: str) -> tuple:
        """Split 'object:state' back into (object_name, state_id).

        Returns (key, None) if there's no state component.
        """
        if key and ":" in key:
            parts = key.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return key, None
        return key, None

    def get_all_known_object_ids(self) -> list[str]:
        """Return all trained object IDs (composite keys including state)."""
        return self._column.get_all_known_object_ids()

    def get_known_objects(self) -> list[str]:
        """Return unique object names (without state suffixes)."""
        keys = self._column.get_all_known_object_ids()
        return list(set(self._parse_composite_key(k)[0] for k in keys))

    def get_current_mlh(self) -> dict:
        """Return the most likely hypothesis."""
        return self._column.get_current_mlh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_state_like(obj) -> bool:
        """Check if an object looks like a State (duck typing)."""
        return hasattr(obj, "use_state") and hasattr(obj, "location")

    def _extract_usable_state(self, observations) -> State | None:
        """Pick the first usable State from observations.

        Handles the observation formats that MontyBase provides:
        - list[State] from sensor modules
        - list[list[State]] from combined SM + LM inputs
        - None / empty

        Uses duck typing so both real State objects and test fakes work.
        """
        if observations is None:
            return None

        # Flatten if nested
        states = []
        if isinstance(observations, list):
            for item in observations:
                if self._is_state_like(item):
                    states.append(item)
                elif isinstance(item, list):
                    states.extend(
                        s for s in item if self._is_state_like(s)
                    )
                elif isinstance(item, dict):
                    for v in item.values():
                        if self._is_state_like(v):
                            states.append(v)
        elif self._is_state_like(observations):
            states = [observations]

        # Process LM inputs as context (top-down)
        for s in states:
            if (
                getattr(s, "sender_type", None) == "LM"
                and getattr(s, "use_state", False)
            ):
                nmf = getattr(s, "non_morphological_features", None)
                if nmf and isinstance(nmf, dict):
                    ctx = nmf.get("active_cells")
                    if ctx is not None:
                        self._column.receive_context(active_cells=ctx)

        # Find first usable SM state
        for s in states:
            if (
                getattr(s, "use_state", False)
                and getattr(s, "sender_type", "SM") != "LM"
            ):
                return s

        # Fallback: any usable state
        for s in states:
            if getattr(s, "use_state", False):
                return s

        return None
