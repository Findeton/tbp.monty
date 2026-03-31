# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""LearningModule adapter for CorticalColumnTorch.

Wraps :class:`CorticalColumnTorch` in the :class:`LearningModule` interface
so it can be used as a drop-in replacement for :class:`EvidenceGraphLM` or
:class:`CorticalColumnLM` inside :class:`MontyForGraphMatching`.

Implements all LearningModule ABC methods plus the non-ABC attributes that
MontyForGraphMatching, its loggers, and its voting infrastructure depend on.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer
from tbp.monty.frameworks.models.cortical_column_torch.column import (
    CorticalColumnTorch,
)
from tbp.monty.frameworks.models.states import GoalState, State

logger = logging.getLogger(__name__)


class CorticalColumnTorchLM(LearningModule):
    """LearningModule adapter wrapping CorticalColumnTorch.

    Same adapter pattern as CorticalColumnLM (Track 8), but wrapping the
    PyTorch-based modern Hopfield column instead of the numpy SDR column.

    Parameters
    ----------
    column_kwargs : dict or None
        Keyword arguments forwarded to CorticalColumnTorch.
    learning_module_id : str
        Unique identifier for this LM.
    vote_evidence_threshold : float
        Minimum scaled evidence to include in votes.
    output_evidence_threshold : float
        Minimum evidence for confident output.
    surprise_gated_output : bool
        If True, get_output returns use_state=False when surprise is low.
    output_surprise_threshold : float
        Surprise below this triggers gated output.
    max_match_steps : int
        Max matching steps before time_out.
    evidence_match_threshold : float
        Evidence above this triggers match.
    evidence_separation_ratio : float
        Top evidence must exceed second-best by this factor.
    """

    def __init__(
        self,
        column_kwargs: dict | None = None,
        learning_module_id: str = "CorticalColumnTorchLM_0",
        vote_evidence_threshold: float = 0.5,
        output_evidence_threshold: float = 1.0,
        surprise_gated_output: bool = False,
        output_surprise_threshold: float = 0.3,
        max_match_steps: int = 100,
        evidence_match_threshold: float = 3.0,
        evidence_separation_ratio: float = 1.5,
        hopfield_voting: bool = False,
        surprise_vote_threshold: float = 0.3,
    ):
        self.learning_module_id = learning_module_id
        self._vote_evidence_threshold = vote_evidence_threshold
        self._output_evidence_threshold = output_evidence_threshold
        self._surprise_gated_output = surprise_gated_output
        self._output_surprise_threshold = output_surprise_threshold
        self._max_match_steps = max_match_steps
        self._evidence_match_threshold = evidence_match_threshold
        self._evidence_separation_ratio = evidence_separation_ratio
        self._hopfield_voting = hopfield_voting
        self._surprise_vote_threshold = surprise_vote_threshold

        # Inner column
        self._column = CorticalColumnTorch(**(column_kwargs or {}))

        # ---- MontyForGraphMatching compatibility attributes ----
        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.primary_target = None
        self.primary_target_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.symmetry_evidence = 0
        self.has_detailed_logger = False
        self.graph_id_to_target = {}
        self.target_to_graph_id = {}
        self.mode = None
        self.possible_matches = {}
        self.possible_poses = {}
        self.possible_paths = {}
        self.pose_similarity_threshold = 0.35
        self.evidence = {}

        # Internal
        self._mode: ExperimentMode | None = None
        self._stepped = False
        self._step_count = 0
        self._last_result: dict = {}
        self._last_input_state: State | None = None
        self._primary_target_object: str | None = None
        self._primary_target_state: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._stepped = False
        self._step_count = 0
        self._last_result = {}
        self._last_input_state = None
        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.possible_matches = {}
        self.evidence = {}

    def pre_episode(self, primary_target=None, **kwargs) -> None:
        self._stepped = False
        self._step_count = 0
        self._last_result = {}
        self._last_input_state = None
        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.stepwise_target_object = None
        self.stepwise_targets_list = []
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.symmetry_evidence = 0
        self.possible_matches = {}
        self.evidence = {}

        # Extract target
        if primary_target is not None:
            if isinstance(primary_target, dict):
                self.primary_target = primary_target.get("object")
                self.primary_target_rotation_quat = primary_target.get(
                    "quat_rotation", np.array([1.0, 0.0, 0.0, 0.0])
                )
                state_id = primary_target.get("state")
            elif isinstance(primary_target, str):
                self.primary_target = primary_target
                self.primary_target_rotation_quat = np.array([1., 0., 0., 0.])
                state_id = None
            else:
                self.primary_target = None
                self.primary_target_rotation_quat = np.array([1., 0., 0., 0.])
                state_id = None
        else:
            self.primary_target = None
            self.primary_target_rotation_quat = np.array([1.0, 0.0, 0.0, 0.0])
            state_id = None

        self._primary_target_object = self.primary_target
        self._primary_target_state = state_id

        # Composite name
        object_name = self._primary_target_object
        composite_name = object_name
        if object_name and state_id is not None:
            composite_name = f"{object_name}:{state_id}"

        column_obj = composite_name if self._mode is ExperimentMode.TRAIN else None
        mode_str = "train" if self._mode is ExperimentMode.TRAIN else "eval"
        self._column.pre_episode(mode=mode_str, object_name=column_obj)

        if (
            self._mode is ExperimentMode.TRAIN
            and self.primary_target is not None
            and composite_name is not None
        ):
            self.graph_id_to_target.setdefault(composite_name, set()).add(
                self.primary_target
            )
            self.target_to_graph_id.setdefault(self.primary_target, set()).add(
                composite_name
            )

    def post_episode(self) -> None:
        self._column.post_episode()

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self._mode = mode
        self.mode = mode

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def matching_step(self, ctx, observations=None) -> None:
        state = self._extract_usable_state(observations)
        if state is None:
            self._last_result = self._column._empty_result()
            return

        self._last_input_state = state
        self._last_result = self._column.step(state)
        self._stepped = True
        self._step_count += 1

        self.evidence = dict(self._column._evidence)
        self._update_possible_matches()
        self._auto_update_terminal_condition()
        self._buffer_observation(state)

    def exploratory_step(self, ctx, observations=None) -> None:
        state = self._extract_usable_state(observations)
        if state is None:
            self._last_result = self._column._empty_result()
            return

        self._last_input_state = state
        self._last_result = self._column.step(state)
        self._stepped = True
        self._step_count += 1
        self.evidence = dict(self._column._evidence)
        self._buffer_observation(state)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def send_out_vote(self) -> dict | None:
        if not self._stepped:
            return None

        # Surprise-gated voting: only vote when the column has converged
        # (low surprise). This prevents noise propagation from uncertain LMs
        # and is biologically analogous to lateral connections firing only
        # when a column has settled into an attractor state.
        if self._hopfield_voting:
            if self._column.surprise > self._surprise_vote_threshold:
                return None

        evidence = self._last_result.get("evidence", {})
        if not evidence:
            return None

        values = list(evidence.values())
        max_ev = max(values) if values else 1.0
        min_ev = min(values) if values else 0.0
        ev_range = max_ev - min_ev if max_ev != min_ev else 1.0

        possible_states = {}
        for obj_id, ev in evidence.items():
            scaled = (2.0 * (ev - min_ev) / ev_range - 1.0) if ev_range > 0 else 0.0
            if scaled < self._vote_evidence_threshold:
                continue

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

        # sensed_pose_rel_body must be (4, 3): location stacked with 3x3 pose
        # vectors, matching the format that _combine_votes expects.
        location = (
            np.asarray(self._last_input_state.location, dtype=np.float64)
            if self._last_input_state is not None
            else np.zeros(3)
        )
        pose = (
            np.asarray(
                self._last_input_state.morphological_features.get(
                    "pose_vectors", np.eye(3)
                ),
                dtype=np.float64,
            ).reshape(3, 3)
            if self._last_input_state is not None
            else np.eye(3)
        )
        sensed_pose = np.vstack([location.reshape(1, 3), pose])

        vote = {
            "possible_states": possible_states,
            "sensed_pose_rel_body": sensed_pose,
        }

        # Hopfield voting: include the sender's settled activation pattern.
        # The receiver uses this as a retrieval cue in its own Hopfield
        # memory, biologically analogous to lateral cortical connections
        # sharing attractor states between columns.
        if self._hopfield_voting:
            active = self._column._active
            if active.abs().sum() > 0:
                vote["hopfield_pattern"] = active.detach().cpu()

        return vote

    def receive_votes(self, votes) -> None:
        if votes is None:
            return

        if isinstance(votes, dict):
            # Check for Hopfield pattern in the vote (surprise-gated voting)
            hopfield_pattern = votes.get("hopfield_pattern")
            if hopfield_pattern is not None and self._hopfield_voting:
                self._apply_hopfield_vote(hopfield_pattern)

            # Standard evidence boosting
            possible_states = votes.get("possible_states", votes)
            if isinstance(possible_states, dict):
                for obj_id, vote_states in possible_states.items():
                    if obj_id == "hopfield_pattern":
                        continue
                    if obj_id not in self._column._evidence:
                        continue
                    if isinstance(vote_states, list):
                        boost = sum(
                            getattr(s, "confidence", 0)
                            for s in vote_states
                        )
                    else:
                        boost = getattr(vote_states, "confidence", 0)
                    self._column._evidence[obj_id] += boost
            self.evidence = dict(self._column._evidence)
            self._update_possible_matches()
            return

        for vote_data in votes:
            if vote_data is None:
                continue

            # Extract Hopfield pattern if present
            if isinstance(vote_data, dict):
                hopfield_pattern = vote_data.get("hopfield_pattern")
                if hopfield_pattern is not None and self._hopfield_voting:
                    self._apply_hopfield_vote(hopfield_pattern)

                possible_states = vote_data.get("possible_states", {})
            else:
                possible_states = {}

            for obj_id, state_list in possible_states.items():
                if obj_id not in self._column._evidence:
                    continue
                vote_boost = sum(
                    s.confidence for s in state_list
                    if hasattr(s, "confidence")
                )
                self._column._evidence[obj_id] += vote_boost
        self.evidence = dict(self._column._evidence)
        self._update_possible_matches()

    def _apply_hopfield_vote(self, pattern) -> None:
        """Use a Hopfield pattern from another LM as a retrieval cue.

        The sender's settled activation pattern is used as a query into this
        LM's own associative memory. The resulting similarity scores bias
        evidence toward objects that match the lateral signal.

        This is biologically analogous to lateral cortical connections
        providing correlated attractor states between columns, accelerating
        convergence when two columns observe complementary views of the
        same object.
        """
        import torch

        cue = pattern.to(self._column.device).float()

        # Use the cue as a Hopfield retrieval query
        retrieved = self._column._hopfield.retrieve(cue)

        # Feed retrieved pattern through associative memory for evidence
        if retrieved.abs().sum() > 0:
            scores = self._column._associative_memory.recall(retrieved)
            for obj_name, score in scores.items():
                if obj_name in self._column._evidence:
                    # Scale the lateral boost — it's supplementary, not primary
                    self._column._evidence[obj_name] += score * 0.5

    # ------------------------------------------------------------------
    # Hierarchy output
    # ------------------------------------------------------------------

    def get_output(self) -> State | None:
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

        # Surprise-gated output
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

        ev = mlh.get("evidence", 0.0)
        confident = ev > self._output_evidence_threshold

        graph_id = mlh.get("graph_id")
        inferred_obj = graph_id
        inferred_state = None
        if graph_id and ":" in str(graph_id):
            parts = str(graph_id).rsplit(":", 1)
            inferred_obj = parts[0]
            try:
                inferred_state = int(parts[1])
            except ValueError:
                inferred_state = None

        object_id_features = (
            sum(ord(c) for c in str(inferred_obj)) if inferred_obj else 0
        )

        # Active cells as numpy for compatibility with hierarchy context
        active_cells = self._column._active.cpu().numpy()

        return State(
            location=location,
            morphological_features={
                "pose_vectors": pose,
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "object_id": object_id_features,
                "graph_id": str(inferred_obj) if inferred_obj else "unknown",
                "evidence": ev,
                "surprise": surprise,
                "active_cells": active_cells,
            },
            confidence=min(ev / max(self._output_evidence_threshold, 1e-6), 1.0),
            use_state=confident,
            sender_id=self.learning_module_id,
            sender_type="LM",
            inferred_state=inferred_state,
        )

    def propose_goal_states(self) -> list:
        return []

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def receive_context(self, **context_signal) -> None:
        self._column.receive_context(**context_signal)

        # Parent LM pattern: if this LM was NOT stepped by a sensor module
        # this cycle, drive it from the received context.  This makes parent
        # columns in a heterarchy functional — they aggregate children's
        # activations through their own Hopfield + associative memory.
        if not self._stepped and context_signal.get("active_cells") is not None:
            result = self._column.step_from_context()
            self._last_result = result
            self._stepped = True
            self._step_count += 1
            self.evidence = dict(self._column._evidence)
            self._update_possible_matches()

    def get_context_signal(self) -> dict | None:
        if not self._stepped:
            return None
        return self._column.get_context_signal()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        sd = {
            "learning_module_id": self.learning_module_id,
            "column": self._column.state_dict(),
            "graph_id_to_target": {
                k: list(v) for k, v in self.graph_id_to_target.items()
            },
            "target_to_graph_id": {
                k: list(v) for k, v in self.target_to_graph_id.items()
            },
        }
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self.learning_module_id = state_dict.get(
            "learning_module_id", self.learning_module_id
        )
        col_sd = state_dict.get("column")
        if col_sd:
            self._column.load_state_dict(col_sd)
        self.evidence = dict(self._column._evidence)

        for k, v in state_dict.get("graph_id_to_target", {}).items():
            self.graph_id_to_target[k] = set(v)
        for k, v in state_dict.get("target_to_graph_id", {}).items():
            self.target_to_graph_id[k] = set(v)

    # ------------------------------------------------------------------
    # Non-ABC methods for MontyForGraphMatching compatibility
    # ------------------------------------------------------------------

    def get_possible_matches(self) -> list:
        return list(self.possible_matches.keys())

    def collect_stats_to_save(self) -> dict:
        return {"possible_matches": self.get_possible_matches()}

    def set_individual_ts(self, terminal_state) -> None:
        self.terminal_state = terminal_state
        if terminal_state == "match":
            mlh = self._column.get_current_mlh()
            self.detected_object = mlh.get("graph_id")
        elif terminal_state == "no_match":
            self.detected_object = None
        else:
            self.detected_object = terminal_state
        self.buffer.stats["individual_ts_reached_at_step"] = self._step_count
        self.buffer.stats["individual_ts_object"] = self.detected_object

    def update_terminal_condition(self) -> str | None:
        matches = self.get_possible_matches()
        n = len(matches)

        if n == 0 and self._stepped:
            self.set_individual_ts("no_match")
        elif n == 1:
            self.set_individual_ts("match")
        elif n > 0:
            evidence = self.evidence
            if evidence:
                sorted_ev = sorted(evidence.values(), reverse=True)
                if len(sorted_ev) >= 2 and sorted_ev[0] > 0:
                    if (
                        sorted_ev[0] > self._evidence_match_threshold
                        and sorted_ev[0]
                        > sorted_ev[1] * self._evidence_separation_ratio
                    ):
                        self.set_individual_ts("match")

        return self.terminal_state

    def add_lm_processing_to_buffer_stats(self, lm_processed: bool) -> None:
        self.buffer.update_stats(
            dict(lm_processed_steps=lm_processed), update_time=False
        )

    def get_unique_pose_if_available(self, object_id):
        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def column(self) -> CorticalColumnTorch:
        return self._column

    def get_all_known_object_ids(self) -> list[str]:
        return self._column.get_all_known_object_ids()

    def get_known_objects(self) -> list[str]:
        keys = self._column.get_all_known_object_ids()
        return list(set(self._parse_composite_key(k)[0] for k in keys))

    def get_current_mlh(self) -> dict:
        mlh = self._column.get_current_mlh()
        graph_id = mlh.get("graph_id", "no_observations_yet")
        evidence_val = mlh.get("evidence", 0.0)

        location = (
            np.asarray(self._last_input_state.location, dtype=np.float64)
            if self._last_input_state is not None
            else np.zeros(3)
        )

        return {
            "graph_id": graph_id if graph_id else "no_observations_yet",
            "location": location,
            "rotation": Rotation.identity(),
            "evidence": evidence_val,
            "scale": 1.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_composite_key(key: str) -> tuple:
        if key and ":" in str(key):
            parts = str(key).rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return key, None
        return key, None

    def _update_possible_matches(self):
        evidence = self._column._evidence
        if not evidence:
            self.possible_matches = {}
            return
        max_ev = max(evidence.values()) if evidence else 0
        threshold = max_ev * 0.1 if max_ev > 0 else 0
        self.possible_matches = {
            obj_id: ev for obj_id, ev in evidence.items() if ev > threshold
        }

    def _auto_update_terminal_condition(self):
        if self._mode is not ExperimentMode.EVAL:
            return
        self.update_terminal_condition()

    def _buffer_observation(self, state: State) -> None:
        try:
            self.buffer.append([state])
        except (ValueError, TypeError, KeyError, AttributeError):
            pass

    @staticmethod
    def _is_state_like(obj) -> bool:
        return hasattr(obj, "use_state") and hasattr(obj, "location")

    def _extract_usable_state(self, observations) -> State | None:
        if observations is None:
            return None

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

        # Process LM inputs as context
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

        # First usable SM state
        for s in states:
            if (
                getattr(s, "use_state", False)
                and getattr(s, "sender_type", "SM") != "LM"
            ):
                return s

        # Fallback
        for s in states:
            if getattr(s, "use_state", False):
                return s

        return None
