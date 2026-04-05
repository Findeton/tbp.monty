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

from collections import Counter, defaultdict
import hashlib
import logging
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.buffer import FeatureAtLocationBuffer
from tbp.monty.frameworks.models.cortical_column_torch.column import (
    CorticalColumnTorch,
)
from tbp.monty.frameworks.models.self_supervised_semi_markov_memory import (
    SelfSupervisedSemiMarkovMemory,
)
from tbp.monty.frameworks.models.semi_markov_transition_memory import (
    SemiMarkovTransitionMemory,
)
from tbp.monty.frameworks.models.states import GoalState, State
from tbp.monty.frameworks.models.temporal_memory import TemporalMemory

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
    goal_state_min_steps : int
        Minimum matched steps before emitting goal states.
    goal_state_min_separation_ratio : float or None
        Optional evidence separation required before emitting goal states.
    """

    def __init__(
        self,
        column_kwargs: dict | None = None,
        learning_module_id: str = "CorticalColumnTorchLM_0",
        vote_evidence_threshold: float = 0.5,
        vote_top_k: int | None = None,
        vote_probability_temperature: float = 1.0,
        output_evidence_threshold: float = 1.0,
        surprise_gated_output: bool = False,
        output_surprise_threshold: float = 0.3,
        max_match_steps: int = 100,
        evidence_match_threshold: float = 3.0,
        evidence_separation_ratio: float = 1.5,
        goal_state_min_steps: int = 0,
        goal_state_min_separation_ratio: float | None = None,
        hopfield_voting: bool = False,
        surprise_vote_threshold: float = 0.3,
        temporal_memory_config: TemporalMemory | dict | None = None,
        temporal_transition_config: SemiMarkovTransitionMemory | dict | None = None,
        self_supervised_temporal_config: (
            SelfSupervisedSemiMarkovMemory | dict | None
        ) = None,
        self_supervised_temporal_evidence_weight: float = 0.0,
        self_supervised_temporal_transition_weight: float = 0.0,
        self_supervised_temporal_prediction_weight: float = 0.0,
        temporal_confusion_threshold: float = 0.5,
        temporal_behavior_weight: float = 0.0,
        temporal_behavior_min_history: int = 4,
        temporal_behavior_min_overlap: float = 0.2,
        temporal_behavior_tempo_invariant: bool = False,
        temporal_behavior_competition: float = 0.0,
        temporal_trace_config: dict | None = None,
        action_predictive_config: dict | None = None,
        inferred_state_config: dict | None = None,
        expected_context_sender_ids: list[str] | None = None,
        context_identity_weight: float = 0.0,
        use_child_graph_context_labels: bool = False,
    ):
        self.learning_module_id = learning_module_id
        self._vote_evidence_threshold = vote_evidence_threshold
        self._vote_top_k = (
            None if vote_top_k is None else max(1, int(vote_top_k))
        )
        self._vote_probability_temperature = max(
            float(vote_probability_temperature),
            1e-6,
        )
        self._output_evidence_threshold = output_evidence_threshold
        self._surprise_gated_output = surprise_gated_output
        self._output_surprise_threshold = output_surprise_threshold
        self._max_match_steps = max_match_steps
        self._evidence_match_threshold = evidence_match_threshold
        self._evidence_separation_ratio = evidence_separation_ratio
        self._goal_state_min_steps = max(int(goal_state_min_steps), 0)
        self._goal_state_min_separation_ratio = (
            None
            if goal_state_min_separation_ratio is None
            else max(float(goal_state_min_separation_ratio), 0.0)
        )
        self._hopfield_voting = hopfield_voting
        self._surprise_vote_threshold = surprise_vote_threshold
        self._temporal_confusion_threshold = temporal_confusion_threshold
        self._self_supervised_temporal_evidence_weight = float(
            self_supervised_temporal_evidence_weight
        )
        self._self_supervised_temporal_transition_weight = float(
            self_supervised_temporal_transition_weight
        )
        self._self_supervised_temporal_prediction_weight = float(
            self_supervised_temporal_prediction_weight
        )
        self._temporal_behavior_weight = temporal_behavior_weight
        self._temporal_behavior_min_history = temporal_behavior_min_history
        self._temporal_behavior_min_overlap = temporal_behavior_min_overlap
        self._temporal_behavior_tempo_invariant = temporal_behavior_tempo_invariant
        self._temporal_behavior_competition = temporal_behavior_competition
        self._expected_context_sender_ids = (
            {str(sender_id) for sender_id in expected_context_sender_ids}
            if expected_context_sender_ids is not None
            else None
        )
        self._context_identity_weight = max(float(context_identity_weight), 0.0)
        self._use_child_graph_context_labels = bool(
            use_child_graph_context_labels
        )
        self._use_inferred_state = inferred_state_config is not None
        inferred_state_config = dict(inferred_state_config or {})
        self._inferred_location_correction_weight = float(
            inferred_state_config.get("location_correction_weight", 0.35)
        )
        self._inferred_flow_update_rate = float(
            inferred_state_config.get("flow_update_rate", 0.35)
        )
        self._use_tracker_location_prior = bool(
            inferred_state_config.get("use_tracker_location_prior", True)
        )
        self._temporal_trace_enabled = temporal_trace_config is not None
        self._temporal_trace_decay_rates = self._build_temporal_trace_decay_rates(
            temporal_trace_config
        )
        self._temporal_trace_weight = 0.0
        self._temporal_trace_input_gain = 0.0
        self._temporal_trace_boundary_surprise_weight = 0.0
        self._temporal_trace_boundary_discontinuity_weight = 0.0
        self._temporal_trace_boundary_threshold = 0.0
        self._action_predictive_enabled = action_predictive_config is not None
        self._action_trace_gain = 0.0
        self._action_boundary_weight = 0.0

        # Inner column
        column_kwargs = dict(column_kwargs or {})
        if self._temporal_trace_enabled:
            column_kwargs.setdefault("use_apical", True)
        if self._action_predictive_enabled:
            action_config = dict(action_predictive_config or {})
            action_predictor_kwargs = dict(
                column_kwargs.get("action_predictor_kwargs", {})
            )
            action_predictor_kwargs.setdefault(
                "motor_dim",
                int(action_config.get("action_dim", 8)),
            )
            action_predictor_kwargs.setdefault(
                "hidden_dim",
                int(action_config.get("hidden_dim", 128)),
            )
            action_predictor_kwargs.setdefault(
                "learning_rate",
                float(action_config.get("learning_rate", 0.001)),
            )
            action_predictor_kwargs.setdefault(
                "n_settle_iters",
                int(action_config.get("n_settle_iters", 3)),
            )
            column_kwargs.setdefault("use_action_conditioned_prediction", True)
            column_kwargs.setdefault(
                "action_prediction_query_weight",
                float(action_config.get("query_bias_weight", 0.2)),
            )
            column_kwargs["action_predictor_kwargs"] = action_predictor_kwargs
            self._action_trace_gain = float(
                action_config.get("trace_gain", 0.2)
            )
            self._action_boundary_weight = float(
                action_config.get("boundary_weight", 0.75)
            )
        self._column = CorticalColumnTorch(**column_kwargs)

        self._temporal_trace_decay: torch.Tensor | None = None
        self._temporal_trace_integration: torch.Tensor | None = None
        self._temporal_trace_bank: torch.Tensor | None = None
        self._temporal_trace_vector: torch.Tensor | None = None
        if self._temporal_trace_enabled:
            trace_config = dict(temporal_trace_config or {})
            self._temporal_trace_weight = float(
                trace_config.get("trace_weight", 0.35)
            )
            self._temporal_trace_input_gain = float(
                trace_config.get("input_gain", 1.0)
            )
            self._temporal_trace_boundary_surprise_weight = float(
                trace_config.get("boundary_surprise_weight", 1.0)
            )
            self._temporal_trace_boundary_discontinuity_weight = float(
                trace_config.get("boundary_discontinuity_weight", 1.0)
            )
            self._temporal_trace_boundary_threshold = float(
                trace_config.get("boundary_threshold", 1.0)
            )
            self._temporal_trace_decay = torch.as_tensor(
                self._temporal_trace_decay_rates,
                dtype=torch.float32,
                device=self._column.device,
            ).unsqueeze(1)
            self._temporal_trace_integration = (
                1.0 - self._temporal_trace_decay
            ).clamp_min(1e-3)
            self._temporal_trace_bank = torch.zeros(
                (self._temporal_trace_decay.shape[0], self._column.n_cells),
                dtype=torch.float32,
                device=self._column.device,
            )
            self._temporal_trace_vector = torch.zeros(
                self._column.n_cells,
                dtype=torch.float32,
                device=self._column.device,
            )

        self._temporal_memory: TemporalMemory | None = None
        if temporal_memory_config is not None:
            if isinstance(temporal_memory_config, TemporalMemory):
                self._temporal_memory = temporal_memory_config
            elif isinstance(temporal_memory_config, dict):
                self._temporal_memory = TemporalMemory(**temporal_memory_config)
            else:
                self._temporal_memory = TemporalMemory()

        self._transition_memory: SemiMarkovTransitionMemory | None = None
        if temporal_transition_config is not None:
            if isinstance(
                temporal_transition_config,
                SemiMarkovTransitionMemory,
            ):
                self._transition_memory = temporal_transition_config
            elif isinstance(temporal_transition_config, dict):
                self._transition_memory = SemiMarkovTransitionMemory(
                    **temporal_transition_config
                )
            else:
                self._transition_memory = SemiMarkovTransitionMemory()

        self._self_supervised_temporal_memory: (
            SelfSupervisedSemiMarkovMemory | None
        ) = None
        if self_supervised_temporal_config is not None:
            if isinstance(
                self_supervised_temporal_config,
                SelfSupervisedSemiMarkovMemory,
            ):
                self._self_supervised_temporal_memory = (
                    self_supervised_temporal_config
                )
            elif isinstance(self_supervised_temporal_config, dict):
                self._self_supervised_temporal_memory = (
                    SelfSupervisedSemiMarkovMemory(
                        **self_supervised_temporal_config
                    )
                )
            else:
                self._self_supervised_temporal_memory = (
                    SelfSupervisedSemiMarkovMemory()
                )

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
        self._last_observed_state: State | None = None
        self._last_input_state: State | None = None
        self._primary_target_object: str | None = None
        self._primary_target_state: int | None = None
        self.stepwise_target_state: int | None = None
        self._temporal_observation_history: list[State] = []
        self._episode_training_labels: set[str] = set()
        self._predicted_temporal_label: str | None = None
        self._temporal_prediction_status: str | None = None
        self._temporal_event_signal = False
        self._current_temporal_label: str | None = None
        self._previous_temporal_label: str | None = None
        self._temporal_behavior_adjustments: dict[str, float] = {}
        self._self_supervised_temporal_adjustments: dict[str, float] = {}
        self._self_supervised_state_object_counts: dict[str, Counter[str]] = (
            defaultdict(Counter)
        )
        self._self_supervised_transition_object_counts: dict[
            str, Counter[str]
        ] = defaultdict(Counter)
        self._external_context: torch.Tensor | None = None
        self._external_context_identity: torch.Tensor | None = None
        self._context_packets: dict[str, dict[str, Any]] = {}
        self._last_context_step_epoch: int = -1
        self._action_context = None
        self._temporal_boundary_pressure = 0.0
        self._temporal_trace_discontinuity = 0.0
        self._temporal_trace_norm = 0.0
        self._temporal_trace_lane_norms: list[float] = []
        self._inferred_location: np.ndarray | None = None
        self._inferred_pose_vectors: np.ndarray | None = None
        self._inferred_flow_vector: np.ndarray | None = None
        self._last_observed_location: np.ndarray | None = None
        self._last_observed_pose_vectors: np.ndarray | None = None
        self._reset_inferred_state()
        self._reset_temporal_trace_state()
        self._reset_evidence_debug_state()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._stepped = False
        self._step_count = 0
        self._last_result = {}
        self._last_observed_state = None
        self._last_input_state = None
        self.buffer = FeatureAtLocationBuffer()
        self.terminal_state = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.detected_rotation_r = None
        self.possible_matches = {}
        self.evidence = {}
        self.stepwise_target_state = None
        self._temporal_observation_history = []
        self._episode_training_labels = set()
        self._predicted_temporal_label = None
        self._temporal_prediction_status = None
        self._temporal_event_signal = False
        self._current_temporal_label = None
        self._previous_temporal_label = None
        self._temporal_behavior_adjustments = {}
        self._self_supervised_temporal_adjustments = {}
        self._external_context = None
        self._external_context_identity = None
        self._context_packets = {}
        self._last_context_step_epoch = -1
        self._action_context = None
        self._reset_inferred_state()
        self._reset_temporal_trace_state()
        self._reset_evidence_debug_state()
        if self._temporal_memory is not None:
            self._temporal_memory.reset_episode()
        if self._transition_memory is not None:
            self._transition_memory.reset_episode()
        if self._self_supervised_temporal_memory is not None:
            self._self_supervised_temporal_memory.reset_episode()

    def pre_episode(self, primary_target=None, **kwargs) -> None:
        self._stepped = False
        self._step_count = 0
        self._last_result = {}
        self._last_observed_state = None
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
        self.stepwise_target_state = None
        self._temporal_observation_history = []
        self._episode_training_labels = set()
        self._predicted_temporal_label = None
        self._temporal_prediction_status = None
        self._temporal_event_signal = False
        self._current_temporal_label = None
        self._previous_temporal_label = None
        self._temporal_behavior_adjustments = {}
        self._self_supervised_temporal_adjustments = {}
        self._external_context = None
        self._external_context_identity = None
        self._context_packets = {}
        self._last_context_step_epoch = -1
        self._action_context = None
        self._reset_inferred_state()
        self._reset_temporal_trace_state()
        self._reset_evidence_debug_state()
        if self._temporal_memory is not None:
            self._temporal_memory.reset_episode()
        if self._transition_memory is not None:
            self._transition_memory.reset_episode()
        if self._self_supervised_temporal_memory is not None:
            self._self_supervised_temporal_memory.reset_episode()

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
        if self._transition_memory is not None:
            self._transition_memory.finalize_episode(
                learn=self._mode is ExperimentMode.TRAIN
            )
        if self._self_supervised_temporal_memory is not None:
            self._self_supervised_temporal_memory.finalize_episode(
                learn=self._mode is ExperimentMode.TRAIN
            )

        self._column.post_episode()

        if self._temporal_memory is not None:
            label = self._resolve_behavior_prototype_label()
            if label is not None and self._temporal_observation_history:
                self._temporal_memory.store_behavior_prototype(
                    label, self._temporal_observation_history
                )
                self._temporal_memory.replay_episode(n_replays=2)

    def set_experiment_mode(self, mode: ExperimentMode) -> None:
        self._mode = mode
        self.mode = mode

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def matching_step(self, ctx, observations=None) -> None:
        self._reset_evidence_debug_state()
        state = self._extract_usable_state(observations)
        if state is None:
            self._record_skipped_step_debug("no_usable_state", observations)
            self._last_result = self._column._empty_result()
            return

        if self._mode is ExperimentMode.TRAIN:
            self._set_training_label_from_state(state)

        self._last_input_state = self._prepare_input_state(state)
        self._apply_inference_context()
        self._last_result = self._column.step(self._last_input_state)
        self._stepped = True
        self._step_count += 1

        winner_path = self._last_evidence_debug.setdefault("winner_path", {})
        winner_path["base"] = self._record_evidence_debug_stage(
            "base_evidence",
            self._column._evidence,
        )

        self._feed_temporal_memory(observations)
        self._update_temporal_trace_state()
        self._apply_temporal_behavior_evidence()
        winner_path["after_temporal_behavior"] = self._record_evidence_debug_stage(
            "after_temporal_behavior_evidence",
            self._column._evidence,
        )
        self._last_evidence_debug["temporal_behavior_adjustments"] = (
            self._snapshot_evidence(self._temporal_behavior_adjustments)
        )
        self._last_evidence_debug["temporal_behavior_scores"] = (
            self._snapshot_evidence(self._last_temporal_behavior_scores)
        )
        self._update_temporal_state()
        self._apply_self_supervised_temporal_evidence()
        winner_path["final"] = self._record_evidence_debug_stage(
            "final_evidence",
            self._column._evidence,
        )
        self._last_evidence_debug["self_supervised_adjustments"] = (
            self._snapshot_evidence(self._self_supervised_temporal_adjustments)
        )
        self._last_evidence_debug["self_supervised_support"] = {
            key: self._snapshot_evidence(value)
            for key, value in self._last_self_supervised_temporal_support.items()
        }
        self._last_evidence_debug["temporal_labels"] = {
            "previous": self._previous_temporal_label,
            "current": self._current_temporal_label,
            "predicted": self._predicted_temporal_label,
        }
        self._last_evidence_debug["query_bias_norms"] = self._get_query_bias_norms()
        self._last_evidence_debug["prediction_mismatch"] = float(
            getattr(self._column, "prediction_mismatch", 0.0)
        )
        self._last_evidence_debug["action_prediction_error"] = float(
            getattr(self._column, "action_prediction_error", 0.0)
        )

        self.evidence = dict(self._column._evidence)
        self._update_possible_matches()
        self._auto_update_terminal_condition()
        self._buffer_observation(self._last_input_state)

    def exploratory_step(self, ctx, observations=None) -> None:
        self._reset_evidence_debug_state()
        state = self._extract_usable_state(observations)
        if state is None:
            self._record_skipped_step_debug("no_usable_state", observations)
            self._last_result = self._column._empty_result()
            return

        if self._mode is ExperimentMode.TRAIN:
            self._set_training_label_from_state(state)

        self._last_input_state = self._prepare_input_state(state)
        self._apply_inference_context()
        self._last_result = self._column.step(self._last_input_state)
        self._stepped = True
        self._step_count += 1

        self._feed_temporal_memory(observations)
        self._update_temporal_trace_state()

        self.evidence = dict(self._column._evidence)
        self._update_temporal_state()
        self._buffer_observation(self._last_input_state)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def _build_ranked_vote_hypotheses(
        self,
        evidence: dict[str, float],
    ) -> list[dict[str, float | int | str]]:
        if not evidence:
            return []

        ranked_items = sorted(
            ((str(obj_id), float(value)) for obj_id, value in evidence.items()),
            key=lambda item: (-item[1], item[0]),
        )
        values = np.asarray([item[1] for item in ranked_items], dtype=np.float64)
        if values.size == 0:
            return []

        shifted = (values - float(values.max())) / self._vote_probability_temperature
        probs = np.exp(shifted)
        probs = probs / max(float(probs.sum()), 1e-8)

        max_ev = float(values.max())
        min_ev = float(values.min())
        ev_range = max(max_ev - min_ev, 1e-8)

        hypotheses = []
        for rank, ((object_id, value), probability) in enumerate(
            zip(ranked_items, probs),
            start=1,
        ):
            hypotheses.append(
                {
                    "object_id": object_id,
                    "evidence": float(value),
                    "probability": float(probability),
                    "scaled_evidence": float((value - min_ev) / ev_range),
                    "rank": rank,
                }
            )

        return hypotheses

    def send_out_vote(self) -> dict | None:
        if not self._stepped:
            return None

        emission_state = self._get_state_for_emission()
        if emission_state is None:
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

        ranked_hypotheses = self._build_ranked_vote_hypotheses(evidence)
        if not ranked_hypotheses:
            return None

        if self._vote_top_k is not None:
            selected_hypotheses = ranked_hypotheses[: self._vote_top_k]
        else:
            selected_hypotheses = [
                hypothesis
                for hypothesis in ranked_hypotheses
                if float(hypothesis["scaled_evidence"])
                >= self._vote_evidence_threshold
            ]
            if not selected_hypotheses:
                selected_hypotheses = ranked_hypotheses[:1]

        possible_states = {}
        for hypothesis in selected_hypotheses:
            obj_id = str(hypothesis["object_id"])
            scaled = float(hypothesis["scaled_evidence"])

            location = np.asarray(emission_state.location, dtype=np.float64)
            pose = np.asarray(
                emission_state.morphological_features.get(
                    "pose_vectors", np.eye(3)
                ),
                dtype=np.float64,
            ).reshape(3, 3)

            vote_state = State(
                location=location,
                morphological_features={
                    "pose_vectors": self._orthonormalize_pose_vectors(pose),
                    "pose_fully_defined": True,
                },
                non_morphological_features={
                    "vote_probability": float(hypothesis["probability"]),
                    "vote_evidence": float(hypothesis["evidence"]),
                    "vote_rank": int(hypothesis["rank"]),
                },
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
        transport_location, transport_pose = self._get_vote_transport_pose()
        sensed_pose = np.vstack([
            transport_location.reshape(1, 3),
            transport_pose,
        ])

        vote = {
            "possible_states": possible_states,
            "sensed_pose_rel_body": sensed_pose,
            "ranked_hypotheses": [
                {
                    **hypothesis,
                    "sender_id": self.learning_module_id,
                }
                for hypothesis in selected_hypotheses
            ],
            "sender_id": self.learning_module_id,
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

        emission_state = self._get_state_for_emission()
        if emission_state is None:
            return None

        mlh = self._column.get_current_mlh()
        surprise = float(self._column.surprise)

        location = np.asarray(emission_state.location, dtype=np.float64)
        pose = self._orthonormalize_pose_vectors(
            emission_state.morphological_features.get("pose_vectors", np.eye(3))
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

        ev = float(mlh.get("evidence", 0.0))
        confident = bool(ev > self._output_evidence_threshold)

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
            confidence=float(
                min(ev / max(self._output_evidence_threshold, 1e-6), 1.0)
            ),
            use_state=confident,
            sender_id=self.learning_module_id,
            sender_type="LM",
            inferred_state=inferred_state,
        )

    def propose_goal_states(self) -> list:
        """Propose a hypothesis-testing goal state using the LFM predicted locations.

        When the LFM (Phase 11) is active, queries the location-feature memory
        with the current feature encoding to obtain per-object predicted locations
        (attention-weighted average of training locations matching current features).
        The most-likely-hypothesis object's predicted location is proposed as the
        next sensor target, directing the motor system to visit the region where
        the training data for the best hypothesis was collected.

        Falls back to the current sensor location if:
          - the LFM is not active or has no patterns stored,
          - no step has been taken yet this episode, or
          - no evidence has been accumulated or max evidence is below threshold.

        This is akin to EvidenceGoalStateGenerator.propose_goal_states() in
        EvidenceGraphLM, but uses the LFM's pred_locs instead of a stored graph.
        """
        from tbp.monty.frameworks.models.states import GoalState

        emission_state = self._get_state_for_emission()
        if not self._stepped or emission_state is None:
            return []

        evidence = dict(self._column._evidence)
        if not evidence:
            return []

        if self._step_count < self._goal_state_min_steps:
            return []

        values = list(evidence.values())
        max_ev = max(values)
        if max_ev < self._evidence_match_threshold:
            return []

        separation_ratio = self._goal_state_min_separation_ratio
        if separation_ratio is not None and len(values) >= 2:
            sorted_ev = sorted((float(value) for value in values), reverse=True)
            if sorted_ev[0] <= sorted_ev[1] * separation_ratio:
                return []

        # Confidence: how far above threshold we are, clipped to [0, 1]
        confidence = min(1.0, max_ev / max(self._evidence_match_threshold, 1e-6))

        # --- LFM path: predict most informative next location ---
        target_location = None
        lfm = self._column._lfm if self._column._use_lfm else None
        if lfm is not None and lfm.n_stored > 0:
            # Re-encode the current features for the LFM query
            _lb = self._column._encoder._location_bits
            import torch
            with torch.no_grad():
                input_vec = self._column._encoder.encode(emission_state)
                feat_enc = input_vec[_lb:]

            _, feat_ev, pred_locs = lfm.query_features(feat_enc)

            if feat_ev and pred_locs:
                # Pick the highest-evidence object that has a predicted location
                best_obj = max(
                    (obj for obj in feat_ev if obj in pred_locs),
                    key=lambda o: feat_ev[o],
                    default=None,
                )
                if best_obj is not None:
                    target_location = np.asarray(
                        pred_locs[best_obj], dtype=np.float64
                    )

        # Fall back to current location if LFM gave nothing useful
        if target_location is None:
            target_location = np.asarray(emission_state.location, dtype=np.float64)

        pose_vectors = np.asarray(
            emission_state.morphological_features.get(
                "pose_vectors", np.eye(3)
            ),
            dtype=np.float64,
        )

        return [
            GoalState(
                location=target_location,
                morphological_features={
                    "pose_vectors": pose_vectors,
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=float(confidence),
                use_state=True,
                sender_id=self.learning_module_id,
                sender_type="GSG",
                goal_tolerances={"location": 0.015},
            )
        ]

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def receive_context(self, **context_signal) -> None:
        active_cells = context_signal.get("active_cells")
        sender_id = context_signal.get("sender_id")

        if sender_id is None:
            if active_cells is not None:
                self._set_external_context(active_cells)
                self._apply_inference_context()

            # Parent LM pattern: if this LM was NOT stepped by a sensor module
            # this cycle, drive it from the received context.  This makes parent
            # columns in a heterarchy functional — they aggregate children's
            # activations through their own Hopfield + associative memory.
            if not self._stepped and active_cells is not None:
                self._finalize_context_only_step()
            return

        self._context_packets[str(sender_id)] = {
            "active_cells": active_cells,
            "confidence": float(context_signal.get("confidence", 0.0)),
            "epoch": int(context_signal.get("sender_step_count", 0)),
            "graph_id": context_signal.get("graph_id"),
            "identity_bias": self._build_context_identity_bias(
                str(sender_id),
                context_signal.get("graph_id"),
                context_signal.get("confidence", 0.0),
            ),
        }

        current_packets = self._get_current_context_packets()
        combined_context, combined_identity = self._compose_context_packets(
            current_packets
        )
        self._external_context_identity = combined_identity
        if combined_context is not None:
            self._external_context = combined_context
        self._apply_inference_context()

        if (
            combined_context is not None
            and self._context_ready_for_step(current_packets)
        ):
            self._finalize_context_only_step(
                step_epoch=self._get_context_epoch(current_packets)
            )

    def get_context_signal(self) -> dict | None:
        if not self._stepped:
            return None
        signal = self._column.get_context_signal()
        if signal is None:
            return None

        mlh = self._column.get_current_mlh()
        evidence = float(mlh.get("evidence", 0.0))
        signal.update(
            {
                "sender_id": self.learning_module_id,
                "graph_id": mlh.get("graph_id"),
                "confidence": float(
                    min(
                        evidence / max(self._output_evidence_threshold, 1e-6),
                        1.0,
                    )
                ),
                "sender_step_count": self._step_count,
            }
        )
        return signal

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
        if self._temporal_memory is not None:
            sd["temporal_memory"] = self._temporal_memory.state_dict()
        if self._transition_memory is not None:
            sd["transition_memory"] = self._transition_memory.state_dict()
        if self._self_supervised_temporal_memory is not None:
            sd["self_supervised_temporal_memory"] = (
                self._self_supervised_temporal_memory.state_dict()
            )
            sd["self_supervised_state_object_counts"] = {
                label: dict(counter)
                for label, counter in self._self_supervised_state_object_counts.items()
            }
            sd["self_supervised_transition_object_counts"] = {
                label: dict(counter)
                for label, counter in self._self_supervised_transition_object_counts.items()
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

        tm_state = state_dict.get("temporal_memory")
        if tm_state is not None and self._temporal_memory is not None:
            self._temporal_memory.load_state_dict(tm_state)

        transition_state = state_dict.get("transition_memory")
        if transition_state is not None and self._transition_memory is not None:
            self._transition_memory.load_state_dict(transition_state)

        self_supervised_state = state_dict.get("self_supervised_temporal_memory")
        if (
            self_supervised_state is not None
            and self._self_supervised_temporal_memory is not None
        ):
            self._self_supervised_temporal_memory.load_state_dict(
                self_supervised_state
            )

        self._self_supervised_state_object_counts = defaultdict(Counter)
        for label, counter_dict in state_dict.get(
            "self_supervised_state_object_counts", {}
        ).items():
            self._self_supervised_state_object_counts[str(label)] = Counter(
                counter_dict
            )

        self._self_supervised_transition_object_counts = defaultdict(Counter)
        for label, counter_dict in state_dict.get(
            "self_supervised_transition_object_counts", {}
        ).items():
            self._self_supervised_transition_object_counts[str(label)] = Counter(
                counter_dict
            )

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
            # Set detected_rotation_quat for logging compatibility.
            rotation = mlh.get("rotation", Rotation.identity())
            if hasattr(rotation, "as_quat"):
                self.buffer.stats["detected_rotation_quat"] = rotation.as_quat()
            else:
                self.buffer.stats["detected_rotation_quat"] = np.array(
                    [0.0, 0.0, 0.0, 1.0]
                )
        elif terminal_state == "no_match":
            self.detected_object = None
        else:
            self.detected_object = terminal_state
        self.buffer.stats["individual_ts_reached_at_step"] = self._step_count
        self.buffer.stats["individual_ts_object"] = self.detected_object
        self.buffer.stats["individual_ts_rot"] = self.buffer.stats["detected_rotation_quat"]

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
        # Keep buffer.on_object in sync so get_last_obs_processed() works.
        # (buffer.append() fails silently because State lacks .displacement,
        # leaving len(buffer)==0 and causing check_if_any_lms_updated() to
        # always return False.)
        self.buffer.on_object.append(lm_processed)

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

    def get_known_temporal_states(self) -> list[str]:
        if self._self_supervised_temporal_memory is not None:
            return self._self_supervised_temporal_memory.get_known_states()
        return []

    def set_action_context(self, action_context) -> None:
        if action_context is None:
            self._action_context = None
            return

        if isinstance(action_context, torch.Tensor):
            self._action_context = action_context.detach().float().clone()
            return

        self._action_context = np.asarray(action_context, dtype=np.float32)

    def get_current_mlh(self) -> dict:
        mlh = self._column.get_current_mlh()
        graph_id = mlh.get("graph_id", "no_observations_yet")
        evidence_val = mlh.get("evidence", 0.0)

        emission_state = self._get_state_for_emission()
        location = (
            np.asarray(emission_state.location, dtype=np.float64)
            if emission_state is not None
            else np.zeros(3)
        )

        return {
            "graph_id": graph_id if graph_id else "no_observations_yet",
            "location": location,
            "rotation": Rotation.identity(),
            "evidence": evidence_val,
            "scale": 1.0,
        }

    def get_temporal_surprise(self) -> float:
        status = self._temporal_prediction_status
        surprise_values = []
        if self._temporal_memory is not None:
            surprise_values.append(
                self._temporal_memory.get_mean_surprise(last_n=5)
            )
        if self._self_supervised_temporal_memory is not None:
            surprise_values.append(
                self._self_supervised_temporal_memory.get_mean_surprise(last_n=5)
            )
        if self._temporal_trace_enabled:
            surprise_values.append(
                max(
                    float(getattr(self._column, "surprise", 1.0)),
                    float(self._temporal_boundary_pressure),
                )
            )

        if not surprise_values:
            if status == "confident":
                return 0.0
            if status == "confused":
                return 1.0
            return 1.0

        surprise = max(surprise_values)
        if status == "confident":
            return min(surprise, 0.25)
        if status == "confused":
            return max(surprise, 0.75)
        return surprise

    def get_temporal_prediction_status(self) -> str | None:
        return self._temporal_prediction_status

    def get_temporal_context(self) -> dict | None:
        context = {}
        if self._temporal_memory is not None:
            tm_context = self._temporal_memory.get_temporal_context()
            if tm_context is not None:
                context.update(tm_context)

        if self._self_supervised_temporal_memory is not None:
            hsmm_context = (
                self._self_supervised_temporal_memory.get_temporal_context()
            )
            if hsmm_context is not None:
                context.update(hsmm_context)
        elif self._transition_memory is not None:
            context.update(
                {
                    "predicted_label": self._predicted_temporal_label,
                    "current_dwell": self._transition_memory.get_current_dwell(),
                    "event_detected": self._temporal_event_signal,
                }
            )

        if self._temporal_trace_enabled and self._step_count > 0:
            context.update(
                {
                    "trace_norm": self._temporal_trace_norm,
                    "boundary_pressure": self._temporal_boundary_pressure,
                    "trace_discontinuity": self._temporal_trace_discontinuity,
                    "trace_scales": int(len(self._temporal_trace_decay_rates)),
                }
            )

        return context or None

    def get_event_signal(self) -> bool:
        return self._temporal_event_signal

    def get_evidence_debug(self) -> dict | None:
        if not self._last_evidence_debug:
            return None

        debug = dict(self._last_evidence_debug)
        for key in (
            "base_evidence",
            "after_temporal_behavior_evidence",
            "final_evidence",
            "temporal_behavior_adjustments",
            "temporal_behavior_scores",
            "self_supervised_adjustments",
        ):
            if key in debug:
                debug[key] = dict(debug[key])

        if "winner_path" in debug:
            debug["winner_path"] = dict(debug["winner_path"])
        if "observation_summary" in debug:
            debug["observation_summary"] = dict(debug["observation_summary"])
        if "temporal_labels" in debug:
            debug["temporal_labels"] = dict(debug["temporal_labels"])
        if "query_bias_norms" in debug:
            debug["query_bias_norms"] = dict(debug["query_bias_norms"])
        if "self_supervised_support" in debug:
            debug["self_supervised_support"] = {
                key: dict(value)
                for key, value in debug["self_supervised_support"].items()
            }

        return debug

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_evidence_debug_state(self) -> None:
        self._last_temporal_behavior_scores: dict[str, float] = {}
        self._last_self_supervised_temporal_support: dict[
            str, dict[str, float]
        ] = {
            "state": {},
            "transition": {},
            "prediction": {},
        }
        self._last_evidence_debug: dict[str, Any] = {}

    @staticmethod
    def _snapshot_evidence(evidence: dict[str, float] | None) -> dict[str, float]:
        if not evidence:
            return {}

        return {
            str(label): float(value)
            for label, value in sorted(
                evidence.items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
        }

    @staticmethod
    def _top_evidence_label(evidence: dict[str, float] | None) -> str | None:
        if not evidence:
            return None

        return max(
            ((str(label), float(value)) for label, value in evidence.items()),
            key=lambda item: (item[1], item[0]),
        )[0]

    def _record_evidence_debug_stage(
        self,
        key: str,
        evidence: dict[str, float] | None,
    ) -> str | None:
        snapshot = self._snapshot_evidence(evidence)
        self._last_evidence_debug[key] = snapshot
        return self._top_evidence_label(snapshot)

    def _record_skipped_step_debug(self, reason: str, observations) -> None:
        self._last_evidence_debug = {
            "step_skipped": True,
            "skip_reason": str(reason),
            "observation_summary": self._summarize_observations(observations),
        }

    @classmethod
    def _summarize_observations(cls, observations) -> dict[str, Any]:
        if observations is None:
            return {
                "state_like_count": 0,
                "use_state_true_count": 0,
                "non_lm_use_state_true_count": 0,
                "sender_type_counts": {},
            }

        states = []
        if isinstance(observations, list):
            for item in observations:
                if cls._is_state_like(item):
                    states.append(item)
                elif isinstance(item, list):
                    states.extend(s for s in item if cls._is_state_like(s))
                elif isinstance(item, dict):
                    states.extend(v for v in item.values() if cls._is_state_like(v))
        elif cls._is_state_like(observations):
            states = [observations]

        sender_type_counts = Counter(
            str(getattr(state, "sender_type", None))
            for state in states
        )
        use_state_true_count = sum(
            int(bool(getattr(state, "use_state", False)))
            for state in states
        )
        non_lm_use_state_true_count = sum(
            int(
                bool(getattr(state, "use_state", False))
                and getattr(state, "sender_type", "SM") != "LM"
            )
            for state in states
        )
        return {
            "state_like_count": len(states),
            "use_state_true_count": use_state_true_count,
            "non_lm_use_state_true_count": non_lm_use_state_true_count,
            "sender_type_counts": {
                label: int(count)
                for label, count in sorted(sender_type_counts.items())
            },
        }

    def _get_query_bias_norms(self) -> dict[str, float]:
        hopfield_query_bias = getattr(self._column, "_hopfield_query_bias", None)
        action_query_bias = getattr(
            self._column,
            "action_conditioned_query_bias",
            None,
        )
        return {
            "trace_hopfield_query_bias_norm": (
                float(hopfield_query_bias.detach().abs().sum().item())
                if hopfield_query_bias is not None
                else 0.0
            ),
            "action_conditioned_query_bias_norm": (
                float(action_query_bias.detach().abs().sum().item())
                if action_query_bias is not None
                else 0.0
            ),
        }

    @staticmethod
    def _parse_composite_key(key: str) -> tuple:
        if key and ":" in str(key):
            parts = str(key).rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return key, None
        return key, None

    @staticmethod
    def _build_temporal_trace_decay_rates(
        trace_config: dict | None,
    ) -> np.ndarray:
        if trace_config is None:
            return np.zeros(0, dtype=np.float32)

        decay_rates = trace_config.get("decay_rates")
        if decay_rates is None:
            n_scales = max(1, int(trace_config.get("n_scales", 4)))
            min_decay = float(trace_config.get("min_decay", 0.2))
            max_decay = float(trace_config.get("max_decay", 0.95))
            min_decay = min(max(min_decay, 1e-3), 0.999)
            max_decay = min(max(max_decay, min_decay), 0.999)
            if n_scales == 1:
                decay_rates = [max_decay]
            else:
                decay_rates = np.geomspace(min_decay, max_decay, num=n_scales)

        return np.clip(
            np.asarray(decay_rates, dtype=np.float32).reshape(-1),
            1e-3,
            0.999,
        )

    def _set_training_label_from_state(self, state: State) -> str | None:
        object_name = self._primary_target_object
        if object_name is None:
            label = self._derive_child_graph_context_label(state)
            if label is None:
                return None

            self._column._current_object = label
            self._episode_training_labels.add(label)
            return label

        state_id = getattr(state, "inferred_state", None)
        if state_id is None:
            state_id = self.stepwise_target_state
        if state_id is None:
            state_id = self._primary_target_state

        if state_id is None:
            label = object_name
        else:
            label = f"{object_name}:{state_id}"

        self._column._current_object = label
        self._episode_training_labels.add(label)
        self.graph_id_to_target.setdefault(label, set()).add(object_name)
        self.target_to_graph_id.setdefault(object_name, set()).add(label)
        return label

    def _derive_child_graph_context_label(self, state: State) -> str | None:
        if not self._use_child_graph_context_labels:
            return None
        if getattr(state, "sender_type", None) != "LM":
            return None

        non_morphological = getattr(state, "non_morphological_features", None) or {}
        child_sender_ids = list(non_morphological.get("child_sender_ids") or [])
        child_graph_ids = list(non_morphological.get("child_graph_ids") or [])

        label_parts = []
        if child_sender_ids and child_graph_ids:
            for sender_id, graph_id in zip(child_sender_ids, child_graph_ids):
                normalized_graph_id = self._normalize_context_graph_id(graph_id)
                if normalized_graph_id is None:
                    continue
                label_parts.append(f"{sender_id}={normalized_graph_id}")
        else:
            sender_id = getattr(state, "sender_id", None)
            graph_id = self._normalize_context_graph_id(
                non_morphological.get("graph_id")
            )
            if sender_id not in (None, "") and graph_id is not None:
                label_parts.append(f"{sender_id}={graph_id}")

        if not label_parts:
            return None

        fingerprint = hashlib.sha1(
            "|".join(sorted(label_parts)).encode("utf-8")
        ).hexdigest()[:12]
        return f"auto_ctx_{fingerprint}"

    def _resolve_behavior_prototype_label(self) -> str | None:
        if self._mode is not ExperimentMode.TRAIN:
            return None

        if len(self._episode_training_labels) == 1:
            return next(iter(self._episode_training_labels))

        if self._episode_training_labels:
            return None

        auto_label = getattr(self._column, "_current_object", None)
        if not auto_label:
            return None

        return str(auto_label)

    def _feed_temporal_memory(self, observations) -> None:
        if self._temporal_memory is None or observations is None:
            return

        for obs in observations:
            if getattr(obs, "sender_type", None) != "SM":
                continue
            if not getattr(obs, "use_state", True):
                continue

            self._temporal_observation_history.append(obs)
            learn = self._mode is ExperimentMode.TRAIN
            self._temporal_memory.step(obs, learn=learn)
            break

    def _apply_temporal_behavior_evidence(self) -> None:
        self._restore_temporal_behavior_adjustments()
        self._last_temporal_behavior_scores = {}

        if self._temporal_memory is None:
            return
        if self._temporal_behavior_weight <= 0:
            return
        if len(self._temporal_observation_history) < self._temporal_behavior_min_history:
            return

        scores = self._temporal_memory.score_behaviors(
            self._temporal_observation_history,
            tempo_invariant=self._temporal_behavior_tempo_invariant,
        )
        self._last_temporal_behavior_scores = self._snapshot_evidence(scores)
        if not scores:
            return
        behavior_name, score = max(scores.items(), key=lambda item: item[1])
        if score < self._temporal_behavior_min_overlap:
            return
        if behavior_name not in self._column._evidence:
            return

        adjustments = {
            behavior_name: score * self._temporal_behavior_weight,
        }

        if self._temporal_behavior_competition > 0:
            for other_name, other_score in scores.items():
                if other_name == behavior_name:
                    continue
                if other_name not in self._column._evidence:
                    continue
                gap = max(0.0, score - other_score)
                adjustments[other_name] = adjustments.get(other_name, 0.0) - (
                    gap * self._temporal_behavior_competition
                )

        self._apply_temporal_behavior_adjustments(adjustments)

    def _restore_temporal_behavior_adjustments(self) -> None:
        if not self._temporal_behavior_adjustments:
            return

        for label, delta in self._temporal_behavior_adjustments.items():
            if label not in self._column._evidence:
                continue
            self._column._evidence[label] = max(
                0.0,
                self._column._evidence[label] - delta,
            )

        self._temporal_behavior_adjustments = {}

    def _apply_temporal_behavior_adjustments(
        self,
        adjustments: dict[str, float],
    ) -> None:
        applied = {}

        for label, delta in adjustments.items():
            if label not in self._column._evidence:
                continue

            current = self._column._evidence[label]
            actual_delta = delta
            if current + actual_delta < 0:
                actual_delta = -current

            self._column._evidence[label] = current + actual_delta
            applied[label] = actual_delta

        self._temporal_behavior_adjustments = applied

    @staticmethod
    def _format_temporal_transition_key(
        previous_label: str | None,
        current_label: str | None,
    ) -> str | None:
        if previous_label is None or current_label is None:
            return None
        return f"{previous_label}\t{current_label}"

    def _get_temporal_object_association_label(self) -> str | None:
        if self._mode is not ExperimentMode.TRAIN:
            return None

        current_object = getattr(self._column, "_current_object", None)
        if current_object:
            return str(current_object)

        if self._primary_target_object is not None:
            return str(self._primary_target_object)

        return None

    def _record_self_supervised_temporal_associations(
        self,
        previous_label: str | None,
        current_label: str | None,
    ) -> None:
        object_label = self._get_temporal_object_association_label()
        if object_label is None or current_label is None:
            return

        self._self_supervised_state_object_counts[current_label][object_label] += 1

        transition_key = self._format_temporal_transition_key(
            previous_label,
            current_label,
        )
        if transition_key is not None:
            self._self_supervised_transition_object_counts[transition_key][
                object_label
            ] += 1

    def _get_self_supervised_temporal_support(
        self,
        counts: Counter[str] | None,
    ) -> dict[str, float]:
        if not counts:
            return {}

        total = sum(counts.values())
        if total <= 0:
            return {}

        return {
            label: float(count) / total
            for label, count in counts.items()
        }

    def _apply_self_supervised_temporal_evidence(self) -> None:
        self._restore_self_supervised_temporal_adjustments()
        self._last_self_supervised_temporal_support = {
            "state": {},
            "transition": {},
            "prediction": {},
        }

        if self._self_supervised_temporal_memory is None:
            return
        if (
            self._self_supervised_temporal_evidence_weight <= 0
            and self._self_supervised_temporal_transition_weight <= 0
            and self._self_supervised_temporal_prediction_weight <= 0
        ):
            return

        adjustments: dict[str, float] = {}
        current_label = self._current_temporal_label
        previous_label = self._previous_temporal_label
        predicted_label = self._predicted_temporal_label
        state_support: dict[str, float] = {}
        transition_support: dict[str, float] = {}
        prediction_support: dict[str, float] = {}

        if current_label is not None and self._self_supervised_temporal_evidence_weight > 0:
            state_support = self._get_self_supervised_temporal_support(
                self._self_supervised_state_object_counts.get(current_label)
            )
            for label, value in state_support.items():
                adjustments[label] = adjustments.get(label, 0.0) + (
                    value * self._self_supervised_temporal_evidence_weight
                )

        if (
            previous_label is not None
            and current_label is not None
            and self._self_supervised_temporal_transition_weight > 0
        ):
            transition_support = self._get_self_supervised_temporal_support(
                self._self_supervised_transition_object_counts.get(
                    self._format_temporal_transition_key(
                        previous_label,
                        current_label,
                    )
                )
            )
            for label, value in transition_support.items():
                adjustments[label] = adjustments.get(label, 0.0) + (
                    value * self._self_supervised_temporal_transition_weight
                )

        if (
            current_label is not None
            and predicted_label is not None
            and self._self_supervised_temporal_prediction_weight > 0
        ):
            prediction_support = self._get_self_supervised_temporal_support(
                self._self_supervised_transition_object_counts.get(
                    self._format_temporal_transition_key(
                        current_label,
                        predicted_label,
                    )
                )
            )
            for label, value in prediction_support.items():
                adjustments[label] = adjustments.get(label, 0.0) + (
                    value * self._self_supervised_temporal_prediction_weight
                )

        self._last_self_supervised_temporal_support = {
            "state": self._snapshot_evidence(state_support),
            "transition": self._snapshot_evidence(transition_support),
            "prediction": self._snapshot_evidence(prediction_support),
        }

        self._apply_self_supervised_temporal_adjustments(adjustments)

    def _restore_self_supervised_temporal_adjustments(self) -> None:
        if not self._self_supervised_temporal_adjustments:
            return

        for label, delta in self._self_supervised_temporal_adjustments.items():
            if label not in self._column._evidence:
                continue
            self._column._evidence[label] = max(
                0.0,
                self._column._evidence[label] - delta,
            )

        self._self_supervised_temporal_adjustments = {}

    def _apply_self_supervised_temporal_adjustments(
        self,
        adjustments: dict[str, float],
    ) -> None:
        applied = {}

        for label, delta in adjustments.items():
            if label not in self._column._evidence:
                continue

            current = self._column._evidence[label]
            actual_delta = delta
            if current + actual_delta < 0:
                actual_delta = -current

            self._column._evidence[label] = current + actual_delta
            applied[label] = actual_delta

        self._self_supervised_temporal_adjustments = applied

    def _reset_temporal_trace_state(self) -> None:
        self._temporal_boundary_pressure = 0.0
        self._temporal_trace_discontinuity = 0.0
        self._temporal_trace_norm = 0.0
        self._temporal_trace_lane_norms = []
        if self._temporal_trace_bank is not None:
            self._temporal_trace_bank.zero_()
        if self._temporal_trace_vector is not None:
            self._temporal_trace_vector.zero_()

    def _coerce_context_tensor(self, active_cells) -> torch.Tensor | None:
        if active_cells is None:
            return None

        if isinstance(active_cells, np.ndarray):
            ctx = torch.from_numpy(active_cells.astype(np.float32)).to(
                self._column.device
            )
        elif isinstance(active_cells, torch.Tensor):
            ctx = active_cells.to(self._column.device).float()
        else:
            return None

        if ctx.ndim != 1:
            ctx = ctx.reshape(-1)

        if ctx.shape[0] != self._column.n_cells:
            ctx = torch.nn.functional.interpolate(
                ctx.unsqueeze(0).unsqueeze(0),
                size=self._column.n_cells,
                mode="linear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        return ctx

    @staticmethod
    def _normalize_context_graph_id(graph_id) -> str | None:
        if graph_id is None:
            return None

        normalized = str(graph_id)
        if normalized in {"", "unknown", "no_observations_yet", "None"}:
            return None
        return normalized

    def _build_context_identity_bias(
        self,
        sender_id: str | None,
        graph_id,
        confidence: float | None,
    ) -> torch.Tensor | None:
        if self._context_identity_weight <= 0.0:
            return None

        normalized_sender = None if sender_id is None else str(sender_id)
        normalized_graph_id = self._normalize_context_graph_id(graph_id)
        if not normalized_sender or normalized_graph_id is None:
            return None

        label = f"{normalized_sender}:{normalized_graph_id}"
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        rng = np.random.RandomState(seed % (2**32 - 1))

        n_active_mc = max(1, int(getattr(self._column, "_n_active_mc", 1)))
        selected_mc = rng.choice(
            self._column.n_minicolumns,
            size=min(n_active_mc, self._column.n_minicolumns),
            replace=False,
        )
        selected_cells = rng.randint(
            0,
            self._column.n_cells_per_minicolumn,
            size=selected_mc.size,
        )
        active_idx = (
            selected_mc * self._column.n_cells_per_minicolumn + selected_cells
        )

        bias = torch.zeros(
            self._column.n_cells,
            dtype=torch.float32,
            device=self._column.device,
        )
        weight = max(
            min(float(confidence if confidence is not None else 1.0), 1.0),
            1e-3,
        )
        bias[torch.as_tensor(active_idx, device=self._column.device)] = (
            self._context_identity_weight * weight
        )
        return bias

    def _get_current_context_packets(self) -> dict[str, dict[str, Any]]:
        return dict(self._context_packets)

    @staticmethod
    def _get_context_epoch(
        packets: dict[str, dict[str, Any]],
    ) -> int | None:
        if not packets:
            return None

        return max(int(packet.get("epoch", 0)) for packet in packets.values())

    def _compose_context_packets(
        self,
        packets: dict[str, dict[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not packets:
            return None, None

        combined_context = None
        combined_identity = None
        total_weight = 0.0

        for packet in packets.values():
            weight = max(float(packet.get("confidence", 0.0)), 1e-3)

            context = self._coerce_context_tensor(packet.get("active_cells"))
            if context is not None:
                if combined_context is None:
                    combined_context = torch.zeros_like(context)
                combined_context += context * weight
                total_weight += weight

            identity_bias = packet.get("identity_bias")
            if identity_bias is not None:
                identity_bias = self._coerce_context_tensor(identity_bias)
                if identity_bias is not None:
                    if combined_identity is None:
                        combined_identity = torch.zeros_like(identity_bias)
                    combined_identity += identity_bias

        if combined_context is not None and total_weight > 0.0:
            combined_context = combined_context / total_weight

        return combined_context, combined_identity

    def _build_context_fused_state(
        self,
        packets: dict[str, dict[str, Any]],
        combined_context: torch.Tensor | None,
    ) -> State | None:
        if not packets:
            return None

        sender_ids = [str(sender_id) for sender_id in packets.keys()]
        child_graph_ids = [packets[sender_id].get("graph_id") for sender_id in sender_ids]
        child_confidences = [
            float(packets[sender_id].get("confidence", 0.0))
            for sender_id in sender_ids
        ]

        non_morphological = {
            "child_sender_ids": sender_ids,
            "child_graph_ids": child_graph_ids,
            "child_confidences": child_confidences,
        }
        if combined_context is not None:
            non_morphological["active_cells"] = (
                combined_context.detach().cpu().numpy()
            )

        return State(
            location=np.zeros(3, dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3, dtype=np.float64),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features=non_morphological,
            confidence=float(max(child_confidences) if child_confidences else 0.0),
            use_state=True,
            sender_id="fused:" + "+".join(sender_ids),
            sender_type="LM",
        )

    def _context_ready_for_step(
        self,
        packets: dict[str, dict[str, Any]],
    ) -> bool:
        if not packets:
            return False
        if self._expected_context_sender_ids is None:
            expected_ready = True
        else:
            expected_ready = self._expected_context_sender_ids.issubset(packets.keys())

        if not expected_ready:
            return False

        context_epoch = self._get_context_epoch(packets)
        if context_epoch is None:
            return False

        return context_epoch > self._last_context_step_epoch

    def _finalize_context_only_step(self, step_epoch: int | None = None) -> None:
        self._reset_evidence_debug_state()

        current_packets = self._get_current_context_packets()
        combined_context, _ = self._compose_context_packets(current_packets)
        context_state = self._build_context_fused_state(
            current_packets,
            combined_context,
        )
        self._last_input_state = None
        self._last_observed_state = context_state

        if self._mode is ExperimentMode.TRAIN and context_state is not None:
            self._set_training_label_from_state(context_state)

        self._last_result = self._column.step_from_context()
        self._stepped = True
        self._step_count += 1
        if step_epoch is not None:
            self._last_context_step_epoch = int(step_epoch)

        winner_path = self._last_evidence_debug.setdefault("winner_path", {})
        winner_path["base"] = self._record_evidence_debug_stage(
            "base_evidence",
            self._column._evidence,
        )

        self._update_temporal_trace_state()
        self._apply_temporal_behavior_evidence()
        winner_path["after_temporal_behavior"] = self._record_evidence_debug_stage(
            "after_temporal_behavior_evidence",
            self._column._evidence,
        )
        self._last_evidence_debug["temporal_behavior_adjustments"] = (
            self._snapshot_evidence(self._temporal_behavior_adjustments)
        )
        self._last_evidence_debug["temporal_behavior_scores"] = (
            self._snapshot_evidence(self._last_temporal_behavior_scores)
        )

        self._update_temporal_state()
        self._apply_self_supervised_temporal_evidence()
        winner_path["final"] = self._record_evidence_debug_stage(
            "final_evidence",
            self._column._evidence,
        )
        self._last_evidence_debug["self_supervised_adjustments"] = (
            self._snapshot_evidence(self._self_supervised_temporal_adjustments)
        )
        self._last_evidence_debug["self_supervised_support"] = {
            key: self._snapshot_evidence(value)
            for key, value in self._last_self_supervised_temporal_support.items()
        }
        self._last_evidence_debug["temporal_labels"] = {
            "previous": self._previous_temporal_label,
            "current": self._current_temporal_label,
            "predicted": self._predicted_temporal_label,
        }
        self._last_evidence_debug["query_bias_norms"] = self._get_query_bias_norms()
        self._last_evidence_debug["prediction_mismatch"] = float(
            getattr(self._column, "prediction_mismatch", 0.0)
        )
        self._last_evidence_debug["action_prediction_error"] = float(
            getattr(self._column, "action_prediction_error", 0.0)
        )

        self.evidence = dict(self._column._evidence)
        self._update_possible_matches()
        self._auto_update_terminal_condition()

    def _set_external_context(self, active_cells) -> None:
        ctx = self._coerce_context_tensor(active_cells)
        if ctx is not None:
            self._external_context = ctx

    def _compose_temporal_trace_context(self) -> torch.Tensor | None:
        if not self._temporal_trace_enabled or self._temporal_trace_vector is None:
            return None

        max_abs = float(self._temporal_trace_vector.detach().abs().max().item())
        if max_abs <= 1e-8:
            return None

        return (self._temporal_trace_vector / max_abs) * self._temporal_trace_weight

    def _compose_hopfield_query_bias(self) -> torch.Tensor | None:
        temporal_bias = self._compose_temporal_trace_context()
        if temporal_bias is None and self._external_context_identity is None:
            return None

        combined = torch.zeros(
            self._column.n_cells,
            dtype=torch.float32,
            device=self._column.device,
        )
        if temporal_bias is not None:
            combined += temporal_bias
        if self._external_context_identity is not None:
            combined += self._external_context_identity

        max_abs = float(combined.detach().abs().max().item())
        if max_abs <= 1e-8:
            return None
        if max_abs > 1.0:
            combined = combined / max_abs
        return combined

    def _compose_inference_context(self) -> torch.Tensor | None:
        if self._external_context is None:
            return None

        combined = self._external_context.clone()

        max_abs = float(combined.detach().abs().max().item())
        if max_abs > 1.0:
            combined = combined / max_abs

        return combined

    def _apply_inference_context(self) -> None:
        self._column.receive_context(action_context=self._action_context)

        combined = self._compose_inference_context()
        if combined is not None:
            self._column.receive_context(active_cells=combined)

        self._column.receive_context(
            hopfield_query_bias=self._compose_hopfield_query_bias()
        )

    def _compute_temporal_boundary_pressure(self) -> tuple[float, float]:
        current_active = getattr(self._column, "_active", None)
        previous_active = getattr(self._column, "_prev_active", None)
        if current_active is None:
            return 0.0, 0.0

        surprise = float(getattr(self._column, "surprise", 1.0))
        prediction_mismatch = float(
            getattr(self._column, "prediction_mismatch", surprise)
        )
        settle_displacement = 0.0
        pre_settle_active = getattr(self._column, "_last_pre_settle", None)
        if pre_settle_active is not None:
            pre_norm = float(pre_settle_active.detach().abs().sum().item())
            curr_norm = float(current_active.detach().abs().sum().item())
            if pre_norm > 1e-8 and curr_norm > 1e-8:
                settle_diff = float(
                    (current_active - pre_settle_active)
                    .detach()
                    .abs()
                    .sum()
                    .item()
                )
                settle_displacement = settle_diff / max(curr_norm + pre_norm, 1e-8)
                settle_displacement = max(0.0, min(1.0, settle_displacement))

        correction_mismatch = max(0.0, prediction_mismatch - surprise)
        mismatch = max(
            surprise,
            0.5 * settle_displacement,
            0.5 * correction_mismatch,
        )
        if self._action_boundary_weight > 0:
            mismatch = max(
                mismatch,
                min(
                    1.0,
                    self._action_boundary_weight
                    * float(getattr(self._column, "action_prediction_error", 0.0)),
                ),
            )

        discontinuity = 0.0
        if previous_active is not None:
            prev_norm = float(previous_active.detach().abs().sum().item())
            curr_norm = float(current_active.detach().abs().sum().item())
            if prev_norm > 1e-8 and curr_norm > 1e-8:
                diff = float(
                    (current_active - previous_active).detach().abs().sum().item()
                )
                discontinuity = diff / max(curr_norm + prev_norm, 1e-8)
                discontinuity = max(0.0, min(1.0, discontinuity))

        pressure_logit = (
            self._temporal_trace_boundary_surprise_weight
            * mismatch
            + self._temporal_trace_boundary_discontinuity_weight * discontinuity
            - self._temporal_trace_boundary_threshold
        )
        pressure = 1.0 / (1.0 + float(np.exp(-pressure_logit)))
        return pressure, discontinuity

    def _update_temporal_trace_state(self) -> None:
        if not self._temporal_trace_enabled:
            return
        if self._temporal_trace_bank is None or self._temporal_trace_vector is None:
            return

        current_active = getattr(self._column, "_active", None)
        if current_active is None:
            return

        active = current_active.detach().float()
        if float(active.abs().sum().item()) <= 1e-8:
            return

        boundary_pressure, discontinuity = self._compute_temporal_boundary_pressure()
        retain = (1.0 - boundary_pressure) * self._temporal_trace_decay
        drive = (
            self._temporal_trace_input_gain
            * self._temporal_trace_integration
            * active.unsqueeze(0)
        )
        action_bias = getattr(self._column, "action_conditioned_query_bias", None)
        if (
            self._action_trace_gain > 0
            and action_bias is not None
            and float(action_bias.detach().abs().sum().item()) > 1e-8
        ):
            action_drive = action_bias.detach().float()
            max_abs = float(action_drive.abs().max().item())
            if max_abs > 1e-8:
                drive = drive + (
                    self._action_trace_gain
                    * self._temporal_trace_integration
                    * (action_drive / max_abs).unsqueeze(0)
                )

        self._temporal_trace_bank.mul_(retain)
        self._temporal_trace_bank.add_(drive)
        self._temporal_trace_vector.copy_(self._temporal_trace_bank.mean(dim=0))

        self._temporal_boundary_pressure = float(boundary_pressure)
        self._temporal_trace_discontinuity = float(discontinuity)
        self._temporal_trace_norm = float(
            self._temporal_trace_vector.detach().abs().sum().item()
        )
        self._temporal_trace_lane_norms = [
            float(value)
            for value in self._temporal_trace_bank.detach().abs().sum(dim=1).cpu().tolist()
        ]

    def _extract_temporal_embedding(self):
        if not self._stepped:
            return None

        active = getattr(self._column, "_active", None)
        if active is None:
            return None

        encoder = getattr(self._column, "_encoder", None)
        if encoder is not None and self._last_input_state is not None:
            try:
                encoded = encoder.encode(self._last_input_state)
                if hasattr(encoded, "detach"):
                    return encoded.detach()
                return encoded
            except (AttributeError, TypeError, ValueError):
                pass

        pre_settle = getattr(self._column, "_last_pre_settle", None)
        if (
            pre_settle is not None
            and hasattr(pre_settle, "detach")
            and float(pre_settle.detach().abs().sum()) > 1e-8
        ):
            return pre_settle.detach()

        if hasattr(active, "detach"):
            return active.detach()
        return active

    def _get_temporal_boundary_surprise(self) -> float:
        surprise_values = [float(getattr(self._column, "surprise", 1.0))]
        if self._temporal_memory is not None:
            surprise_values.append(
                self._temporal_memory.get_mean_surprise(last_n=1)
            )
        return max(surprise_values)

    def _update_temporal_state(self) -> None:
        status = None
        predicted_label = None
        event_detected = False
        current_label = None
        previous_label = self._current_temporal_label

        if self._self_supervised_temporal_memory is not None:
            info = self._self_supervised_temporal_memory.observe(
                self._extract_temporal_embedding(),
                learn=self._mode is ExperimentMode.TRAIN,
                surprise=self._get_temporal_boundary_surprise(),
            )
            current_label = info.get("current_label")
            status = info.get("prediction_status")
            predicted_label = info.get("predicted_label")
            event_detected = bool(info.get("event_detected", False))
            if self._mode is ExperimentMode.TRAIN:
                self._record_self_supervised_temporal_associations(
                    previous_label,
                    current_label,
                )
        elif self._transition_memory is not None:
            info = self._transition_memory.observe(
                self._extract_temporal_label(),
                learn=self._mode is ExperimentMode.TRAIN,
            )
            current_label = info.get("current_label")
            status = info.get("prediction_status")
            predicted_label = info.get("predicted_label")
            event_detected = bool(info.get("event_detected", False))

        if status is None:
            surprise_values = []
            if self._temporal_memory is not None:
                surprise_values.append(
                    self._temporal_memory.get_mean_surprise(last_n=5)
                )
            if self._self_supervised_temporal_memory is not None:
                surprise_values.append(
                    self._self_supervised_temporal_memory.get_mean_surprise(last_n=5)
                )
            surprise = max(surprise_values) if surprise_values else None
        else:
            surprise = None

        if surprise is not None:
            status = (
                "confident"
                if surprise <= self._temporal_confusion_threshold
                else "confused"
            )

        self._predicted_temporal_label = predicted_label
        self._temporal_prediction_status = status
        self._temporal_event_signal = event_detected
        self._previous_temporal_label = previous_label
        self._current_temporal_label = current_label

    def _extract_temporal_label(self) -> str | None:
        if not self._stepped:
            return None

        mlh = self._column.get_current_mlh()
        graph_id = mlh.get("graph_id")
        evidence = float(mlh.get("evidence", 0.0))
        if graph_id in (None, "no_observations_yet"):
            return None
        if evidence < self._output_evidence_threshold:
            return None
        return str(graph_id)

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
        """Update terminal condition each matching step (both train and eval).

        During training with empty memory, this immediately sets terminal_state
        to "no_match", which causes MontyForEvidenceGraphMatching to switch to
        exploratory mode on the first on-object step—matching EvidenceGraphLM
        behavior.
        """
        self.update_terminal_condition()

    def _buffer_observation(self, state: State) -> None:
        try:
            self.buffer.append([state])
        except (ValueError, TypeError, KeyError, AttributeError):
            pass

    @staticmethod
    def _is_state_like(obj) -> bool:
        return hasattr(obj, "use_state") and hasattr(obj, "location")

    @staticmethod
    def _orthonormalize_pose_vectors(pose_vectors) -> np.ndarray:
        pose = np.asarray(pose_vectors, dtype=np.float64).reshape(3, 3)
        try:
            left, _, right = np.linalg.svd(pose, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.eye(3, dtype=np.float64)

        orthonormal = left @ right
        if float(np.linalg.det(orthonormal)) < 0:
            left[:, -1] *= -1.0
            orthonormal = left @ right
        return orthonormal.astype(np.float64)

    def _reset_inferred_state(self) -> None:
        self._inferred_location = None
        self._inferred_pose_vectors = None
        self._inferred_flow_vector = None
        self._last_observed_location = None
        self._last_observed_pose_vectors = None

    def _get_observed_pose_vectors(self, state: State) -> np.ndarray | None:
        morphological = getattr(state, "morphological_features", None) or {}
        pose_vectors = morphological.get("pose_vectors")
        if pose_vectors is None:
            return None

        try:
            return self._orthonormalize_pose_vectors(pose_vectors)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None

    def _estimate_inferred_location_prior(
        self,
        observed_location: np.ndarray,
    ) -> np.ndarray | None:
        if not self._use_tracker_location_prior:
            return None

        mlh = self._column.get_current_mlh()
        object_id = mlh.get("graph_id")
        if object_id in (None, "no_observations_yet"):
            return None

        tracker = getattr(self._column, "_predictive_tracker", None)
        if tracker is not None and tracker.is_anchored(object_id):
            return np.asarray(tracker.get_location(object_id), dtype=np.float64)

        estimator = getattr(self._column, "_reference_frame_estimator", None)
        if estimator is not None and estimator.has_rotation(object_id):
            return np.asarray(
                estimator.transform(object_id, observed_location),
                dtype=np.float64,
            )

        return None

    def _update_inferred_location(self, state: State) -> np.ndarray:
        observed_location = np.asarray(state.location, dtype=np.float64)

        if (
            self._inferred_location is None
            or self._last_observed_location is None
        ):
            inferred_location = np.zeros(3, dtype=np.float64)
        else:
            inferred_location = self._inferred_location + (
                observed_location - self._last_observed_location
            )

        prior_location = self._estimate_inferred_location_prior(observed_location)
        if prior_location is not None:
            weight = np.clip(self._inferred_location_correction_weight, 0.0, 1.0)
            inferred_location = (
                (1.0 - weight) * inferred_location + weight * prior_location
            )

        self._last_observed_location = observed_location.copy()
        self._inferred_location = inferred_location.astype(np.float64)
        return self._inferred_location.copy()

    def _update_inferred_pose_vectors(self, state: State) -> np.ndarray:
        observed_pose = self._get_observed_pose_vectors(state)
        if observed_pose is None:
            if self._inferred_pose_vectors is None:
                self._inferred_pose_vectors = np.eye(3, dtype=np.float64)
            return self._inferred_pose_vectors.copy()

        if (
            self._inferred_pose_vectors is None
            or self._last_observed_pose_vectors is None
        ):
            inferred_pose = np.eye(3, dtype=np.float64)
        else:
            relative_rotation = observed_pose @ self._last_observed_pose_vectors.T
            inferred_pose = self._orthonormalize_pose_vectors(
                relative_rotation @ self._inferred_pose_vectors
            )

        self._last_observed_pose_vectors = observed_pose.copy()
        self._inferred_pose_vectors = inferred_pose.astype(np.float64)
        return self._inferred_pose_vectors.copy()

    def _update_inferred_flow_features(
        self,
        non_morphological_features: dict[str, Any],
    ) -> dict[str, Any]:
        if not non_morphological_features:
            return {}

        flow_direction = non_morphological_features.get("flow_direction")
        flow_magnitude = non_morphological_features.get("flow_magnitude")
        if flow_direction is None and flow_magnitude is None:
            return dict(non_morphological_features)

        if flow_direction is None:
            observed_vector = np.zeros(3, dtype=np.float64)
        else:
            observed_direction = np.asarray(flow_direction, dtype=np.float64).reshape(-1)
            observed_vector = np.zeros(3, dtype=np.float64)
            observed_vector[: min(3, observed_direction.size)] = observed_direction[:3]

        if flow_magnitude is not None:
            observed_magnitude = float(np.asarray(flow_magnitude, dtype=np.float64).reshape(-1)[0])
            observed_vector = observed_vector * observed_magnitude

        if self._inferred_flow_vector is None:
            inferred_vector = observed_vector
        else:
            update_rate = np.clip(self._inferred_flow_update_rate, 0.0, 1.0)
            inferred_vector = (
                (1.0 - update_rate) * self._inferred_flow_vector
                + update_rate * observed_vector
            )

        self._inferred_flow_vector = inferred_vector.astype(np.float64)

        updated = dict(non_morphological_features)
        magnitude = float(np.linalg.norm(self._inferred_flow_vector))
        updated["flow_magnitude"] = np.asarray([magnitude], dtype=np.float64)
        if magnitude > 1e-8:
            updated["flow_direction"] = (
                self._inferred_flow_vector / magnitude
            ).astype(np.float64)
        else:
            updated["flow_direction"] = np.zeros(3, dtype=np.float64)

        return updated

    def _build_inferred_input_state(self, state: State) -> State:
        inferred_location = self._update_inferred_location(state)
        inferred_pose_vectors = self._update_inferred_pose_vectors(state)

        morphological_features = dict(state.morphological_features or {})
        morphological_features["pose_vectors"] = inferred_pose_vectors
        morphological_features["pose_fully_defined"] = bool(
            morphological_features.get("pose_fully_defined", True)
        )

        non_morphological_features = self._update_inferred_flow_features(
            dict(state.non_morphological_features or {})
        )

        return State(
            location=inferred_location,
            morphological_features=morphological_features,
            non_morphological_features=non_morphological_features,
            confidence=float(getattr(state, "confidence", 0.0)),
            use_state=bool(getattr(state, "use_state", False)),
            sender_id=str(getattr(state, "sender_id", self.learning_module_id)),
            sender_type=str(getattr(state, "sender_type", "SM")),
            inferred_state=getattr(state, "inferred_state", None),
        )

    def _prepare_input_state(self, state: State) -> State:
        self._last_observed_state = state
        if not self._use_inferred_state:
            return state
        if getattr(state, "sender_type", "SM") == "LM":
            return state
        return self._build_inferred_input_state(state)

    def _get_state_for_emission(self) -> State | None:
        if self._last_input_state is not None:
            return self._last_input_state
        return self._last_observed_state

    def _get_vote_transport_pose(self) -> tuple[np.ndarray, np.ndarray]:
        transport_state = self._last_observed_state or self._last_input_state
        if transport_state is None:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

        location = np.asarray(transport_state.location, dtype=np.float64)
        pose_vectors = np.asarray(
            (transport_state.morphological_features or {}).get(
                "pose_vectors", np.eye(3)
            ),
            dtype=np.float64,
        ).reshape(3, 3)
        return location, self._orthonormalize_pose_vectors(pose_vectors)

    def _fuse_lm_context(self, states) -> np.ndarray | None:
        accumulated = None
        total_weight = 0.0

        for state in states:
            non_morphological = getattr(state, "non_morphological_features", None)
            if not isinstance(non_morphological, dict):
                continue

            active_cells = non_morphological.get("active_cells")
            if active_cells is None:
                continue

            context = np.asarray(active_cells, dtype=np.float32).reshape(-1)
            if context.size == 0:
                continue

            if accumulated is None:
                accumulated = np.zeros_like(context, dtype=np.float32)
            elif accumulated.shape != context.shape:
                source_positions = np.linspace(0.0, 1.0, num=context.size)
                target_positions = np.linspace(0.0, 1.0, num=accumulated.size)
                context = np.interp(target_positions, source_positions, context).astype(
                    np.float32
                )

            weight = max(float(getattr(state, "confidence", 0.0)), 1e-6)
            accumulated += context * weight
            total_weight += weight

        if accumulated is None or total_weight <= 0:
            return None

        return accumulated / total_weight

    def _fuse_lm_states(self, states) -> State | None:
        if not states:
            return None
        if len(states) == 1:
            return states[0]

        ranked_states = sorted(
            states,
            key=lambda state: (
                -float(getattr(state, "confidence", 0.0)),
                str(getattr(state, "sender_id", "")),
            ),
        )
        weights = np.asarray(
            [max(float(getattr(state, "confidence", 0.0)), 1e-6) for state in ranked_states],
            dtype=np.float64,
        )
        weights = weights / max(float(weights.sum()), 1e-8)
        anchor = ranked_states[0]

        location = np.zeros(3, dtype=np.float64)
        pose_accumulator = np.zeros((3, 3), dtype=np.float64)
        on_object = True
        for weight, state in zip(weights, ranked_states):
            location += weight * np.asarray(state.location, dtype=np.float64)
            pose_accumulator += weight * np.asarray(
                state.morphological_features.get("pose_vectors", np.eye(3)),
                dtype=np.float64,
            )
            on_object = on_object and bool(
                state.morphological_features.get("on_object", True)
            )

        non_morphological = dict(
            getattr(anchor, "non_morphological_features", {}) or {}
        )
        fused_context = self._fuse_lm_context(ranked_states)
        if fused_context is not None:
            non_morphological["active_cells"] = fused_context
        non_morphological["child_sender_ids"] = [
            str(getattr(state, "sender_id", "")) for state in ranked_states
        ]
        non_morphological["child_graph_ids"] = [
            ((getattr(state, "non_morphological_features", None) or {}).get("graph_id"))
            for state in ranked_states
        ]
        non_morphological["child_confidences"] = [
            float(getattr(state, "confidence", 0.0)) for state in ranked_states
        ]

        return State(
            location=location,
            morphological_features={
                "pose_vectors": self._orthonormalize_pose_vectors(pose_accumulator),
                "pose_fully_defined": True,
                "on_object": on_object,
            },
            non_morphological_features=non_morphological,
            confidence=float(max(float(state.confidence) for state in ranked_states)),
            use_state=True,
            sender_id="fused:" + "+".join(non_morphological["child_sender_ids"]),
            sender_type="LM",
            inferred_state=getattr(anchor, "inferred_state", None),
        )

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

        lm_states = [
            state
            for state in states
            if getattr(state, "sender_type", None) == "LM"
            and getattr(state, "use_state", False)
        ]
        fused_context = self._fuse_lm_context(lm_states)
        if fused_context is not None:
            self._set_external_context(fused_context)

        # First usable SM state
        for s in states:
            if (
                getattr(s, "use_state", False)
                and getattr(s, "sender_type", "SM") != "LM"
            ):
                return s

        fused_lm_state = self._fuse_lm_states(lm_states)
        if fused_lm_state is not None:
            return fused_lm_state

        # Fallback
        for s in states:
            if getattr(s, "use_state", False):
                return s

        return None
