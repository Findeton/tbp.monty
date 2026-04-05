# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""CorticalColumnTorchLM experiment with real Panda3D rendering and 3D models.

Mirrors ``Panda3DCorticalColumnExperiment`` but uses CorticalColumnTorchLM
(PyTorch modern Hopfield) as the learning module.

Supports:
- Flat mode: 1 CameraSM → 1 CorticalColumnTorchLM
- Hierarchical mode: CameraSM + ChangeDetectingSM → 2 child LMs + 1 parent LM
- Real 3D models (glTF/GLB) with skeletal animation
- Real sensor processing (CameraSM: depth→3D, HSV, curvatures)
- Real motor control (InformedPolicy with Panda3D actuators)
- Multi-LM voting (surprise-gated when hierarchical)
- Apical dendrites for top-down context

Usage::

    exp = Panda3DTorchExperiment(
        model_path="Fox.glb",
        asset_search_paths=["/path/to/assets"],
        hierarchical=True,
    )
    exp.train("fox_walk", anim_name="Walk")
    result = exp.evaluate(anim_name="Walk")
    exp.close()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import quaternion
import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)
from tbp.monty.frameworks.models.motor_policies import InformedPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize

logger = logging.getLogger(__name__)


class AxisAwareConstantSampler(ConstantSampler):
    """Constant sampler with explicit reverse and tangential axis moves."""

    def __init__(
        self,
        *args,
        allow_backward_actions=False,
        tangential_directions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._allow_backward_actions = bool(allow_backward_actions)
        self._tangential_directions = [
            tuple(float(value) for value in direction)
            for direction in (
                tangential_directions
                or [
                    (-1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                ]
            )
        ]

    def sample_move_forward(self, agent_id, rng):
        distance = float(self.translation_distance)
        if self._allow_backward_actions and float(rng.rand()) < 0.5:
            distance = -distance
        return MoveForward(agent_id=agent_id, distance=distance)

    def sample_move_tangentially(self, agent_id, rng):
        direction = self.direction
        if self._tangential_directions:
            direction = self._tangential_directions[
                int(rng.randint(len(self._tangential_directions)))
            ]
        return MoveTangentially(
            agent_id=agent_id,
            distance=float(self.translation_distance),
            direction=direction,
        )


class Panda3DTorchExperiment:
    """Train and evaluate CorticalColumnTorchLMs with real Panda3D rendering.

    Parameters
    ----------
    model_path : str or Path
        Path to a glTF/GLB 3D model file.
    hierarchical : bool
        If True, 3-LM heterarchy. If False, flat single-LM.
    resolution : tuple
        Render resolution (height, width).
    fov : float
        Horizontal field of view.
    near, far : float
        Clip planes.
    initial_distance : float
        Camera distance from object.
    object_position, object_scale : tuple
        Object placement.
    column_kwargs : dict or None
        Base kwargs for CorticalColumnTorch.
    parent_column_kwargs : dict or None
        Kwargs for parent LM's column (hierarchical only).
    camera_features : list or None
        Features for CameraSM.
    flow_threshold : float
        ChangeDetectingSM threshold.
    asset_search_paths : list or None
        Directories for model resolution.
    rotation_degrees, translation_distance : float
        Motor policy step sizes.
    hopfield_voting : bool
        If True, enable surprise-gated Hopfield voting between LMs.
    hopfield_surprise_threshold : float
        Surprise threshold for Hopfield voting gating.
    """

    CAMERA_SM_ID = "patch_0"
    CHANGE_SM_ID = "change_detector"
    _ACTOR_SYNC_RENDER_PASSES = 3
    _EPISODE_SENSOR_WARMUP_STEPS = 1

    def __init__(
        self,
        model_path,
        hierarchical=False,
        resolution=(64, 64),
        fov=90.0,
        near=0.01,
        far=10.0,
        initial_distance=3.0,
        object_position=(0.0, 0.0, 0.0),
        object_scale=(1.0, 1.0, 1.0),
        column_kwargs=None,
        morphology_column_kwargs=None,
        behavior_column_kwargs=None,
        parent_column_kwargs=None,
        camera_features=None,
        flow_threshold=0.005,
        asset_search_paths=None,
        rotation_degrees=5.0,
        translation_distance=0.004,
        motor_actions=None,
        allow_backward_actions=False,
        tangential_directions=None,
        hopfield_voting=False,
        hopfield_surprise_threshold=0.3,
        conditional_voting=False,
        vote_confident_threshold=2,
        predictive_voting=False,
        temporal_confusion_threshold=0.5,
        vote_after_steps=2,
        vote_cooldown_steps=0,
        goal_state_driven_actions=False,
        authoritative_goal_sender_ids=None,
        allow_action_sampler_fallback=False,
        lm_kwargs=None,
        morphology_lm_kwargs=None,
        behavior_lm_kwargs=None,
        parent_lm_kwargs=None,
    ):
        self._model_path = Path(model_path)
        self._hierarchical = hierarchical
        self._resolution = resolution
        self._fov = fov
        self._near = near
        self._far = far
        self._initial_distance = initial_distance
        self._object_position = object_position
        self._object_scale = object_scale
        self._column_kwargs = column_kwargs or {}
        self._morphology_column_kwargs = morphology_column_kwargs or {}
        self._behavior_column_kwargs = behavior_column_kwargs or {}
        self._parent_column_kwargs = parent_column_kwargs or {}
        self._seed = int(self._column_kwargs.get("seed", 42))
        self._camera_features = camera_features or [
            "pose_vectors", "on_object", "hsv",
        ]
        self._flow_threshold = flow_threshold
        self._asset_search_paths = asset_search_paths or []
        self._rotation_degrees = rotation_degrees
        self._translation_distance = translation_distance
        self._motor_actions = list(
            motor_actions
            or [MoveForward, TurnLeft, TurnRight, LookUp, LookDown]
        )
        self._allow_backward_actions = bool(allow_backward_actions)
        self._tangential_directions = list(tangential_directions or [])
        self._hopfield_voting = hopfield_voting
        self._hopfield_surprise_threshold = hopfield_surprise_threshold
        self._conditional_voting = conditional_voting
        self._vote_confident_threshold = int(vote_confident_threshold)
        self._predictive_voting = predictive_voting
        self._temporal_confusion_threshold = temporal_confusion_threshold
        self._vote_after_steps = vote_after_steps
        self._vote_cooldown_steps = int(vote_cooldown_steps)
        self._goal_state_driven_actions = bool(goal_state_driven_actions)
        self._authoritative_goal_sender_ids = list(
            authoritative_goal_sender_ids or []
        )
        self._allow_action_sampler_fallback = allow_action_sampler_fallback
        self._lm_kwargs = lm_kwargs or {}
        self._morphology_lm_kwargs = morphology_lm_kwargs or {}
        self._behavior_lm_kwargs = behavior_lm_kwargs or {}
        self._parent_lm_kwargs = parent_lm_kwargs or {}

        self._agent_id = AgentID("agent_id_0")
        self._panda3d_sensor_id = SensorID("sensor_0")

        self._sim = None
        self._obj_id = None
        self._anim_obj = None
        self._depth_transform = None
        self._d3d_transform = None
        self._monty = None

    def _sync_render(self, passes=None):
        if self._sim is None:
            return

        render_passes = int(
            self._ACTOR_SYNC_RENDER_PASSES if passes is None else passes
        )
        for _ in range(max(1, render_passes)):
            self._sim._render()

    def _warmup_episode_observation(self, anim_name=None, frame=None, steps=None):
        if self._sim is None:
            return

        warmup_steps = int(
            self._EPISODE_SENSOR_WARMUP_STEPS if steps is None else steps
        )
        if warmup_steps <= 0:
            return

        warmup_frame = 0 if frame is None else int(frame)
        # Fresh Panda3D offscreen buffers can be one observation behind the
        # newly positioned camera on a cold start. Discard a small number of
        # no-op reads before the first LM step so training/eval begins from a
        # settled sensor frame.
        for idx in range(warmup_steps):
            self._step_environment(
                [],
                frame=warmup_frame,
                anim_name=anim_name,
                step=-(idx + 1),
            )

    # ======================== Setup ========================

    def _setup(self):
        """Initialize Panda3D, sensors, CorticalColumnTorchLMs, and Monty."""
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
        )

        agent = Panda3DAgent(
            agent_id=self._agent_id,
            sensor_id=str(self._panda3d_sensor_id),
            resolution=self._resolution,
            fov=self._fov,
        )
        self._sim = Panda3DSimulator(
            agents=[agent],
            near=self._near,
            far=self._far,
            asset_search_paths=self._asset_search_paths,
        )

        info = self._sim.add_object(
            name=str(self._model_path),
            position=self._object_position,
            scale=self._object_scale,
            animated=True,
        )
        self._obj_id = info.object_id
        self._anim_obj = self._sim.get_animated_object(self._obj_id)
        # Newly swapped animated Actors can report stale transforms until the
        # scene graph has been advanced at least once. Give Panda3D a few sync
        # render passes here so subsequent animation inspection and the first
        # episode start from a settled pose state.
        self._sync_render()

        self._depth_transform = Panda3DDepthNormalize(
            agent_id=self._agent_id,
            near=self._near,
            far=self._far,
        )
        self._d3d_transform = DepthTo3DLocations(
            agent_id=self._agent_id,
            sensor_ids=[self._panda3d_sensor_id],
            resolutions=[self._resolution],
            hfov=self._fov,
            world_coord=True,
            get_all_points=True,
        )

        if self._hierarchical:
            self._setup_hierarchical()
        else:
            self._setup_flat()

        logger.info(
            "Panda3DTorchExperiment: model=%s, hierarchical=%s, "
            "animations=%s, n_lms=%d",
            self._model_path.name,
            self._hierarchical,
            self._anim_obj.animation_names,
            len(self._monty.learning_modules),
        )

    def _setup_flat(self):
        sm = CameraSM(
            sensor_module_id=self.CAMERA_SM_ID,
            features=self._camera_features,
        )

        col_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=False,
            seed=self._seed,
        )
        col_kw.update(self._column_kwargs)

        lm = CorticalColumnTorchLM(
            column_kwargs=col_kw,
            learning_module_id="lm_0",
            **self._lm_kwargs,
        )

        motor_system = self._build_motor_system()

        self._monty = MontyForEvidenceGraphMatching(
            sensor_modules=[sm],
            learning_modules=[lm],
            motor_system=motor_system,
            sm_to_agent_dict={self.CAMERA_SM_ID: self._agent_id},
            sm_to_lm_matrix=[[0]],
            lm_to_lm_matrix=[[]],
            lm_to_lm_vote_matrix=[[]],
            min_eval_steps=9999,
            min_train_steps=9999,
            num_exploratory_steps=9999,
            max_total_steps=99999,
            conditional_voting=self._conditional_voting,
            vote_confident_threshold=self._vote_confident_threshold,
            hopfield_voting=self._hopfield_voting,
            hopfield_surprise_threshold=self._hopfield_surprise_threshold,
            predictive_voting=self._predictive_voting,
            temporal_confusion_threshold=self._temporal_confusion_threshold,
            vote_after_steps=self._vote_after_steps,
            vote_cooldown_steps=self._vote_cooldown_steps,
            authoritative_goal_sender_ids=self._authoritative_goal_sender_ids,
        )

    def _setup_hierarchical(self):
        sm_camera = CameraSM(
            sensor_module_id=self.CAMERA_SM_ID,
            features=self._camera_features,
        )
        sm_change = ChangeDetectingSM(
            sensor_module_id=self.CHANGE_SM_ID,
            flow_threshold=self._flow_threshold,
            global_flow_suppression=False,
        )

        child_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=True,
            seed=self._seed,
        )
        child_kw.update(self._column_kwargs)
        behavior_seed = int(child_kw.get("seed", self._seed)) + 1
        morphology_child_kw = dict(child_kw)
        morphology_child_kw.update(self._morphology_column_kwargs)
        behavior_child_kw = dict(child_kw)
        behavior_child_kw.update(self._behavior_column_kwargs)
        behavior_child_kw.setdefault("seed", behavior_seed)

        lm_morph = CorticalColumnTorchLM(
            column_kwargs=morphology_child_kw,
            learning_module_id="lm_morphology",
            **self._morphology_lm_kwargs,
        )
        lm_behav = CorticalColumnTorchLM(
            column_kwargs=behavior_child_kw,
            learning_module_id="lm_behavior",
            **self._behavior_lm_kwargs,
        )

        parent_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=True,
            defer_context_auto_label=True,
            seed=behavior_seed + 1,
        )
        parent_kw.update(self._parent_column_kwargs)

        parent_lm_kwargs = dict(
            expected_context_sender_ids=["lm_morphology", "lm_behavior"],
            context_identity_weight=0.35,
            use_child_graph_context_labels=True,
        )
        parent_lm_kwargs.update(self._parent_lm_kwargs)

        lm_parent = CorticalColumnTorchLM(
            column_kwargs=parent_kw,
            learning_module_id="lm_parent",
            **parent_lm_kwargs,
        )

        motor_system = self._build_motor_system()

        # Enable hopfield_voting on child LMs when requested
        if self._hopfield_voting:
            lm_morph._hopfield_voting = True
            lm_morph._surprise_vote_threshold = self._hopfield_surprise_threshold
            lm_behav._hopfield_voting = True
            lm_behav._surprise_vote_threshold = self._hopfield_surprise_threshold

        self._monty = MontyForEvidenceGraphMatching(
            sensor_modules=[sm_camera, sm_change],
            learning_modules=[lm_morph, lm_behav, lm_parent],
            motor_system=motor_system,
            sm_to_agent_dict={
                self.CAMERA_SM_ID: self._agent_id,
                self.CHANGE_SM_ID: self._agent_id,
            },
            sm_to_lm_matrix=[[0], [1], []],
            lm_to_lm_matrix=[[], [], [0, 1]],
            lm_to_lm_vote_matrix=[[1], [0], []],
            min_eval_steps=9999,
            min_train_steps=9999,
            num_exploratory_steps=9999,
            max_total_steps=99999,
            conditional_voting=self._conditional_voting,
            vote_confident_threshold=self._vote_confident_threshold,
            hopfield_voting=self._hopfield_voting,
            hopfield_surprise_threshold=self._hopfield_surprise_threshold,
            predictive_voting=self._predictive_voting,
            temporal_confusion_threshold=self._temporal_confusion_threshold,
            vote_after_steps=self._vote_after_steps,
            vote_cooldown_steps=self._vote_cooldown_steps,
            authoritative_goal_sender_ids=self._authoritative_goal_sender_ids,
        )

    def _build_motor_system(self):
        action_sampler = AxisAwareConstantSampler(
            actions=self._motor_actions,
            rotation_degrees=self._rotation_degrees,
            translation_distance=self._translation_distance,
            allow_backward_actions=self._allow_backward_actions,
            tangential_directions=self._tangential_directions,
        )
        motor_policy = InformedPolicy(
            action_sampler=action_sampler,
            agent_id=self._agent_id,
            use_goal_state_driven_actions=self._goal_state_driven_actions,
            view_finder_id="view_finder",
        )
        return MotorSystem(policy=motor_policy)

    def _resolve_frame_schedule(self, anim_name=None, n_steps=None, frame_schedule=None):
        animated = anim_name is not None
        n_frames = self._anim_obj.get_num_frames(anim_name) if animated else 1

        if frame_schedule is not None:
            frames = [int(frame) % max(n_frames, 1) for frame in frame_schedule]
        elif animated:
            total_steps = n_steps or n_frames
            frames = [step % n_frames for step in range(total_steps)]
        else:
            total_steps = n_steps or 30
            frames = [0 for _ in range(total_steps)]

        return frames, n_frames

    def _apply_stepwise_targets(
        self,
        object_name,
        step,
        frame,
        total_steps,
        n_frames,
        state_provider=None,
    ):
        state_targets = {}

        for lm in self._monty.learning_modules:
            lm.stepwise_target_object = object_name
            lm.stepwise_target_state = None

        if state_provider is None:
            return state_targets

        state_targets = state_provider(
            step=step,
            frame=frame,
            total_steps=total_steps,
            n_frames=n_frames,
            object_name=object_name,
            experiment=self,
        )
        if state_targets is None:
            return {}
        if not isinstance(state_targets, dict):
            raise TypeError("state_provider must return a dict or None")

        for lm in self._monty.learning_modules:
            if lm.learning_module_id in state_targets:
                lm.stepwise_target_state = state_targets[lm.learning_module_id]

        return dict(state_targets)

    @staticmethod
    def _encode_action_context(actions):
        context = np.zeros(8, dtype=np.float32)
        if not actions:
            return context

        for action in actions:
            name = str(getattr(action, "name", "")).lower()
            rotation = float(getattr(action, "rotation_degrees", 0.0))
            distance = float(getattr(action, "distance", 0.0))
            context[5] += 1.0

            if name == "move_forward":
                context[0] += distance
                context[6] += 1.0
                continue

            if name == "turn_left":
                context[1] -= rotation
                context[7] += 1.0
                continue

            if name == "turn_right":
                context[1] += rotation
                context[7] += 1.0
                continue

            if name == "look_up":
                context[2] += rotation
                context[7] += 1.0
                continue

            if name == "look_down":
                context[2] -= rotation
                context[7] += 1.0
                continue

            if name == "move_tangentially":
                direction = np.asarray(
                    getattr(action, "direction", (0.0, 0.0, 0.0)),
                    dtype=np.float32,
                ).reshape(-1)
                if direction.size > 0:
                    context[3] += distance * float(direction[0])
                if direction.size > 1:
                    context[0] += distance * float(direction[1])
                if direction.size > 2:
                    context[4] += distance * float(direction[2])
                context[6] += 1.0
                continue

            if name == "orient_horizontal":
                context[1] += rotation
                context[3] += float(getattr(action, "left_distance", 0.0))
                context[0] += float(getattr(action, "forward_distance", 0.0))
                context[6] += 1.0
                context[7] += 1.0
                continue

            if name == "orient_vertical":
                context[2] -= rotation
                context[4] -= float(getattr(action, "down_distance", 0.0))
                context[0] += float(getattr(action, "forward_distance", 0.0))
                context[6] += 1.0
                context[7] += 1.0
                continue

            if name in {"set_agent_pitch", "set_sensor_pitch"}:
                context[2] += rotation
                context[7] += 1.0
                continue

            if name == "set_yaw":
                context[1] += rotation
                context[7] += 1.0

        return context

    @staticmethod
    def _simplify_temporal_context(context):
        if not context:
            return None

        keep = [
            "current_label",
            "predicted_label",
            "current_dwell",
            "event_detected",
            "step_count",
            "mean_surprise",
            "known_states",
            "match_score",
            "trace_norm",
            "boundary_pressure",
            "trace_discontinuity",
            "trace_scales",
        ]
        simplified = {}
        for key in keep:
            if key not in context:
                continue
            value = context[key]
            if isinstance(value, np.generic):
                value = value.item()
            simplified[key] = value

        return simplified or None

    @staticmethod
    def _copy_context_signal(context_signal):
        if not context_signal:
            return None

        simplified = {
            "sender_id": context_signal.get("sender_id"),
            "graph_id": context_signal.get("graph_id"),
            "confidence": float(context_signal.get("confidence", 0.0)),
            "sender_step_count": int(context_signal.get("sender_step_count", 0)),
        }

        active_cells = context_signal.get("active_cells")
        if isinstance(active_cells, np.ndarray):
            simplified["active_cells"] = active_cells.astype(np.float32).copy()
        elif isinstance(active_cells, torch.Tensor):
            simplified["active_cells"] = (
                active_cells.detach().cpu().float().numpy().copy()
            )
        else:
            simplified["active_cells"] = None

        return simplified

    def _collect_step_trace(
        self,
        step,
        frame,
        state_targets=None,
        collect_context_signals=False,
    ):
        trace = {
            "step": int(step),
            "frame": int(frame),
            "state_targets": dict(state_targets or {}),
            "learning_modules": {},
        }

        for lm in self._monty.learning_modules:
            lm_trace = {}
            if hasattr(lm, "get_current_mlh"):
                mlh = lm.get_current_mlh()
                lm_trace["graph_id"] = mlh.get("graph_id")
                lm_trace["evidence"] = float(mlh.get("evidence", 0.0))
            if hasattr(lm, "get_temporal_prediction_status"):
                lm_trace["temporal_status"] = lm.get_temporal_prediction_status()
            if hasattr(lm, "get_temporal_surprise"):
                lm_trace["temporal_surprise"] = float(lm.get_temporal_surprise())
            if hasattr(lm, "get_temporal_context"):
                context = self._simplify_temporal_context(lm.get_temporal_context())
                if context is not None:
                    lm_trace["temporal_context"] = context
            if hasattr(lm, "get_evidence_debug"):
                evidence_debug = lm.get_evidence_debug()
                if evidence_debug is not None:
                    lm_trace["evidence_debug"] = evidence_debug
            if collect_context_signals and hasattr(lm, "get_context_signal"):
                context_signal = self._copy_context_signal(lm.get_context_signal())
                if context_signal is not None:
                    lm_trace["context_signal"] = context_signal

            trace["learning_modules"][lm.learning_module_id] = lm_trace

        return trace

    def _sample_fallback_actions(self, ctx):
        if self._monty is None:
            return []

        policy = self._monty.motor_system._policy
        action_sampler = getattr(policy, "action_sampler", None)
        agent_id = getattr(policy, "agent_id", self._agent_id)
        if action_sampler is None or not hasattr(action_sampler, "sample"):
            return []

        action = action_sampler.sample(agent_id, ctx.rng)
        if action is None:
            return []

        if hasattr(policy, "fixme_undo_last_action"):
            try:
                policy._undo_action = policy.fixme_undo_last_action(action)
            except AttributeError:
                pass

        return [action]

    def run_episode(
        self,
        mode,
        object_name=None,
        anim_name=None,
        n_steps=None,
        frame_schedule=None,
        state_provider=None,
        collect_trace=False,
        collect_context_signals=False,
        collect_action_history=False,
        forced_action_sequences=None,
        action_context_mode="executed",
    ):
        """Run a real animated-mesh episode with optional per-step targets."""
        if self._sim is None:
            self._setup()

        if action_context_mode not in {"executed", "none"}:
            raise ValueError(
                "action_context_mode must be 'executed' or 'none'"
            )

        frames, n_frames = self._resolve_frame_schedule(
            anim_name=anim_name,
            n_steps=n_steps,
            frame_schedule=frame_schedule,
        )
        total_steps = len(frames)
        if (
            forced_action_sequences is not None
            and len(forced_action_sequences) != total_steps
        ):
            raise ValueError(
                "forced_action_sequences must match the episode length"
            )

        ctx = RuntimeContext(rng=np.random.RandomState(self._seed))
        target_name = object_name
        if mode is ExperimentMode.EVAL and target_name is None:
            target_name = "placeholder"
        target = {"object": target_name, "quat_rotation": [1, 0, 0, 0]}

        self._monty.set_experiment_mode(mode)
        self._monty.pre_episode(primary_target=target)

        if mode is ExperimentMode.TRAIN:
            self._monty.switch_to_exploratory_step()
            for sm in self._monty.sensor_modules:
                sm.is_exploring = True

        initial_frame = frames[0] if anim_name is not None and frames else None
        self._position_camera_initial(anim_name=anim_name, frame=initial_frame)
        self._warmup_episode_observation(
            anim_name=anim_name,
            frame=initial_frame,
        )

        pending_actions = []
        trace = []
        action_history = []
        action_context_history = []
        for step, frame in enumerate(frames):
            state_targets = self._apply_stepwise_targets(
                object_name=object_name,
                step=step,
                frame=frame,
                total_steps=total_steps,
                n_frames=n_frames,
                state_provider=state_provider,
            )

            if forced_action_sequences is None:
                actions_to_apply = pending_actions
            else:
                actions_to_apply = list(forced_action_sequences[step])

            obs = self._step_environment(
                actions_to_apply,
                frame=frame,
                anim_name=anim_name,
                step=step,
            )
            if action_context_mode == "none":
                action_context = np.zeros(8, dtype=np.float32)
            else:
                action_context = self._encode_action_context(actions_to_apply)
            for lm in self._monty.learning_modules:
                if hasattr(lm, "set_action_context"):
                    lm.set_action_context(action_context)
            generated_actions = self._monty.step(ctx, obs)
            if (
                not generated_actions
                and self._allow_action_sampler_fallback
            ):
                generated_actions = self._sample_fallback_actions(ctx)
            pending_actions = generated_actions

            if collect_action_history:
                action_history.append(list(actions_to_apply))
                action_context_history.append(action_context.copy())

            if object_name is not None:
                for lm in self._monty.learning_modules:
                    lm.stepwise_target_object = object_name

            if collect_trace:
                trace.append(
                    self._collect_step_trace(
                        step=step,
                        frame=frame,
                        state_targets=state_targets,
                        collect_context_signals=collect_context_signals,
                    )
                )

        result = {"total_steps": total_steps}

        if mode is ExperimentMode.TRAIN:
            for lm in self._monty.learning_modules:
                lm.detected_object = object_name
                lm.detected_rotation_r = None
                if lm.buffer.get_num_observations_on_object() > 0:
                    lm.buffer.stats["detected_location_rel_body"] = (
                        lm.buffer.get_current_location(input_channel="first")
                    )
                else:
                    lm.buffer.stats["detected_location_rel_body"] = np.zeros(3)

        self._monty.post_episode()

        if mode is ExperimentMode.TRAIN:
            for lm in self._monty.learning_modules:
                result[lm.learning_module_id] = {
                    "known_objects": lm.get_all_known_object_ids(),
                }
        else:
            for lm in self._monty.learning_modules:
                if hasattr(lm, "get_current_mlh"):
                    result[lm.learning_module_id] = dict(lm.get_current_mlh())

            primary_lm = self._monty.learning_modules[0]
            result["graph_id"] = primary_lm.get_current_mlh().get("graph_id")
            result["evidence"] = {
                lm.learning_module_id: dict(lm.evidence)
                for lm in self._monty.learning_modules
            }

        if collect_trace:
            result["trace"] = trace
            if collect_context_signals:
                result["trace_contains_context_signals"] = True
        if collect_action_history:
            result["action_history"] = action_history
            result["action_context_history"] = action_context_history
            result["action_context_mode"] = action_context_mode
            result["forced_action_replay"] = forced_action_sequences is not None

        return result

    # ======================== Public API ========================

    def train(self, object_name, anim_name=None, n_steps=None):
        """Train LMs on the loaded 3D model.

        Returns dict with training results per LM.
        """
        result = self.run_episode(
            mode=ExperimentMode.TRAIN,
            object_name=object_name,
            anim_name=anim_name,
            n_steps=n_steps,
        )
        total_steps = result["total_steps"]
        logger.info("Trained '%s' in %d steps: %s", object_name, total_steps, result)
        return result

    def evaluate(self, anim_name=None, n_steps=None):
        """Evaluate recognition on the loaded model.

        Returns dict with per-LM MLH and evidence.
        """
        result = self.run_episode(
            mode=ExperimentMode.EVAL,
            anim_name=anim_name,
            n_steps=n_steps,
        )

        logger.info("Evaluate result: %s", result.get("graph_id"))
        return result

    def swap_model(self, model_path, object_scale=None, initial_distance=None):
        """Replace the 3D model without rebuilding Monty.

        Parameters
        ----------
        model_path : str or Path
            Path to the new glTF/GLB model.
        object_scale : tuple or None
            Optional new object scale.
        initial_distance : float or None
            Optional new camera distance to use for subsequent episodes.
        """
        if self._sim is None:
            self._setup()

        self._model_path = Path(model_path)
        if object_scale is not None:
            self._object_scale = object_scale
        if initial_distance is not None:
            self._initial_distance = float(initial_distance)

        self._sim.remove_all_objects()
        info = self._sim.add_object(
            name=str(self._model_path),
            position=self._object_position,
            scale=self._object_scale,
            animated=True,
        )
        self._obj_id = info.object_id
        self._anim_obj = self._sim.get_animated_object(self._obj_id)
        self._sync_render()

    @property
    def monty(self):
        if self._monty is None:
            raise RuntimeError("Not initialized — call train() first")
        return self._monty

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ======================== Private ========================

    def _position_camera_initial(self, anim_name=None, frame=None):
        from panda3d.core import LVector3f

        cam_np = self._sim._agent_buffers[self._agent_id]["camera_np"]

        if anim_name is not None and frame is not None:
            self._anim_obj.pose(frame, anim_name)
            # Animated Actor joint transforms are not always stable
            # immediately after pose() on the first episode after a model
            # swap. Force a few render/update passes before querying bounds so
            # camera placement is based on the posed mesh, not stale or
            # singular transforms.
            self._sync_render()

        obj_np = self._sim._objects[self._obj_id]
        bounds = obj_np.getTightBounds()
        if bounds:
            lo, hi = bounds
            cx = float((lo[0] + hi[0]) / 2)
            cy = float((lo[1] + hi[1]) / 2)
            cz = float((lo[2] + hi[2]) / 2)
        else:
            cx, cy, cz = self._object_position

        cam_np.setPos(cx, cy - self._initial_distance, cz)
        cam_np.lookAt(LVector3f(cx, cy, cz))
        self._sync_motor_state()

    def _sync_motor_state(self):
        buf = self._sim._agent_buffers[self._agent_id]
        cam_np = buf["camera_np"]

        if "_last_set_position" in buf:
            position = buf.pop("_last_set_position")
        else:
            pos = cam_np.getPos()
            position = (float(pos[0]), float(pos[1]), float(pos[2]))

        if "_last_set_rotation" in buf:
            q = buf.pop("_last_set_rotation")
            rot_q = quaternion.quaternion(q[0], q[1], q[2], q[3])
        else:
            quat = cam_np.getQuat()
            rot_q = quaternion.quaternion(
                float(quat.getW()),
                float(quat.getX()),
                float(quat.getY()),
                float(quat.getZ()),
            )

        sm_ids = [self.CAMERA_SM_ID]
        if self._hierarchical:
            sm_ids.append(self.CHANGE_SM_ID)

        sensors = {}
        for sm_id in sm_ids:
            sensors[SensorID(sm_id)] = SensorState(
                position=(0.0, 0.0, 0.0), rotation=rot_q,
            )

        agent_state = AgentState(
            position=position, rotation=rot_q, sensors=sensors,
        )
        self._monty.motor_system._state = MotorSystemState(
            {self._agent_id: agent_state}
        )

    def _step_environment(self, actions, frame, anim_name, step=0):
        for action in actions:
            action.act(self._sim)

        if anim_name is not None:
            self._anim_obj.pose(frame, anim_name)

        raw_obs, proprio = self._sim.step([])
        self._sync_motor_state()

        transform_ctx = TransformContext(
            rng=np.random.RandomState(self._seed + int(step)), state=proprio,
        )
        raw_obs = self._depth_transform(raw_obs, transform_ctx)
        raw_obs = self._d3d_transform(raw_obs, transform_ctx)

        sensor_data = raw_obs[self._agent_id][self._panda3d_sensor_id]

        if self._hierarchical:
            obs = Observations({
                self._agent_id: {
                    self.CAMERA_SM_ID: sensor_data,
                    self.CHANGE_SM_ID: sensor_data,
                    "view_finder": sensor_data,
                }
            })
        else:
            obs = Observations({
                self._agent_id: {
                    self.CAMERA_SM_ID: sensor_data,
                    "view_finder": sensor_data,
                }
            })

        return obs
