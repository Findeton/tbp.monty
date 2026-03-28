# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Behavior training with proper Monty sensorimotor loop and Panda3D rendering.

2 SMs + 2 LMs architecture wired through MontyForEvidenceGraphMatching:

  SM 0: CameraSM           -> LM 0 (morphology)
  SM 1: ChangeDetectingSM  -> LM 1 (behavior)

  sm_to_lm_matrix:       [[0], [1]]
  lm_to_lm_vote_matrix:  [[1], [0]]

Motor system: InformedPolicy drives camera exploration via standard Monty
actions (MoveForward, TurnLeft, LookUp, etc.) executed by Panda3D actuators.
The animation advances each step independently of motor actions.

Per-step sensorimotor loop:
  1. Execute motor actions in Panda3D (camera moves)
  2. Advance animation to next frame
  3. Panda3D renders RGBA + depth
  4. Panda3DDepthNormalize + DepthTo3DLocations -> semantic_3d point cloud
  5. Observations duplicated under both SM keys
  6. monty.step() -> SMs process -> LMs update evidence -> motor policy decides
  7. Returned actions feed back to step 1

Usage::

    exp = Panda3DBehaviorExperiment(
        model_path="Fox.glb",
        object_scale=(0.01, 0.01, 0.01),
    )
    exp.train_behavior("Walk", "walk_behavior")
    exp.train_behavior("Run", "run_behavior")
    mlh = exp.match_behavior("Walk")
    print(mlh["graph_id"])  # "walk_behavior"
    exp.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
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


class Panda3DBehaviorExperiment:
    """Train and match behaviors using proper Monty sensorimotor loop.

    Architecture::

        SM 0: CameraSM          -> LM 0 (morphology)
        SM 1: ChangeDetectingSM  -> LM 1 (behavior)
        Lateral voting: LM 0 <-> LM 1
        Motor: InformedPolicy (explores object surface)

    The motor system produces actions (MoveForward, TurnLeft, etc.) that
    are executed by the Panda3D simulator's actuators. The animation
    advances independently each step, so the agent explores the object
    while the behavior unfolds.

    Parameters
    ----------
    model_path : str or Path
        Path to a glTF/GLB file with skeletal animation.
    resolution : tuple
        Render resolution (height, width).
    fov : float
        Horizontal field of view in degrees.
    near, far : float
        Near and far clip planes.
    initial_distance : float
        Starting distance from the object center.
    object_position, object_scale : tuple
        Object placement and scale in scene.
    flow_threshold : float
        Minimum flow magnitude for ChangeDetectingSM.
    camera_features : list
        Features for CameraSM to extract.
    morphology_lm_kwargs : dict or None
        Override defaults for morphology EvidenceGraphLM.
    behavior_lm_kwargs : dict or None
        Override defaults for behavior EvidenceGraphLM.
    hpc_kwargs : dict or None
        When provided, adds a HippocampalModule as a third LM that
        receives outputs from both morphology and behavior LMs.
        Passed as kwargs to HippocampalModule.__init__().
    goal_state_driven : bool
        When True, enables goal-state-driven motor control during eval.
        The morphology LM's GoalStateGenerator proposes hypothesis-testing
        jumps, and the motor policy executes them via SetAgentPose. During
        training, exploratory mode is used regardless of this setting.
    depth_from_motion : bool
        When True, replaces ground-truth depth with estimated depth from
        motion parallax (RGBA + ego-motion). Tests the system's ability
        to work without a depth sensor.
    asset_search_paths : list or None
        Additional directories for glTF model file resolution.
    rotation_degrees : float
        Rotation step size for motor policy actions.
    translation_distance : float
        Translation step size for motor policy actions.
    """

    CAMERA_SM_ID = "camera"
    CHANGE_SM_ID = "change_detector"

    def __init__(
        self,
        model_path,
        resolution=(64, 64),
        fov=90.0,
        near=0.01,
        far=10.0,
        initial_distance=3.0,
        object_position=(0.0, 0.0, 0.0),
        object_scale=(1.0, 1.0, 1.0),
        flow_threshold=0.005,
        camera_features=None,
        morphology_lm_kwargs=None,
        behavior_lm_kwargs=None,
        hpc_kwargs=None,
        goal_state_driven=False,
        depth_from_motion=False,
        asset_search_paths=None,
        rotation_degrees=5.0,
        translation_distance=0.004,
        # Legacy alias
        orbit_radius=None,
    ):
        self._model_path = Path(model_path)
        self._resolution = resolution
        self._fov = fov
        self._near = near
        self._far = far
        # orbit_radius is a legacy alias for initial_distance
        self._initial_distance = orbit_radius if orbit_radius is not None else initial_distance
        self._object_position = object_position
        self._object_scale = object_scale
        self._flow_threshold = flow_threshold
        self._camera_features = camera_features or [
            "pose_vectors", "on_object", "hsv",
        ]
        self._morphology_lm_kwargs = morphology_lm_kwargs or {}
        self._behavior_lm_kwargs = behavior_lm_kwargs or {}
        self._hpc_kwargs = hpc_kwargs
        self._goal_state_driven = goal_state_driven
        self._depth_from_motion = depth_from_motion
        self._asset_search_paths = asset_search_paths or []
        self._rotation_degrees = rotation_degrees
        self._translation_distance = translation_distance

        self._agent_id = AgentID("agent_id_0")
        self._panda3d_sensor_id = SensorID("sensor_0")

        # Lazily initialized in _setup()
        self._sim = None
        self._obj_id = None
        self._anim_obj = None
        self._depth_transform = None
        self._d3d_transform = None
        self._monty = None

    def _setup(self):
        """Initialize Panda3D, transforms, and MontyForEvidenceGraphMatching."""
        from tbp.monty.frameworks.environment_utils.transforms import (
            DepthTo3DLocations,
        )

        # -- Panda3D simulator --
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

        # -- Transforms --
        if self._depth_from_motion:
            from tbp.monty.simulators.panda3d.depth_from_motion import (
                DepthFromMotion,
            )
            self._depth_transform = DepthFromMotion(
                agent_id=self._agent_id,
                hfov=self._fov,
                resolution=self._resolution,
                far_plane=self._far,
            )
        else:
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

        # -- Sensor Modules --
        sm_camera = CameraSM(
            sensor_module_id=self.CAMERA_SM_ID,
            features=self._camera_features,
        )
        sm_change = ChangeDetectingSM(
            sensor_module_id=self.CHANGE_SM_ID,
            flow_threshold=self._flow_threshold,
            global_flow_suppression=False,
        )

        # -- Learning Modules --
        max_gs = max(15.0, self._initial_distance * 5)

        morph_defaults = dict(
            max_match_distance=0.5,
            tolerances={
                self.CAMERA_SM_ID: {
                    "hsv": [0.1, 0.2, 0.2],
                }
            },
            feature_weights={
                self.CAMERA_SM_ID: {
                    "hsv": np.array([1.0, 1.0, 1.0]),
                }
            },
            max_graph_size=max_gs,
            num_model_voxels_per_dim=50,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        # Goal-state-driven exploration: attach GSG to morphology LM
        if self._goal_state_driven:
            from tbp.monty.frameworks.models.goal_state_generation import (
                EvidenceGoalStateGenerator,
            )
            morph_defaults["gsg"] = EvidenceGoalStateGenerator(
                goal_tolerances={"location": 0.015},
                elapsed_steps_factor=5,
                min_post_goal_success_steps=3,
                desired_object_distance=self._initial_distance,
            )
        morph_defaults.update(self._morphology_lm_kwargs)
        lm_morphology = EvidenceGraphLM(**morph_defaults)

        behavior_defaults = dict(
            max_match_distance=0.5,
            tolerances={
                self.CHANGE_SM_ID: {
                    "flow_magnitude": [1.0],
                    "flow_direction": [0.5, 0.5, 0.5],
                }
            },
            feature_weights={
                self.CHANGE_SM_ID: {
                    "flow_magnitude": np.array([1.0]),
                    "flow_direction": np.array([1.0, 1.0, 1.0]),
                }
            },
            max_graph_size=max_gs,
            num_model_voxels_per_dim=50,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        behavior_defaults.update(self._behavior_lm_kwargs)
        lm_behavior = EvidenceGraphLM(**behavior_defaults)

        # -- Motor system: InformedPolicy with ConstantSampler --
        action_sampler = ConstantSampler(
            actions=[
                MoveForward, TurnLeft, TurnRight,
                LookUp, LookDown,
            ],
            rotation_degrees=self._rotation_degrees,
            translation_distance=self._translation_distance,
        )
        motor_policy = InformedPolicy(
            action_sampler=action_sampler,
            agent_id=self._agent_id,
            use_goal_state_driven_actions=self._goal_state_driven,
        )
        motor_system = MotorSystem(policy=motor_policy)

        # -- Optional HPC as third LM --
        learning_modules = [lm_morphology, lm_behavior]
        sm_to_lm_matrix = [[0], [1]]
        lm_to_lm_matrix = [[], []]
        lm_to_lm_vote_matrix = [[1], [0]]

        if self._hpc_kwargs is not None:
            from tbp.monty.frameworks.models.hippocampal_module import (
                HippocampalModule,
            )
            lm_hpc = HippocampalModule(**self._hpc_kwargs)
            lm_hpc.learning_module_id = "lm_hpc"
            learning_modules.append(lm_hpc)
            # HPC has no SM connections; receives from both LM 0 and LM 1
            sm_to_lm_matrix.append([])
            lm_to_lm_matrix = [[], [], [0, 1]]
            # HPC doesn't participate in voting
            lm_to_lm_vote_matrix = [[1], [0], []]

        # -- MontyForEvidenceGraphMatching --
        self._monty = MontyForEvidenceGraphMatching(
            sensor_modules=[sm_camera, sm_change],
            learning_modules=learning_modules,
            motor_system=motor_system,
            sm_to_agent_dict={
                self.CAMERA_SM_ID: self._agent_id,
                self.CHANGE_SM_ID: self._agent_id,
            },
            sm_to_lm_matrix=sm_to_lm_matrix,
            lm_to_lm_matrix=lm_to_lm_matrix,
            lm_to_lm_vote_matrix=lm_to_lm_vote_matrix,
            min_eval_steps=9999,
            min_train_steps=9999,
            num_exploratory_steps=9999,
            max_total_steps=99999,
        )
        # Assign LM IDs (normally done by MontyExperiment)
        lm_morphology.learning_module_id = "lm_morphology"
        lm_behavior.learning_module_id = "lm_behavior"

        logger.info(
            "Panda3DBehaviorExperiment: model=%s, animations=%s, hpc=%s",
            self._model_path.name,
            self._anim_obj.animation_names,
            self._hpc_kwargs is not None,
        )

    # ======================== Public API ========================

    def train_behavior(
        self,
        anim_name,
        object_name=None,
        n_repetitions=1,
        *,
        morphology_name=None,
        behavior_name=None,
    ):
        """Train both LMs on a behavior (animation).

        Runs the full sensorimotor loop: the motor policy explores the
        object while the animation plays. CameraSM extracts morphological
        features, ChangeDetectingSM extracts motion features.

        Each LM can learn under a different ID so that morphology and
        behavior are stored independently.  For example::

            exp.train_behavior("Walk", morphology_name="fox",
                               behavior_name="walking")

        If only ``object_name`` is given, both LMs use it (legacy mode).

        Args:
            anim_name: Animation name in the glTF model.
            object_name: Shared name for both LMs (legacy). Ignored when
                ``morphology_name`` and ``behavior_name`` are provided.
            n_repetitions: Number of full animation passes.
            morphology_name: Object ID for the morphology LM (LM 0).
            behavior_name: Object ID for the behavior LM (LM 1).

        Returns:
            Dict with ``total_steps`` and per-LM stats.
        """
        # Resolve names: split takes precedence over shared
        if morphology_name is None and behavior_name is None:
            if object_name is None:
                raise ValueError(
                    "Provide object_name or morphology_name + behavior_name"
                )
            morphology_name = object_name
            behavior_name = object_name
        if self._sim is None:
            self._setup()

        n_frames = self._anim_obj.get_num_frames(anim_name)
        total_frames = n_frames * n_repetitions
        ctx = RuntimeContext(rng=np.random.RandomState(42))

        target = {
            "object": morphology_name,
            "quat_rotation": [1, 0, 0, 0],
        }

        # Set experiment mode (propagates to each LM via set_experiment_mode)
        self._monty.set_experiment_mode(ExperimentMode.TRAIN)
        self._monty.pre_episode(primary_target=target)

        # Set stepwise_target_object on EvidenceGraphLMs so that
        # GraphLM.get_output() returns high-confidence States with the
        # known target identity. This enables HPC to learn associations.
        morph_lm = self._monty.learning_modules[0]
        behav_lm = self._monty.learning_modules[1]
        morph_lm.stepwise_target_object = morphology_name
        behav_lm.stepwise_target_object = behavior_name

        # Force exploratory mode for training (skip initial matching step)
        self._monty.switch_to_exploratory_step()
        for sm in self._monty.sensor_modules:
            sm.is_exploring = True

        # Position camera at initial distance, looking at posed model center
        self._position_camera_initial(anim_name=anim_name, frame=0)

        # Sensorimotor loop: actions from previous step feed into next
        actions = []
        step = 0
        for _rep in range(n_repetitions):
            for frame in range(n_frames):
                obs = self._step_environment(actions, frame, anim_name)
                actions = self._monty.step(ctx, obs)
                # Restore stepwise_target_object after each step.
                # _set_stepwise_targets() overwrites it to "no_label" since
                # Panda3D observations lack semantic IDs. We re-set it so
                # that get_output() (called at the start of the next step)
                # returns the correct training target to HPC.
                morph_lm.stepwise_target_object = morphology_name
                behav_lm.stepwise_target_object = behavior_name
                step += 1

        # Finalize: set detected object per LM (split naming).
        #
        # Morphology LM: if the shape was already learned (from a
        # previous behavior session), skip — the mesh is the same and
        # observations from a different camera trajectory cannot be
        # aligned without a matching step.
        #
        # Behavior LM: always update.  Flow features are relative
        # (frame-to-frame deltas), so data from multiple morphologies
        # can be accumulated with identity rotation.  This makes
        # the learned behavior more morphology-invariant.
        morph_lm = self._monty.learning_modules[0]
        behav_lm = self._monty.learning_modules[1]

        # -- Morphology LM --
        morph_known = morphology_name in morph_lm.get_all_known_object_ids()
        if morph_known:
            morph_lm.detected_object = None  # skip memory update
        else:
            morph_lm.detected_object = morphology_name
        morph_lm.detected_rotation_r = None
        if morph_lm.buffer.get_num_observations_on_object() > 0:
            morph_lm.buffer.stats["detected_location_rel_body"] = (
                morph_lm.buffer.get_current_location(input_channel="first")
            )
        else:
            morph_lm.buffer.stats["detected_location_rel_body"] = np.zeros(3)

        # -- Behavior LM: always update (extend if exists) --
        behav_lm.detected_object = behavior_name
        behav_lm.detected_rotation_r = R.identity()
        if behav_lm.buffer.get_num_observations_on_object() > 0:
            loc = behav_lm.buffer.get_current_location(input_channel="first")
            behav_lm.buffer.stats["detected_location_rel_body"] = loc
            behav_lm.buffer.stats["detected_location_on_model"] = loc
        else:
            behav_lm.buffer.stats["detected_location_rel_body"] = np.zeros(3)
            behav_lm.buffer.stats["detected_location_on_model"] = np.zeros(3)

        self._monty.post_episode()

        result = {"total_steps": step}
        for lm in self._monty.learning_modules:
            result[lm.learning_module_id] = {
                "known_objects": lm.get_all_known_object_ids(),
            }

        logger.info(
            "Trained '%s' as morph='%s' behav='%s' in %d steps: %s",
            anim_name, morphology_name, behavior_name, step, result,
        )
        return result

    def match_behavior(self, anim_name, n_steps=None):
        """Match observed behavior against learned models.

        Runs the sensorimotor loop: the motor policy explores while
        LMs accumulate evidence. Lateral voting between morphology
        and behavior LMs refines the match.

        Returns:
            Dict with per-LM MLH and combined results.
        """
        if self._sim is None:
            self._setup()

        n_frames = self._anim_obj.get_num_frames(anim_name)
        if n_steps is not None:
            n_frames = min(n_frames, n_steps)

        ctx = RuntimeContext(rng=np.random.RandomState(42))

        target = {
            "object": "placeholder",
            "quat_rotation": [1, 0, 0, 0],
        }

        self._monty.set_experiment_mode(ExperimentMode.EVAL)
        self._monty.pre_episode(primary_target=target)

        # Position camera at initial distance, looking at posed model center
        self._position_camera_initial(anim_name=anim_name, frame=0)

        # Sensorimotor loop
        actions = []
        for frame in range(n_frames):
            obs = self._step_environment(actions, frame, anim_name)
            actions = self._monty.step(ctx, obs)

        # Collect MLH from EvidenceGraphLMs (skip HPC which has no MLH)
        result = {}
        for lm in self._monty.learning_modules:
            if not hasattr(lm, "get_current_mlh"):
                continue
            mlh = dict(lm.get_current_mlh())
            result[lm.learning_module_id] = mlh

        # Per-LM top-level convenience keys
        morph_lm = self._monty.learning_modules[0]
        behav_lm = self._monty.learning_modules[1]

        result["morphology_id"] = morph_lm.get_current_mlh().get("graph_id")
        result["behavior_id"] = behav_lm.get_current_mlh().get("graph_id")
        # Legacy alias
        result["graph_id"] = result["behavior_id"]

        result["morphology_evidence"] = morph_lm.evidence
        result["behavior_evidence"] = behav_lm.evidence

        # Temporal memory results (if enabled on behavior LM)
        if hasattr(behav_lm, "_temporal_memory") and behav_lm._temporal_memory:
            tm = behav_lm._temporal_memory
            result["temporal_surprise_history"] = tm.get_surprise_history()
            result["temporal_mean_surprise"] = tm.get_mean_surprise()
            # Chain predictions forward from last observed SDR
            predictions = []
            current = tm._prev_sdr
            for _ in range(5):
                p = tm.predict_next(from_sdr=current)
                if p is None:
                    break
                predictions.append(p)
                current = p
            result["temporal_predictions"] = predictions

        return result

    def get_animation_names(self):
        """Return available animation names from the loaded model."""
        if self._sim is None:
            self._setup()
        return list(self._anim_obj.animation_names)

    @property
    def monty(self):
        """Access the MontyForEvidenceGraphMatching instance."""
        if self._monty is None:
            raise RuntimeError(
                "Not initialized — call train_behavior() first"
            )
        return self._monty

    @property
    def morphology_lm(self):
        """Access the morphology EvidenceGraphLM (LM 0)."""
        return self.monty.learning_modules[0]

    @property
    def behavior_lm(self):
        """Access the behavior EvidenceGraphLM (LM 1)."""
        return self.monty.learning_modules[1]

    @property
    def hpc(self):
        """Access the HippocampalModule (LM 2), if configured."""
        if self._hpc_kwargs is None:
            raise RuntimeError("HPC not configured — pass hpc_kwargs to __init__")
        return self.monty.learning_modules[2]

    def swap_model(self, model_path, object_scale=None):
        """Replace the current 3D model without rebuilding the Monty system.

        Keeps all learned graphs in both LMs intact.  Useful for
        cross-morphology training: train on model A, swap to model B,
        train more, then match against either.

        Args:
            model_path: Path to a new glTF/GLB file.
            object_scale: Optional new scale; defaults to the original.
        """
        if self._sim is None:
            self._setup()

        self._model_path = Path(model_path)
        if object_scale is not None:
            self._object_scale = object_scale

        self._sim.remove_all_objects()
        info = self._sim.add_object(
            name=str(self._model_path),
            position=self._object_position,
            scale=self._object_scale,
            animated=True,
        )
        self._obj_id = info.object_id
        self._anim_obj = self._sim.get_animated_object(self._obj_id)

        logger.info(
            "Swapped model to %s, animations=%s",
            self._model_path.name, self._anim_obj.animation_names,
        )

    def close(self):
        """Clean up Panda3D resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ======================== Private ========================

    def _position_camera_initial(self, anim_name=None, frame=0):
        """Set the camera to the initial position, looking at the object.

        Positions the camera at ``initial_distance`` along -Y, aimed at the
        object's visual center. For animated models, the bounding box is
        computed AFTER posing to the first frame, since skeletal deformation
        can shift the visual center significantly from the rest pose.
        """
        from panda3d.core import LVector3f
        cam_np = self._sim._agent_buffers[self._agent_id]["camera_np"]

        # Pose the model first so the bbox reflects deformed geometry
        if anim_name is not None:
            self._anim_obj.pose(frame, anim_name)

        # Compute visual center from posed bounding box
        obj_np = self._sim._objects[self._obj_id]
        bounds = obj_np.getTightBounds()
        if bounds:
            lo, hi = bounds
            center_x = float((lo[0] + hi[0]) / 2)
            center_y = float((lo[1] + hi[1]) / 2)
            center_z = float((lo[2] + hi[2]) / 2)
        else:
            center_x = self._object_position[0]
            center_y = self._object_position[1]
            center_z = self._object_position[2]

        # Position camera along -Y from object center, at same height
        cam_np.setPos(center_x, center_y - self._initial_distance, center_z)
        cam_np.lookAt(LVector3f(center_x, center_y, center_z))

        # Sync motor system state from simulator
        self._sync_motor_state()

    def _sync_motor_state(self):
        """Read current camera pose from Panda3D and update motor system state.

        This keeps CameraSM.update_state() and the motor policy in sync
        with the actual camera position after actions are executed.

        Uses exact float64 values stored by actuate_set_agent_pose when
        available, since Panda3D's internal float32 representation loses
        precision on quaternion round-trips.
        """
        buf = self._sim._agent_buffers[self._agent_id]
        cam_np = buf["camera_np"]

        # Prefer exact values stored by SetAgentPose (float64-precise)
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

        sensor_state = SensorState(
            position=(0.0, 0.0, 0.0),
            rotation=rot_q,
        )
        agent_state = AgentState(
            position=position,
            rotation=rot_q,
            sensors={
                SensorID(self.CAMERA_SM_ID): sensor_state,
                SensorID(self.CHANGE_SM_ID): sensor_state,
            },
        )
        self._monty.motor_system._state = MotorSystemState(
            {self._agent_id: agent_state}
        )

    def _step_environment(self, actions, frame, anim_name):
        """Execute actions, advance animation, render, transform, return obs.

        This is the environment side of the sensorimotor loop:
        1. Execute motor actions in Panda3D (moves camera)
        2. Advance animation to the given frame
        3. Render and transform to semantic_3d
        4. Duplicate observations under both SM keys

        Args:
            actions: List of Action objects from monty.step()
            frame: Animation frame to pose
            anim_name: Animation name

        Returns:
            Observations dict keyed by agent and SM IDs.
        """
        # 1. Execute motor actions (camera moves)
        for action in actions:
            action.act(self._sim)

        # 2. Advance animation
        self._anim_obj.pose(frame, anim_name)

        # 3. Render (sim.step with no additional actions)
        raw_obs, proprio = self._sim.step([])

        # 4. Sync motor system state from actual camera position
        self._sync_motor_state()

        # 5. Transform depth → semantic_3d
        transform_ctx = TransformContext(
            rng=np.random.RandomState(0), state=proprio,
        )
        raw_obs = self._depth_transform(raw_obs, transform_ctx)
        raw_obs = self._d3d_transform(raw_obs, transform_ctx)

        # 6. Duplicate sensor data under both SM keys.
        # Also add "view_finder" alias — required by InformedPolicy's
        # _should_undo_jump() for depth-at-center checks during
        # goal-state-driven exploration.
        sensor_data = raw_obs[self._agent_id][self._panda3d_sensor_id]
        obs = Observations({
            self._agent_id: {
                self.CAMERA_SM_ID: sensor_data,
                self.CHANGE_SM_ID: sensor_data,
                "view_finder": sensor_data,
            }
        })

        return obs
