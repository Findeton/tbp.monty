# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""CorticalColumnLM experiment with real Panda3D rendering and 3D models.

Mirrors the structure of ``Panda3DBehaviorExperiment`` but uses
CorticalColumnLM as the learning module. Two architectures are supported:

**Flat (single column per SM, lateral voting):**

    SM 0: CameraSM → LM 0 (CorticalColumnLM)
    SM 1: CameraSM → LM 1 (CorticalColumnLM)  [optional second patch]

    sm_to_lm_matrix:       [[0], [1]]
    lm_to_lm_vote_matrix:  [[1], [0]]

**Hierarchical (two sensory columns + one parent column with apical):**

    SM 0: CameraSM          → LM 0 (CorticalColumnLM, use_apical=True)
    SM 1: ChangeDetectingSM → LM 1 (CorticalColumnLM, use_apical=True)
                               LM 2 (CorticalColumnLM, parent, use_apical=True)
                                      ← receives output from LM 0 + LM 1

    sm_to_lm_matrix:       [[0], [1], []]
    lm_to_lm_matrix:       [[], [], [0, 1]]
    lm_to_lm_vote_matrix:  [[1], [0], []]

Apical dendrites carry top-down context: after each step, LM 2's
``get_context_signal()`` broadcasts its active cell pattern to LM 0
and LM 1 via ``_dispatch_context_signals()``.  On the next step, the
child LMs' apical dendrites bias winner cell selection toward cells
predicted by the parent's context — implementing real cortical
heterarchy.

Motor system: InformedPolicy drives camera exploration via standard
Monty actions (MoveForward, TurnLeft, LookUp, etc.) executed by
Panda3D actuators. For animated models, the animation advances each
step independently of motor actions.

Usage::

    exp = Panda3DCorticalColumnExperiment(
        model_path="Fox.glb",
        asset_search_paths=["/path/to/assets"],
        object_scale=(0.01, 0.01, 0.01),
        hierarchical=True,     # enables 3-LM heterarchy
    )
    exp.train("fox_walk", anim_name="Walk")  # animated training
    exp.train("fox_static")                   # frozen-frame training
    mlh = exp.evaluate(anim_name="Walk")
    print(mlh["graph_id"])
    exp.close()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import quaternion

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import ConstantSampler
from tbp.monty.frameworks.actions.actions import (
    LookDown,
    LookUp,
    MoveForward,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.change_detecting_sm import ChangeDetectingSM
from tbp.monty.frameworks.models.cortical_column import CorticalColumnLM
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


class Panda3DCorticalColumnExperiment:
    """Train and evaluate CorticalColumnLMs with real Panda3D rendering.

    Parameters
    ----------
    model_path : str or Path
        Path to a glTF/GLB 3D model file with skeletal animation.
    hierarchical : bool
        If True, builds a 3-LM heterarchy (2 sensory + 1 parent) with
        apical dendrites. If False, uses a flat 2-LM mutual voting setup.
    resolution : tuple
        Render resolution (height, width).
    fov : float
        Horizontal field of view in degrees.
    near, far : float
        Near and far clip planes.
    initial_distance : float
        Starting camera distance from object center.
    object_position, object_scale : tuple
        Object placement and scale in the scene.
    column_kwargs : dict or None
        Base kwargs for CorticalColumn. ``use_apical`` is forced True
        for hierarchical mode.
    parent_column_kwargs : dict or None
        Kwargs for the parent LM's CorticalColumn (hierarchical only).
    camera_features : list or None
        Features for CameraSM extraction.
    flow_threshold : float
        Flow threshold for ChangeDetectingSM (hierarchical mode).
    asset_search_paths : list or None
        Additional directories for model file resolution.
    rotation_degrees : float
        Rotation step size for motor policy.
    translation_distance : float
        Translation step size for motor policy.
    """

    CAMERA_SM_ID = "patch_0"
    CHANGE_SM_ID = "change_detector"

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
        parent_column_kwargs=None,
        camera_features=None,
        flow_threshold=0.005,
        asset_search_paths=None,
        rotation_degrees=5.0,
        translation_distance=0.004,
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
        self._parent_column_kwargs = parent_column_kwargs or {}
        self._camera_features = camera_features or [
            "pose_vectors", "on_object", "hsv",
        ]
        self._flow_threshold = flow_threshold
        self._asset_search_paths = asset_search_paths or []
        self._rotation_degrees = rotation_degrees
        self._translation_distance = translation_distance

        self._agent_id = AgentID("agent_id_0")
        self._panda3d_sensor_id = SensorID("sensor_0")

        # Lazily initialized
        self._sim = None
        self._obj_id = None
        self._anim_obj = None
        self._depth_transform = None
        self._d3d_transform = None
        self._monty = None

    # ======================== Setup ========================

    def _setup(self):
        """Initialize Panda3D, sensor modules, CorticalColumnLMs, and Monty."""
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
            "Panda3DCorticalColumnExperiment: model=%s, hierarchical=%s, "
            "animations=%s, n_lms=%d",
            self._model_path.name,
            self._hierarchical,
            self._anim_obj.animation_names,
            len(self._monty.learning_modules),
        )

    def _setup_flat(self):
        """2-LM flat architecture: 1 CameraSM → 1 CorticalColumnLM."""
        sm = CameraSM(
            sensor_module_id=self.CAMERA_SM_ID,
            features=self._camera_features,
        )

        col_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=False,
            seed=42,
        )
        col_kw.update(self._column_kwargs)

        lm = CorticalColumnLM(
            column_kwargs=col_kw,
            learning_module_id="lm_0",
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
        )

    def _setup_hierarchical(self):
        """3-LM heterarchy: 2 sensory columns + 1 parent with apical."""
        sm_camera = CameraSM(
            sensor_module_id=self.CAMERA_SM_ID,
            features=self._camera_features,
        )
        sm_change = ChangeDetectingSM(
            sensor_module_id=self.CHANGE_SM_ID,
            flow_threshold=self._flow_threshold,
            global_flow_suppression=False,
        )

        # Child columns: use_apical=True so they receive top-down context
        child_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=True,
            seed=42,
        )
        child_kw.update(self._column_kwargs)

        lm_morph = CorticalColumnLM(
            column_kwargs=child_kw,
            learning_module_id="lm_morphology",
        )
        lm_behav = CorticalColumnLM(
            column_kwargs={**child_kw, "seed": 43},
            learning_module_id="lm_behavior",
        )

        # Parent column: receives combined outputs from children
        parent_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=True,
            seed=44,
        )
        parent_kw.update(self._parent_column_kwargs)

        lm_parent = CorticalColumnLM(
            column_kwargs=parent_kw,
            learning_module_id="lm_parent",
        )

        motor_system = self._build_motor_system()

        # Heterarchy wiring:
        # SM0→LM0, SM1→LM1, LM2 has no direct SM
        # LM2 receives LM-to-LM input from LM0 and LM1
        # Lateral voting: LM0↔LM1 (parent doesn't vote spatially)
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
        )

    def _build_motor_system(self):
        """Create motor system with InformedPolicy."""
        action_sampler = ConstantSampler(
            actions=[MoveForward, TurnLeft, TurnRight, LookUp, LookDown],
            rotation_degrees=self._rotation_degrees,
            translation_distance=self._translation_distance,
        )
        motor_policy = InformedPolicy(
            action_sampler=action_sampler,
            agent_id=self._agent_id,
            use_goal_state_driven_actions=False,
        )
        return MotorSystem(policy=motor_policy)

    # ======================== Public API ========================

    def train(self, object_name, anim_name=None, n_steps=None):
        """Train LMs on the loaded 3D model.

        Args:
            object_name: Name to store the learned representation under.
            anim_name: If provided, play this animation while training
                (temporal/behavior features). If None, trains on a frozen
                first frame (static morphology).
            n_steps: Number of sensorimotor steps to train for.
                Defaults to the animation length or 30 for static.

        Returns:
            Dict with training results per LM.
        """
        if self._sim is None:
            self._setup()

        animated = anim_name is not None
        if animated:
            n_frames = self._anim_obj.get_num_frames(anim_name)
            total_steps = n_steps or n_frames
        else:
            total_steps = n_steps or 30

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        target = {"object": object_name, "quat_rotation": [1, 0, 0, 0]}

        self._monty.set_experiment_mode(ExperimentMode.TRAIN)
        self._monty.pre_episode(primary_target=target)

        # Set stepwise target on all LMs
        for lm in self._monty.learning_modules:
            lm.stepwise_target_object = object_name

        self._monty.switch_to_exploratory_step()
        for sm in self._monty.sensor_modules:
            sm.is_exploring = True

        # Position camera
        self._position_camera_initial(
            anim_name=anim_name, frame=0 if animated else None,
        )

        # Sensorimotor loop
        actions = []
        for step in range(total_steps):
            frame = step % n_frames if animated else 0
            obs = self._step_environment(
                actions, frame=frame, anim_name=anim_name,
            )
            actions = self._monty.step(ctx, obs)

            # Restore target after _set_stepwise_targets overwrites it
            for lm in self._monty.learning_modules:
                lm.stepwise_target_object = object_name

        # Finalize
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

        result = {"total_steps": total_steps}
        for lm in self._monty.learning_modules:
            result[lm.learning_module_id] = {
                "known_objects": lm.get_all_known_object_ids(),
            }
        logger.info("Trained '%s' in %d steps: %s", object_name, total_steps, result)
        return result

    def evaluate(self, anim_name=None, n_steps=None):
        """Evaluate recognition on the loaded model.

        Args:
            anim_name: If provided, play this animation during eval.
                If None, uses a frozen first frame.
            n_steps: Number of steps. Defaults to animation length or 30.

        Returns:
            Dict with per-LM most-likely-hypothesis and evidence.
        """
        if self._sim is None:
            self._setup()

        animated = anim_name is not None
        if animated:
            n_frames = self._anim_obj.get_num_frames(anim_name)
            total_steps = n_steps or n_frames
        else:
            total_steps = n_steps or 30

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        target = {"object": "placeholder", "quat_rotation": [1, 0, 0, 0]}

        self._monty.set_experiment_mode(ExperimentMode.EVAL)
        self._monty.pre_episode(primary_target=target)

        self._position_camera_initial(
            anim_name=anim_name, frame=0 if animated else None,
        )

        actions = []
        for step in range(total_steps):
            frame = step % n_frames if animated else 0
            obs = self._step_environment(
                actions, frame=frame, anim_name=anim_name,
            )
            actions = self._monty.step(ctx, obs)

        # Collect MLH from all CorticalColumnLMs
        result = {}
        for lm in self._monty.learning_modules:
            if hasattr(lm, "get_current_mlh"):
                mlh = dict(lm.get_current_mlh())
                result[lm.learning_module_id] = mlh

        # Top-level summary
        primary_lm = self._monty.learning_modules[0]
        result["graph_id"] = primary_lm.get_current_mlh().get("graph_id")
        result["evidence"] = {
            lm.learning_module_id: dict(lm.evidence)
            for lm in self._monty.learning_modules
        }

        logger.info("Evaluate result: %s", result.get("graph_id"))
        return result

    def swap_model(self, model_path, object_scale=None):
        """Replace the 3D model without rebuilding the Monty system."""
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

    @property
    def monty(self):
        if self._monty is None:
            raise RuntimeError("Not initialized — call train() first")
        return self._monty

    def close(self):
        """Clean up Panda3D resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ======================== Private ========================

    def _position_camera_initial(self, anim_name=None, frame=None):
        """Position camera at initial_distance, looking at the object."""
        from panda3d.core import LVector3f

        cam_np = self._sim._agent_buffers[self._agent_id]["camera_np"]

        if anim_name is not None and frame is not None:
            self._anim_obj.pose(frame, anim_name)

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
        """Read camera pose from Panda3D and update motor system state."""
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

    def _step_environment(self, actions, frame, anim_name):
        """Execute actions, advance animation/frame, render, return obs.

        For animated models (anim_name is not None), poses the skeleton
        to the given frame. For frozen models, frame=0 with no animation.
        """
        for action in actions:
            action.act(self._sim)

        if anim_name is not None:
            self._anim_obj.pose(frame, anim_name)

        raw_obs, proprio = self._sim.step([])
        self._sync_motor_state()

        transform_ctx = TransformContext(
            rng=np.random.RandomState(0), state=proprio,
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
