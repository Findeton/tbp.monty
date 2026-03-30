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
        hopfield_voting=False,
        hopfield_surprise_threshold=0.3,
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
        self._hopfield_voting = hopfield_voting
        self._hopfield_surprise_threshold = hopfield_surprise_threshold

        self._agent_id = AgentID("agent_id_0")
        self._panda3d_sensor_id = SensorID("sensor_0")

        self._sim = None
        self._obj_id = None
        self._anim_obj = None
        self._depth_transform = None
        self._d3d_transform = None
        self._monty = None

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
            seed=42,
        )
        col_kw.update(self._column_kwargs)

        lm = CorticalColumnTorchLM(
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
            hopfield_voting=self._hopfield_voting,
            hopfield_surprise_threshold=self._hopfield_surprise_threshold,
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
            seed=42,
        )
        child_kw.update(self._column_kwargs)

        lm_morph = CorticalColumnTorchLM(
            column_kwargs=child_kw,
            learning_module_id="lm_morphology",
        )
        lm_behav = CorticalColumnTorchLM(
            column_kwargs={**child_kw, "seed": 43},
            learning_module_id="lm_behavior",
        )

        parent_kw = dict(
            n_minicolumns=2048,
            n_cells_per_minicolumn=8,
            sparsity=0.03,
            use_apical=True,
            seed=44,
        )
        parent_kw.update(self._parent_column_kwargs)

        lm_parent = CorticalColumnTorchLM(
            column_kwargs=parent_kw,
            learning_module_id="lm_parent",
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
            hopfield_voting=self._hopfield_voting,
            hopfield_surprise_threshold=self._hopfield_surprise_threshold,
        )

    def _build_motor_system(self):
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

        Returns dict with training results per LM.
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

        for lm in self._monty.learning_modules:
            lm.stepwise_target_object = object_name

        self._monty.switch_to_exploratory_step()
        for sm in self._monty.sensor_modules:
            sm.is_exploring = True

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

            for lm in self._monty.learning_modules:
                lm.stepwise_target_object = object_name

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

        Returns dict with per-LM MLH and evidence.
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

        result = {}
        for lm in self._monty.learning_modules:
            if hasattr(lm, "get_current_mlh"):
                mlh = dict(lm.get_current_mlh())
                result[lm.learning_module_id] = mlh

        primary_lm = self._monty.learning_modules[0]
        result["graph_id"] = primary_lm.get_current_mlh().get("graph_id")
        result["evidence"] = {
            lm.learning_module_id: dict(lm.evidence)
            for lm in self._monty.learning_modules
        }

        logger.info("Evaluate result: %s", result.get("graph_id"))
        return result

    def swap_model(self, model_path, object_scale=None):
        """Replace the 3D model without rebuilding Monty."""
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
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ======================== Private ========================

    def _position_camera_initial(self, anim_name=None, frame=None):
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
