# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Sensorimotor temporal training on animated Panda3D objects.

Orchestrates the full sensorimotor loop:

1. **Motor**: Agent takes an action (moves/rotates around the object)
2. **Animate**: Object advances one animation frame (deformation)
3. **Sense**: Panda3D renders the scene → RGBA + depth
4. **Transform**: ``Panda3DDepthNormalize`` + ``DepthTo3DLocations`` → 3D point cloud
5. **Process**: ``CameraSM`` extracts features → ``State`` (location, normals,
   curvatures, HSV)
6. **Learn**: ``TemporalMemory`` encodes the State as an SDR, predicts from the
   previous SDR, computes surprise, and learns the transition via Hebbian update

The agent's motor actions and the object's animation both contribute to changing
what Monty observes — TemporalMemory learns the joint sensorimotor temporal pattern.

Usage::

    trainer = Panda3DTemporalTrainer(
        model_path="arm.glb",
        output_dir="./debug_output",   # enables observability
    )
    results = trainer.train_episode(n_repetitions=3)
    print(results["mean_surprise_final"])
    trainer.close()
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.models.sensor_modules import CameraSM
from tbp.monty.frameworks.models.temporal_memory import TemporalMemory
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.debug_output import TemporalTrainingDebugger
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator
from tbp.monty.simulators.panda3d.transforms import Panda3DDepthNormalize

logger = logging.getLogger(__name__)


class OrbitalMotorPolicy:
    """Simple deterministic motor policy that orbits the camera around origin.

    Moves the camera along a circular path at a fixed distance from the
    origin, changing elevation periodically. This provides systematic
    sensorimotor exploration of the animated object from multiple viewpoints.

    Parameters
    ----------
    agent_id : AgentID
        Agent to control.
    orbit_radius : float
        Distance from the origin.
    azimuth_step_deg : float
        Degrees to rotate horizontally per step.
    elevation_range_deg : tuple[float, float]
        (min, max) elevation angles in degrees.
    elevation_steps : int
        Number of steps per elevation cycle.
    """

    def __init__(
        self,
        agent_id: AgentID = AgentID("agent_id_0"),
        orbit_radius: float = 0.5,
        azimuth_step_deg: float = 15.0,
        elevation_range_deg: tuple = (-30.0, 30.0),
        elevation_steps: int = 12,
    ):
        self.agent_id = agent_id
        self.orbit_radius = orbit_radius
        self.azimuth_step_deg = azimuth_step_deg
        self.elevation_range_deg = elevation_range_deg
        self.elevation_steps = elevation_steps
        self._step = 0

    def get_camera_pose(self, step: int = None):
        """Compute camera position and look-at-origin rotation for the given step.

        Returns (position, rotation_wxyz) tuple.
        """
        if step is None:
            step = self._step

        azimuth_rad = math.radians(step * self.azimuth_step_deg)

        # Oscillate elevation within range
        el_min, el_max = self.elevation_range_deg
        el_range = el_max - el_min
        t = (step % self.elevation_steps) / max(self.elevation_steps - 1, 1)
        elevation_deg = el_min + t * el_range
        elevation_rad = math.radians(elevation_deg)

        # Spherical to cartesian (Y-forward convention for Panda3D)
        x = self.orbit_radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
        y = -self.orbit_radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
        z = self.orbit_radius * math.sin(elevation_rad)
        position = (x, y, z)

        # Look-at-origin rotation
        rotation_wxyz = _look_at_origin(position)

        self._step = step + 1
        return position, rotation_wxyz

    def reset(self):
        self._step = 0


class RandomMotorPolicy:
    """Random motor policy that samples movement actions each step.

    Parameters
    ----------
    agent_id : AgentID
        Agent to control.
    move_distance : float
        Distance for forward/tangential movement.
    rotation_degrees : float
        Degrees for turning/looking.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        agent_id: AgentID = AgentID("agent_id_0"),
        move_distance: float = 0.02,
        rotation_degrees: float = 5.0,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.move_distance = move_distance
        self.rotation_degrees = rotation_degrees
        self._rng = np.random.RandomState(seed)

    def sample_action(self) -> Action:
        """Sample a random motor action."""
        action_type = self._rng.choice(5)
        if action_type == 0:
            return MoveForward(self.agent_id, distance=self.move_distance)
        elif action_type == 1:
            return TurnLeft(self.agent_id, rotation_degrees=self.rotation_degrees)
        elif action_type == 2:
            return TurnRight(self.agent_id, rotation_degrees=self.rotation_degrees)
        elif action_type == 3:
            return LookUp(self.agent_id, rotation_degrees=self.rotation_degrees)
        else:
            return LookDown(self.agent_id, rotation_degrees=self.rotation_degrees)

    def reset(self):
        pass


class Panda3DTemporalTrainer:
    """Trains TemporalMemory on animated Panda3D objects with sensorimotor exploration.

    Implements the full sensorimotor loop: the agent actively moves around
    an animated object, observations are transformed through the Monty pipeline
    to produce State objects, and TemporalMemory learns the joint temporal
    patterns arising from both agent movement and object deformation.

    Parameters
    ----------
    model_path : str or Path
        Path to a glTF/GLB file with skeletal animation.
    resolution : tuple[int, int]
        Render resolution (height, width).
    fov : float
        Camera horizontal field of view in degrees.
    near : float
        Near clip plane distance.
    far : float
        Far clip plane distance.
    motor_policy : str
        Motor policy type: ``"orbital"`` for deterministic orbit,
        ``"random"`` for random exploration.
    orbit_radius : float
        Distance from origin for orbital policy.
    object_position : tuple
        Position of the animated object in the scene.
    object_scale : tuple
        Scale of the animated object.
    anim_name : str or None
        Name of the animation to play. If None, uses the first animation
        found in the model. For models with multiple animations (e.g. Fox
        has "Survey", "Walk", "Run"), specify which one to train on.
    sm_features : list[str]
        Features for CameraSM to extract.
    tm_kwargs : dict or None
        Extra keyword arguments for TemporalMemory.
    output_dir : str, Path, or None
        If set, enables debug output (frames, states, surprise curve, video).
    """

    def __init__(
        self,
        model_path: str | Path,
        resolution: tuple = (64, 64),
        fov: float = 90.0,
        near: float = 0.01,
        far: float = 10.0,
        motor_policy: str = "orbital",
        orbit_radius: float = 0.5,
        object_position: tuple = (0.0, 0.0, 0.0),
        object_scale: tuple = (1.0, 1.0, 1.0),
        anim_name: str | None = None,
        sm_features: list | None = None,
        tm_kwargs: dict | None = None,
        output_dir: str | Path | None = None,
    ):
        self._model_path = Path(model_path)
        self._anim_name = anim_name
        self._resolution = resolution
        self._fov = fov
        self._near = near
        self._far = far
        self._object_position = object_position
        self._object_scale = object_scale

        # Defaults
        if sm_features is None:
            sm_features = [
                "on_object",
                "hsv",
                "principal_curvatures_log",
            ]
        self._sm_features = sm_features
        self._tm_kwargs = tm_kwargs or {}

        # Agent ID/Sensor ID
        self._agent_id = AgentID("agent_id_0")
        self._sensor_id = SensorID("patch")

        # Motor policy
        if motor_policy == "orbital":
            self._motor = OrbitalMotorPolicy(
                agent_id=self._agent_id,
                orbit_radius=orbit_radius,
            )
        elif motor_policy == "random":
            self._motor = RandomMotorPolicy(agent_id=self._agent_id)
        else:
            raise ValueError(f"Unknown motor_policy: {motor_policy}")

        # Debug output
        self._debugger = None
        if output_dir is not None:
            self._debugger = TemporalTrainingDebugger(output_dir)

        # Components (initialized lazily in _setup)
        self._sim = None
        self._obj_id = None
        self._anim_obj = None
        self._n_frames = 0
        self._depth_transform = None
        self._d3d_transform = None
        self._camera_sm = None
        self._tm = None

    def _setup(self):
        """Initialize simulator, pipeline components, and temporal memory."""
        from tbp.monty.frameworks.environment_utils.transforms import DepthTo3DLocations

        # Create agent and simulator
        agent = Panda3DAgent(
            agent_id=self._agent_id,
            sensor_id=self._sensor_id,
            resolution=self._resolution,
            fov=self._fov,
        )
        self._sim = Panda3DSimulator(
            agents=[agent],
            near=self._near,
            far=self._far,
        )

        # Add animated object
        info = self._sim.add_object(
            name=str(self._model_path),
            position=self._object_position,
            scale=self._object_scale,
            animated=True,
        )
        self._obj_id = info.object_id
        self._anim_obj = self._sim.get_animated_object(self._obj_id)

        # Resolve animation name
        if self._anim_name is None:
            self._anim_name = self._anim_obj.animation_names[0]
        elif self._anim_name not in self._anim_obj.animation_names:
            raise ValueError(
                f"Animation '{self._anim_name}' not found. "
                f"Available: {self._anim_obj.animation_names}"
            )
        self._n_frames = self._anim_obj.get_num_frames(self._anim_name)

        # Transforms
        self._depth_transform = Panda3DDepthNormalize(
            agent_id=self._agent_id,
            near=self._near,
            far=self._far,
        )
        self._d3d_transform = DepthTo3DLocations(
            agent_id=self._agent_id,
            sensor_ids=[self._sensor_id],
            resolutions=[self._resolution],
            hfov=self._fov,
            world_coord=True,
            get_all_points=True,
        )

        # Sensor module
        self._camera_sm = CameraSM(
            sensor_module_id=str(self._sensor_id),
            features=self._sm_features,
        )
        self._camera_sm.pre_episode()

        # Temporal memory
        self._tm = TemporalMemory(**self._tm_kwargs)

        logger.info(
            "Panda3DTemporalTrainer setup: model=%s, anim=%s, "
            "%d animation frames, resolution=%s, motor=%s",
            self._model_path.name,
            self._anim_name,
            self._n_frames,
            self._resolution,
            type(self._motor).__name__,
        )

    def train_episode(
        self,
        n_repetitions: int = 1,
        n_replay: int = 2,
    ) -> Dict[str, Any]:
        """Run one training episode: loop through animation with sensorimotor exploration.

        Each step:
        1. Motor policy determines camera pose (sensorimotor exploration)
        2. Animation advances one frame (object deformation)
        3. Scene is rendered → RGBA + depth
        4. Transforms produce 3D point cloud
        5. CameraSM extracts features → State
        6. TemporalMemory encodes, predicts, learns

        Parameters
        ----------
        n_repetitions : int
            Number of full passes through the animation.
        n_replay : int
            Number of offline replay passes after training.

        Returns
        -------
        dict with training results:
            mean_surprise_first : float — mean surprise on first pass
            mean_surprise_final : float — mean surprise on final pass
            total_steps : int — total training steps
            usable_states : int — steps where State.use_state was True
            surprise_history : list[float] — all surprise values
        """
        if self._sim is None:
            self._setup()

        self._tm.reset_episode()
        if hasattr(self._motor, "reset"):
            self._motor.reset()

        total_steps = self._n_frames * n_repetitions
        step = 0
        usable_states = 0
        surprise_first_pass = []
        surprise_final_pass = []

        for rep in range(n_repetitions):
            for frame in range(self._n_frames):
                # 1. Motor: position camera
                action_name = self._apply_motor(step)

                # 2. Animate: advance to this frame
                self._anim_obj.pose(frame, self._anim_name)

                # 3. Sense: render
                obs, proprio = self._sim.step([])

                # 4. Transform: depth normalize + depth-to-3D
                obs = self._apply_transforms(obs, proprio)

                # 5. Process: CameraSM → State
                state = self._extract_state(obs, proprio)

                # 6. Learn: TemporalMemory step
                surprise = None
                if state is not None and state.use_state:
                    tm_result = self._tm.step(state, learn=True)
                    surprise = tm_result["surprise"]
                    usable_states += 1

                    if rep == 0:
                        surprise_first_pass.append(surprise)
                    if rep == n_repetitions - 1:
                        surprise_final_pass.append(surprise)

                # Debug output
                if self._debugger is not None:
                    raw_obs = obs[self._agent_id][self._sensor_id]
                    self._debugger.save_frame(
                        step=step,
                        rgba=raw_obs["rgba"],
                        depth=raw_obs["depth"],
                        state=state,
                        surprise=surprise,
                        action_name=action_name,
                    )

                step += 1

        # Replay for consolidation
        if n_replay > 0:
            self._tm.replay_episode(n_replays=n_replay)

        # Compute results
        results = {
            "mean_surprise_first": (
                float(np.mean(surprise_first_pass)) if surprise_first_pass else None
            ),
            "mean_surprise_final": (
                float(np.mean(surprise_final_pass)) if surprise_final_pass else None
            ),
            "total_steps": step,
            "usable_states": usable_states,
            "n_animation_frames": self._n_frames,
            "n_repetitions": n_repetitions,
            "surprise_history": self._tm.get_surprise_history(),
        }

        # Save debug summary
        if self._debugger is not None:
            self._debugger.save_surprise_curve()
            self._debugger.save_summary(extra=results)
            video_path = self._debugger.make_video(fps=10)
            if video_path:
                results["video_path"] = str(video_path)

        logger.info(
            "Training complete: %d steps, %d usable states, "
            "surprise first=%.3f final=%.3f",
            step,
            usable_states,
            results["mean_surprise_first"] or 0,
            results["mean_surprise_final"] or 0,
        )

        return results

    def _apply_motor(self, step: int) -> str:
        """Apply motor policy to position the camera."""
        if isinstance(self._motor, OrbitalMotorPolicy):
            position, rotation = self._motor.get_camera_pose(step)
            # Set camera pose directly via simulator
            info = self._sim._agent_buffers[self._agent_id]
            cam_np = info["camera_np"]
            self._sim._set_node_pose(cam_np, position, rotation)
            return f"orbital_step_{step}"
        elif isinstance(self._motor, RandomMotorPolicy):
            action = self._motor.sample_action()
            action.act(self._sim)
            return action.name
        return "unknown"

    def _apply_transforms(self, obs, proprio):
        """Apply depth normalization and DepthTo3DLocations."""
        ctx = TransformContext(rng=np.random.RandomState(0), state=proprio)
        obs = self._depth_transform(obs, ctx)
        obs = self._d3d_transform(obs, ctx)
        return obs

    def _extract_state(self, obs, proprio):
        """Run CameraSM to produce a State from transformed observations."""
        agent_state = proprio[self._agent_id]
        self._camera_sm.update_state(agent_state)
        sensor_obs = obs[self._agent_id][self._sensor_id]
        from tbp.monty.context import RuntimeContext
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        return self._camera_sm.step(ctx, sensor_obs)

    @property
    def temporal_memory(self) -> Optional[TemporalMemory]:
        return self._tm

    @property
    def n_animation_frames(self) -> int:
        return self._n_frames

    def close(self):
        """Clean up resources."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None


def _look_at_origin(position):
    """Compute a quaternion rotation that looks from ``position`` toward origin.

    Returns (w, x, y, z) quaternion compatible with Panda3D.
    """
    from scipy.spatial.transform import Rotation

    pos = np.array(position, dtype=np.float64)
    forward = -pos / (np.linalg.norm(pos) + 1e-12)

    # Choose up vector (avoid degeneracy when forward is near vertical)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-12
    up = np.cross(right, forward)

    # Panda3D convention: Y is forward, Z is up, X is right
    # Build rotation matrix: columns are the axes in world space
    mat = np.eye(3)
    mat[:, 0] = right
    mat[:, 1] = forward
    mat[:, 2] = up

    r = Rotation.from_matrix(mat)
    q = r.as_quat()  # returns (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))  # (w, x, y, z)
