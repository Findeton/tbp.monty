# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Panda3D simulator implementing the Simulator protocol.

Provides GPU-accelerated offscreen rendering of 3D scenes with RGBA + depth
output compatible with Monty's observation pipeline. Supports primitive
shapes and glTF/GLB 3D models.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import quaternion
from scipy.spatial.transform import Rotation

# Configure Panda3D BEFORE any panda3d imports
os.environ.setdefault("PANDA_PRC_DIR", "")
os.environ.setdefault("PANDA_PRC_FILENAME", "")

from panda3d.core import (
    AmbientLight,
    Camera,
    DirectionalLight,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    GraphicsPipeSelection,
    LColor,
    LVector3f,
    NodePath,
    PerspectiveLens,
    Texture,
    WindowProperties,
    loadPrcFileData,
)

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    ObjectID,
    ObjectInfo,
    SemanticID,
)
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import QuaternionWXYZ, VectorXYZ
from tbp.monty.simulators.panda3d.animation import AnimatedObject
from tbp.monty.simulators.panda3d.assets import AssetRegistry
from tbp.monty.simulators.panda3d.primitives import PRIMITIVE_GENERATORS

logger = logging.getLogger(__name__)

# Configure Panda3D for offscreen headless rendering
loadPrcFileData("", "window-type none")
loadPrcFileData("", "audio-library-name null")
loadPrcFileData("", "notify-level-display error")


class Panda3DSimulator:
    """GPU-accelerated 3D simulator using Panda3D.

    Implements the Simulator protocol. Renders scenes offscreen to produce
    RGBA + depth observations in the same format that the existing
    DepthTo3DLocations → ObservationProcessor pipeline expects.

    Parameters
    ----------
    agents : list[Panda3DAgent]
        Agent configurations with camera sensors.
    near : float
        Near clip plane distance (meters).
    far : float
        Far clip plane distance (meters).
    """

    def __init__(
        self,
        agents: list,
        near: float = 0.01,
        far: float = 10.0,
        asset_search_paths: list | None = None,
    ):
        from direct.showbase.ShowBase import ShowBase

        self._base = ShowBase()
        self._agents = agents
        self._near = near
        self._far = far
        self._objects: Dict[ObjectID, NodePath] = {}
        self._object_semantic_ids: Dict[ObjectID, Optional[SemanticID]] = {}
        self._next_object_id = 0
        self._next_semantic_id = 1
        self._assets = AssetRegistry(search_paths=asset_search_paths)
        self._animated_objects: Dict[ObjectID, AnimatedObject] = {}

        # Get graphics pipe
        pipe_sel = GraphicsPipeSelection.getGlobalPtr()
        self._pipe = pipe_sel.makeDefaultPipe()
        if self._pipe is None:
            raise RuntimeError("No graphics pipe available for Panda3D rendering")

        # Set up scene lighting
        self._setup_lighting()

        # Set up per-agent rendering
        self._agent_buffers: Dict[AgentID, dict] = {}
        for agent in self._agents:
            self._setup_agent(agent)

    def _setup_lighting(self):
        """Add ambient + directional lighting to the scene."""
        ambient = AmbientLight("ambient")
        ambient.setColor(LColor(0.4, 0.4, 0.4, 1.0))
        ambient_np = self._base.render.attachNewNode(ambient)
        self._base.render.setLight(ambient_np)

        sun = DirectionalLight("sun")
        sun.setColor(LColor(0.8, 0.8, 0.8, 1.0))
        sun_np = self._base.render.attachNewNode(sun)
        sun_np.setHpr(45, -45, 0)
        self._base.render.setLight(sun_np)

    def _setup_agent(self, agent):
        """Create offscreen buffers and camera for an agent."""
        resolution = agent.resolution

        # Create offscreen buffer
        fbprops = FrameBufferProperties()
        fbprops.setRgbColor(True)
        fbprops.setRgbaBits(8, 8, 8, 8)
        fbprops.setDepthBits(32)

        winprops = WindowProperties()
        winprops.setSize(resolution[0], resolution[1])

        buf = self._base.graphicsEngine.makeOutput(
            self._pipe, f"buf_{agent.agent_id}", 0,
            fbprops, winprops, GraphicsPipe.BFRefuseWindow,
        )
        if buf is None:
            raise RuntimeError(f"Failed to create offscreen buffer for {agent.agent_id}")

        # Render textures for readback
        color_tex = Texture()
        buf.addRenderTexture(
            color_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor
        )

        depth_tex = Texture()
        buf.addRenderTexture(
            depth_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth
        )

        # Camera with perspective lens
        cam = Camera(f"cam_{agent.agent_id}")
        lens = PerspectiveLens()
        lens.setFov(agent.fov)
        lens.setNearFar(self._near, self._far)
        lens.setFilmSize(resolution[0], resolution[1])
        cam.setLens(lens)

        cam_np = NodePath(cam)
        cam_np.reparentTo(self._base.render)

        # Set initial agent position/rotation
        self._set_node_pose(cam_np, agent.position, agent.rotation)

        # Create display region
        dr = buf.makeDisplayRegion()
        dr.setCamera(cam_np)

        self._agent_buffers[agent.agent_id] = {
            "buffer": buf,
            "color_tex": color_tex,
            "depth_tex": depth_tex,
            "camera_np": cam_np,
            "lens": lens,
            "agent": agent,
        }

    def _set_node_pose(self, np_node, position, rotation):
        """Set a node's position and rotation from Monty conventions.

        Parameters
        ----------
        np_node : NodePath
            The Panda3D node to transform.
        position : VectorXYZ
            (x, y, z) position.
        rotation : QuaternionWXYZ
            (w, x, y, z) quaternion.
        """
        np_node.setPos(position[0], position[1], position[2])
        # Panda3D quaternion: (w, x, y, z) → setQuat takes (w, x, y, z)
        from panda3d.core import LQuaternionf
        q = LQuaternionf(rotation[0], rotation[1], rotation[2], rotation[3])
        np_node.setQuat(q)

    def remove_all_objects(self) -> None:
        """Remove all objects from the scene."""
        for obj_id, anim_obj in self._animated_objects.items():
            anim_obj.cleanup()
        self._animated_objects.clear()
        for obj_np in self._objects.values():
            obj_np.removeNode()
        self._objects.clear()
        self._object_semantic_ids.clear()

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: SemanticID | None = None,
        primary_target_object: ObjectID | None = None,
        animated: bool = False,
    ) -> ObjectInfo:
        """Add an object to the scene.

        Supports primitive shapes (sphere, cube, cone, cylinder) and
        glTF/GLB 3D models. For models, pass the file path as ``name``.

        Parameters
        ----------
        animated : bool
            If True and the object is a glTF/GLB file, load it as an
            animated Actor with skeletal animation support. Use
            ``pose_object()`` or ``get_animated_object()`` to control
            the animation.
        """
        obj_id = ObjectID(self._next_object_id)
        self._next_object_id += 1

        if semantic_id is None:
            semantic_id = SemanticID(self._next_semantic_id)
            self._next_semantic_id += 1

        # Try primitive generators first
        generator = PRIMITIVE_GENERATORS.get(name)
        if generator is not None:
            obj_np = generator(name=f"{name}_{obj_id}")
        elif self._assets.can_load(name):
            if animated:
                resolved = self._assets.resolve_path(name)
                if resolved is None:
                    raise FileNotFoundError(
                        f"Model file not found: '{name}'"
                    )
                anim_obj = AnimatedObject(resolved, self._base)
                obj_np = anim_obj.node_path
                self._animated_objects[obj_id] = anim_obj
            else:
                obj_np = self._assets.load(
                    name, instance_name=f"model_{obj_id}"
                )
        else:
            raise ValueError(
                f"Unknown object '{name}'. Available primitives: "
                f"{list(PRIMITIVE_GENERATORS.keys())}. "
                f"For 3D models, provide a path to a .gltf or .glb file."
            )

        obj_np.reparentTo(self._base.render)
        obj_np.setScale(scale[0], scale[1], scale[2])
        self._set_node_pose(obj_np, position, rotation)

        # Tag for semantic segmentation
        obj_np.setTag("semantic_id", str(int(semantic_id)))

        self._objects[obj_id] = obj_np
        self._object_semantic_ids[obj_id] = semantic_id

        return ObjectInfo(object_id=obj_id, semantic_id=semantic_id)

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        """Execute actions and return observations.

        Each action's `act()` method is called with this simulator as the
        actuator. Then the scene is rendered and observations are collected.
        """
        for action in actions:
            action.act(self)

        self._render()
        return self._get_observations(), self._get_states()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        """Reset agent positions and return initial observations."""
        for agent in self._agents:
            info = self._agent_buffers[agent.agent_id]
            self._set_node_pose(info["camera_np"], agent.position, agent.rotation)

        self._render()
        return self._get_observations(), self._get_states()

    def _render(self) -> None:
        """Render the scene, updating animated characters if present."""
        if self._animated_objects:
            # taskMgr.step() runs the data loop (character update)
            # followed by the igLoop (render). This is needed for
            # skeletal animation to apply joint transforms to vertices.
            self._base.taskMgr.step()
            self._base.taskMgr.step()  # double step for buffer swap
        else:
            self._base.graphicsEngine.renderFrame()
            self._base.graphicsEngine.renderFrame()

    def close(self) -> None:
        """Clean up Panda3D resources."""
        for anim_obj in self._animated_objects.values():
            anim_obj.cleanup()
        self._animated_objects.clear()
        if self._assets is not None:
            self._assets.clear_cache()
        if self._base is not None:
            self._base.destroy()
            self._base = None

    # --- Animation control ---

    def get_animated_object(self, obj_id: ObjectID) -> AnimatedObject:
        """Get the AnimatedObject for a given object ID.

        Raises KeyError if the object is not animated.
        """
        if obj_id not in self._animated_objects:
            raise KeyError(
                f"Object {obj_id} is not animated. "
                f"Use add_object(..., animated=True) to load animated models."
            )
        return self._animated_objects[obj_id]

    def pose_object(
        self,
        obj_id: ObjectID,
        frame: int,
        anim_name: Optional[str] = None,
    ) -> None:
        """Pose an animated object at a specific frame.

        Parameters
        ----------
        obj_id : ObjectID
            ID returned by add_object.
        frame : int
            Frame number (0-indexed).
        anim_name : str or None
            Animation name. If None, uses the first animation.
        """
        self.get_animated_object(obj_id).pose(frame, anim_name)

    def step_animation(
        self,
        obj_id: ObjectID,
        frame: int,
        anim_name: Optional[str] = None,
    ) -> tuple[Observations, "ProprioceptiveState"]:
        """Pose an animated object and render, returning observations.

        Convenience for: ``pose_object(obj_id, frame)`` → ``step([])`` →
        return ``(observations, proprioceptive_state)``.
        """
        self.pose_object(obj_id, frame, anim_name)
        return self.step([])

    # --- Observation extraction ---

    def _get_observations(self) -> Observations:
        """Read back rendered RGBA + depth and package as Observations."""
        obs: Observations = defaultdict(dict)

        for agent in self._agents:
            info = self._agent_buffers[agent.agent_id]
            color_tex = info["color_tex"]
            depth_tex = info["depth_tex"]
            resolution = agent.resolution

            # Read RGBA
            rgba = self._read_rgba(color_tex, resolution)

            # Read and linearize depth
            depth = self._read_depth(depth_tex, resolution)

            sensor_obs = {
                "rgba": rgba,
                "depth": depth,
            }

            obs[agent.agent_id][SensorID(agent.sensor_id)] = sensor_obs

        return obs

    def _read_rgba(self, tex, resolution):
        """Read RGBA texture to numpy array.

        Returns
        -------
        np.ndarray shape (H, W, 4) dtype uint8
        """
        h, w = resolution
        if tex.hasRamImage():
            data = memoryview(tex.getRamImage())
            # Panda3D stores BGRA; convert to RGBA
            bgra = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4).copy()
            rgba = bgra[:, :, [2, 1, 0, 3]]  # BGRA → RGBA
            # Panda3D renders upside down; flip vertically
            rgba = np.flipud(rgba)
            return rgba
        return np.zeros((h, w, 4), dtype=np.uint8)

    def _read_depth(self, tex, resolution):
        """Read depth texture and linearize to metric distance.

        Panda3D depth buffer is in clip space [0, 1].
        Linearize: z = near * far / (far - d * (far - near))

        Returns
        -------
        np.ndarray shape (H, W, 1) dtype float32
            Linearized depth in meters.
        """
        h, w = resolution
        near = self._near
        far = self._far

        if tex.hasRamImage():
            data = memoryview(tex.getRamImage())
            d_clip = np.frombuffer(data, dtype=np.float32).reshape(h, w).copy()
            # Flip vertically (Panda3D convention)
            d_clip = np.flipud(d_clip)
            # Linearize
            denom = far - d_clip * (far - near)
            denom = np.maximum(denom, 1e-8)  # avoid division by zero
            z_linear = (near * far) / denom
            return z_linear.reshape(h, w, 1).astype(np.float32)

        return np.full((h, w, 1), far, dtype=np.float32)

    def _get_states(self) -> ProprioceptiveState:
        """Extract proprioceptive state for all agents."""
        result = ProprioceptiveState()

        for agent in self._agents:
            info = self._agent_buffers[agent.agent_id]
            cam_np = info["camera_np"]

            pos = cam_np.getPos()
            quat = cam_np.getQuat()

            position = (float(pos.x), float(pos.y), float(pos.z))
            # Convert Panda3D quaternion to numpy quaternion (w, x, y, z)
            rot = quaternion.quaternion(
                float(quat.getR()),
                float(quat.getI()),
                float(quat.getJ()),
                float(quat.getK()),
            )

            sensor_state = SensorState(
                position=(0.0, 0.0, 0.0),  # sensor at agent origin
                rotation=rot,
            )

            result[agent.agent_id] = AgentState(
                position=position,
                rotation=rot,
                sensors={SensorID(agent.sensor_id): sensor_state},
            )

        return result

    # --- Actuator methods (Action dispatch) ---

    def actuate_move_forward(self, action) -> None:
        """Move the camera forward along its look direction."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        # In Panda3D, Y is forward by default
        cam_np.setPos(cam_np, 0, action.distance, 0)

    def actuate_turn_left(self, action) -> None:
        """Turn the camera left (positive heading)."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setH(cam_np.getH() + action.rotation_degrees)

    def actuate_turn_right(self, action) -> None:
        """Turn the camera right (negative heading)."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setH(cam_np.getH() - action.rotation_degrees)

    def actuate_look_up(self, action) -> None:
        """Pitch the camera up."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        new_pitch = cam_np.getP() + action.rotation_degrees
        if hasattr(action, "constraint_degrees"):
            new_pitch = min(new_pitch, action.constraint_degrees)
        cam_np.setP(new_pitch)

    def actuate_look_down(self, action) -> None:
        """Pitch the camera down."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        new_pitch = cam_np.getP() - action.rotation_degrees
        if hasattr(action, "constraint_degrees"):
            new_pitch = max(new_pitch, -action.constraint_degrees)
        cam_np.setP(new_pitch)

    def actuate_set_agent_pose(self, action) -> None:
        """Set the agent's absolute pose."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        loc = action.location
        q = action.rotation_quat
        # rotation_quat may be a numpy.quaternion — convert to (w,x,y,z) tuple
        if hasattr(q, "w") and hasattr(q, "x"):
            q = (float(q.w), float(q.x), float(q.y), float(q.z))
        self._set_node_pose(cam_np, tuple(loc), tuple(q))
        # Store exact float64 values for precise motor state sync.
        # Panda3D uses float32 internally, so reading back loses precision.
        info["_last_set_position"] = tuple(loc)
        info["_last_set_rotation"] = tuple(q)

    def actuate_set_sensor_pose(self, action) -> None:
        """Set a sensor's absolute pose (same as agent for single-sensor)."""
        self.actuate_set_agent_pose(action)

    def actuate_set_sensor_rotation(self, action) -> None:
        """Set the sensor's rotation."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        from panda3d.core import LQuaternionf
        q = action.rotation_quat
        if hasattr(q, "w") and hasattr(q, "x"):
            q = (float(q.w), float(q.x), float(q.y), float(q.z))
        cam_np.setQuat(LQuaternionf(q[0], q[1], q[2], q[3]))

    def actuate_move_tangentially(self, action) -> None:
        """Move tangentially to the object surface."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setPos(cam_np, action.distance, 0, 0)

    def actuate_orient_horizontal(self, action) -> None:
        """Orient horizontally."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setH(action.rotation_degrees)

    def actuate_orient_vertical(self, action) -> None:
        """Orient vertically."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setP(action.rotation_degrees)

    def actuate_set_yaw(self, action) -> None:
        """Set the agent's yaw."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setH(action.rotation_degrees)

    def actuate_set_agent_pitch(self, action) -> None:
        """Set the agent's pitch."""
        info = self._agent_buffers[action.agent_id]
        cam_np = info["camera_np"]
        cam_np.setP(action.rotation_degrees)

    def actuate_set_sensor_pitch(self, action) -> None:
        """Set the sensor's pitch."""
        self.actuate_set_agent_pitch(action)
