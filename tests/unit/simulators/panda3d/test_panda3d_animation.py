# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for animated glTF models in the Panda3D simulator.

Validates:
1. Animated models load and expose their animations
2. Posing at different frames produces different rendered output
3. Programmatic joint control via controlJoint deforms geometry
4. AnimatedObject provides frame counting and joint introspection
5. Animation integrates with the simulator step/render pipeline
"""

import json
import math
import os
import struct
import tempfile
import unittest

import numpy as np
import pytest

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.sensors import SensorID

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.simulators.panda3d.agents import Panda3DAgent
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator


pytestmark = pytest.mark.xdist_group(name="panda3d")


AGENT_ID = AgentID("test_cam")
SENSOR_ID = "sensor_0"


def _make_animated_gltf(directory, filename="animated.gltf"):
    """Create a minimal glTF with a 2-joint skeleton and rotation animation.

    The model is a rectangular bar from y=0 to y=2 (in glTF Y-up coords).
    Joint 0 (Bone0) is at the origin; joint 1 (Bone1) is at y=1.
    The animation rotates Bone1 by 90 degrees around the Z axis and back.
    """
    s = 0.3
    vertices = [
        (-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s),
        (-s, 2, -s), (s, 2, -s), (s, 2, s), (-s, 2, s),
    ]
    indices = [
        0, 1, 2, 0, 2, 3,  # bottom
        4, 6, 5, 4, 7, 6,  # top
        0, 4, 5, 0, 5, 1,  # front
        2, 6, 7, 2, 7, 3,  # back
        0, 3, 7, 0, 7, 4,  # left
        1, 5, 6, 1, 6, 2,  # right
    ]
    # Bottom 4 verts → joint 0, top 4 → joint 1
    joints_data = [(0, 0, 0, 0)] * 4 + [(1, 0, 0, 0)] * 4
    weights_data = [(1.0, 0, 0, 0)] * 8

    vert_bin = b"".join(struct.pack("<3f", *v) for v in vertices)
    idx_bin = b"".join(struct.pack("<H", i) for i in indices)
    while len(idx_bin) % 4:
        idx_bin += b"\x00"
    joints_bin = b"".join(struct.pack("<4B", *j) for j in joints_data)
    weights_bin = b"".join(struct.pack("<4f", *w) for w in weights_data)

    # Inverse bind matrices
    ibm0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ibm1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1]
    ibm_bin = struct.pack("<16f", *ibm0) + struct.pack("<16f", *ibm1)

    # Rotation keyframes: identity → 90° around Z → identity
    s45 = math.sin(math.radians(45))
    c45 = math.cos(math.radians(45))
    anim_times = struct.pack("<3f", 0.0, 0.5, 1.0)
    # glTF quaternions are (x, y, z, w)
    anim_rots = struct.pack(
        "<12f", 0, 0, 0, 1, 0, 0, s45, c45, 0, 0, 0, 1
    )

    offset = 0
    buf_views = []
    all_parts = [
        vert_bin, idx_bin, joints_bin, weights_bin,
        ibm_bin, anim_times, anim_rots,
    ]
    for part in all_parts:
        buf_views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(part),
        })
        offset += len(part)
    all_data = b"".join(all_parts)

    gltf_data = {
        "asset": {"version": "2.0", "generator": "monty-test"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "Armature", "children": [2]},
            {"name": "SkinnedMesh", "mesh": 0, "skin": 0},
            {"name": "Bone0", "children": [3]},
            {"name": "Bone1", "translation": [0, 1, 0]},
        ],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3,
                },
                "indices": 1,
            }],
        }],
        "skins": [{
            "joints": [2, 3],
            "inverseBindMatrices": 4,
            "skeleton": 2,
        }],
        "animations": [{
            "name": "BendArm",
            "channels": [{
                "sampler": 0,
                "target": {"node": 3, "path": "rotation"},
            }],
            "samplers": [{
                "input": 5, "output": 6, "interpolation": "LINEAR",
            }],
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8,
             "type": "VEC3", "max": [s, 2, s], "min": [-s, 0, -s]},
            {"bufferView": 1, "componentType": 5123,
             "count": len(indices), "type": "SCALAR",
             "max": [7], "min": [0]},
            {"bufferView": 2, "componentType": 5121, "count": 8,
             "type": "VEC4", "max": [1, 0, 0, 0], "min": [0, 0, 0, 0]},
            {"bufferView": 3, "componentType": 5126, "count": 8,
             "type": "VEC4", "max": [1, 0, 0, 0], "min": [0, 0, 0, 0]},
            {"bufferView": 4, "componentType": 5126, "count": 2,
             "type": "MAT4"},
            {"bufferView": 5, "componentType": 5126, "count": 3,
             "type": "SCALAR", "max": [1.0], "min": [0.0]},
            {"bufferView": 6, "componentType": 5126, "count": 3,
             "type": "VEC4"},
        ],
        "bufferViews": buf_views,
        "buffers": [{"uri": "anim.bin", "byteLength": len(all_data)}],
    }

    gltf_path = os.path.join(directory, filename)
    bin_path = os.path.join(directory, "anim.bin")
    with open(gltf_path, "w") as f:
        json.dump(gltf_data, f)
    with open(bin_path, "wb") as f:
        f.write(all_data)
    return gltf_path


def _make_simulator(position=(0, 0, 0), resolution=(64, 64), **kwargs):
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=position,
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=resolution,
        fov=90.0,
    )
    return Panda3DSimulator(agents=[agent], near=0.01, far=20.0, **kwargs)


class TestAnimatedObjectLoading(unittest.TestCase):
    """Loading animated glTF models."""

    def test_load_animated_model(self):
        """Can load an animated glTF as animated=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1),
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(gltf_path, animated=True)
                self.assertIsNotNone(info.object_id)
                self.assertIn(info.object_id, sim._animated_objects)
            finally:
                sim.close()

    def test_animation_names(self):
        """Animated model exposes its animation names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                self.assertIn("BendArm", anim.animation_names)
            finally:
                sim.close()

    def test_frame_count(self):
        """Can query number of frames in animation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                self.assertGreater(anim.get_num_frames(), 1)
            finally:
                sim.close()

    def test_joint_names(self):
        """Can query joint names from animated model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                joint_names = anim.get_joint_names()
                self.assertIn("Bone0", joint_names)
                self.assertIn("Bone1", joint_names)
            finally:
                sim.close()

    def test_expose_joint(self):
        """Can expose a live joint node for animated transform inspection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                joint = anim.expose_joint("Bone1")
                self.assertFalse(joint.isEmpty())
            finally:
                sim.close()

    def test_non_animated_object_raises(self):
        """get_animated_object raises for non-animated objects."""
        sim = _make_simulator()
        try:
            info = sim.add_object("sphere", position=(0, 3, 0))
            with self.assertRaises(KeyError):
                sim.get_animated_object(info.object_id)
        finally:
            sim.close()


class TestAnimationRendering(unittest.TestCase):
    """Animation produces different rendered output at different frames."""

    def test_different_frames_different_rgba(self):
        """Posing at distant frames produces different RGBA output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1),
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()

                # Pose at frame 0
                anim.pose(0)
                obs0, _ = sim.step([])
                rgba0 = obs0[AGENT_ID][SensorID(SENSOR_ID)]["rgba"].copy()

                # Pose at a distant frame (2/3 through animation)
                far_frame = (n_frames * 2) // 3
                anim.pose(far_frame)
                obs1, _ = sim.step([])
                rgba1 = obs1[AGENT_ID][SensorID(SENSOR_ID)]["rgba"].copy()

                diff = np.abs(rgba0.astype(float) - rgba1.astype(float)).sum()
                self.assertGreater(
                    diff, 0,
                    f"Frames 0 and {far_frame} should render differently"
                )
            finally:
                sim.close()

    def test_different_frames_different_depth(self):
        """Posing at different frames produces different depth output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1),
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                n_frames = anim.get_num_frames()

                anim.pose(0)
                obs0, _ = sim.step([])
                d0 = obs0[AGENT_ID][SensorID(SENSOR_ID)]["depth"].copy()

                far_frame = (n_frames * 2) // 3
                anim.pose(far_frame)
                obs1, _ = sim.step([])
                d1 = obs1[AGENT_ID][SensorID(SENSOR_ID)]["depth"].copy()

                diff = np.abs(d0 - d1).sum()
                self.assertGreater(diff, 0, "Depth should change with pose")
            finally:
                sim.close()

    def test_advance_wraps(self):
        """advance() wraps around at the end of the animation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)
                n = anim.get_num_frames()

                anim.pose(n - 2)
                new_frame = anim.advance(5)
                self.assertEqual(new_frame, (n - 2 + 5) % n)
            finally:
                sim.close()

    def test_pose_object_convenience(self):
        """pose_object() convenience method works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(3, -5, 1),
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(gltf_path, animated=True)
                # Should not raise
                sim.pose_object(info.object_id, 10)
                obs, _ = sim.step([])
                rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
                self.assertEqual(rgba.shape, (64, 64, 4))
            finally:
                sim.close()


class TestProgrammaticAnimation(unittest.TestCase):
    """Programmatic joint control via controlJoint."""

    def test_control_joint_changes_rendering(self):
        """Moving a controlled joint produces different depth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(
                position=(0, -5, 1),
                asset_search_paths=[tmpdir],
            )
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)

                # Render at default pose
                obs0, _ = sim.step([])
                d0 = obs0[AGENT_ID][SensorID(SENSOR_ID)]["depth"].copy()

                # Control joint and move it
                bone1 = anim.control_joint("Bone1")
                bone1.setPos(bone1.getPos() + (2, 0, 0))
                obs1, _ = sim.step([])
                d1 = obs1[AGENT_ID][SensorID(SENSOR_ID)]["depth"].copy()

                diff = np.abs(d0 - d1).sum()
                self.assertGreater(
                    diff, 0, "controlJoint should change rendered depth"
                )
            finally:
                sim.close()

    def test_release_joint(self):
        """release_joint() removes the control node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info = sim.add_object(gltf_path, animated=True)
                anim = sim.get_animated_object(info.object_id)

                anim.control_joint("Bone1")
                self.assertIn("Bone1", anim._controlled_joints)

                anim.release_joint("Bone1")
                self.assertNotIn("Bone1", anim._controlled_joints)
            finally:
                sim.close()


class TestAnimationCleanup(unittest.TestCase):
    """Animated objects are properly cleaned up."""

    def test_remove_all_cleans_animated(self):
        """remove_all_objects cleans up animated objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object(gltf_path, animated=True)
                self.assertEqual(len(sim._animated_objects), 1)

                sim.remove_all_objects()
                self.assertEqual(len(sim._animated_objects), 0)
                self.assertEqual(len(sim._objects), 0)
            finally:
                sim.close()

    def test_close_cleans_animated(self):
        """close() cleans up animated objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            sim = _make_simulator(asset_search_paths=[tmpdir])
            sim.add_object(gltf_path, animated=True)
            sim.close()
            self.assertEqual(len(sim._animated_objects), 0)


if __name__ == "__main__":
    unittest.main()
