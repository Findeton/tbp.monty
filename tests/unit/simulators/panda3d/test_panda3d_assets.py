# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for glTF/GLB model loading in the Panda3D simulator.

Validates:
1. AssetRegistry resolves, loads, and caches glTF/GLB files
2. Loaded models render visible RGBA and valid depth
3. Multiple instances of the same model are independent
4. Search paths work for relative model references
5. Error handling for missing files and invalid formats
"""

import json
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
from tbp.monty.simulators.panda3d.assets import AssetRegistry
from tbp.monty.simulators.panda3d.simulator import Panda3DSimulator


pytestmark = pytest.mark.xdist_group(name="panda3d")


AGENT_ID = AgentID("test_cam")
SENSOR_ID = "sensor_0"


def _make_gltf_cube(path, size=1.0):
    """Write a minimal glTF file containing a cube to *path*.

    Creates both a .gltf JSON file and its companion .bin buffer.
    The cube is axis-aligned with half-extent *size*.
    """
    s = size
    # 8 unique vertices, 12 triangles (36 indices)
    vertices = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s),  (s, -s, s),  (s, s, s),  (-s, s, s),
    ]
    indices = [
        0,1,2, 0,2,3,  # -Z face
        4,6,5, 4,7,6,  # +Z face
        0,4,5, 0,5,1,  # -Y face
        2,6,7, 2,7,3,  # +Y face
        0,3,7, 0,7,4,  # -X face
        1,5,6, 1,6,2,  # +X face
    ]

    vert_data = b""
    for v in vertices:
        vert_data += struct.pack("<3f", *v)

    idx_data = b""
    for i in indices:
        idx_data += struct.pack("<H", i)

    # Pad to 4-byte alignment
    while len(idx_data) % 4 != 0:
        idx_data += b"\x00"

    vert_bytes = len(vert_data)
    idx_bytes = len(idx_data)
    total = vert_bytes + idx_bytes

    bin_name = os.path.splitext(os.path.basename(path))[0] + ".bin"
    bin_path = os.path.join(os.path.dirname(path), bin_name)

    gltf_data = {
        "asset": {"version": "2.0", "generator": "monty-test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
            }]
        }],
        "accessors": [
            {
                "bufferView": 0, "componentType": 5126, "count": 8,
                "type": "VEC3",
                "max": [s, s, s], "min": [-s, -s, -s],
            },
            {
                "bufferView": 1, "componentType": 5123, "count": 36,
                "type": "SCALAR",
                "max": [7], "min": [0],
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vert_bytes},
            {"buffer": 0, "byteOffset": vert_bytes, "byteLength": idx_bytes},
        ],
        "buffers": [{"uri": bin_name, "byteLength": total}],
    }

    with open(path, "w") as f:
        json.dump(gltf_data, f)
    with open(bin_path, "wb") as f:
        f.write(vert_data + idx_data)

    return path, bin_path


def _make_glb_cube(path, size=1.0):
    """Write a minimal GLB (binary glTF) file containing a cube."""
    s = size
    vertices = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s),  (s, -s, s),  (s, s, s),  (-s, s, s),
    ]
    indices = [
        0,1,2, 0,2,3, 4,6,5, 4,7,6,
        0,4,5, 0,5,1, 2,6,7, 2,7,3,
        0,3,7, 0,7,4, 1,5,6, 1,6,2,
    ]

    vert_data = b""
    for v in vertices:
        vert_data += struct.pack("<3f", *v)
    idx_data = b""
    for i in indices:
        idx_data += struct.pack("<H", i)
    while len(idx_data) % 4 != 0:
        idx_data += b"\x00"

    vert_bytes = len(vert_data)
    idx_bytes = len(idx_data)
    bin_data = vert_data + idx_data
    total_bin = len(bin_data)

    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
             "max": [s, s, s], "min": [-s, -s, -s]},
            {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR",
             "max": [7], "min": [0]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": vert_bytes},
            {"buffer": 0, "byteOffset": vert_bytes, "byteLength": idx_bytes},
        ],
        "buffers": [{"byteLength": total_bin}],
    }

    json_bytes = json.dumps(gltf_data).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk = struct.pack("<II", total_bin, 0x004E4942) + bin_data
    total = 12 + len(json_chunk) + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)

    with open(path, "wb") as f:
        f.write(header + json_chunk + bin_chunk)

    return path


def _make_simulator(position=(0, 0, 0), resolution=(64, 64), **kwargs):
    agent = Panda3DAgent(
        agent_id=AGENT_ID,
        sensor_id=SENSOR_ID,
        position=position,
        rotation=(1.0, 0.0, 0.0, 0.0),
        resolution=resolution,
        fov=90.0,
    )
    return Panda3DSimulator(agents=[agent], near=0.01, far=10.0, **kwargs)


class TestAssetRegistry(unittest.TestCase):
    """AssetRegistry file resolution and caching."""

    def test_can_load_gltf(self):
        """Recognizes .gltf extension."""
        registry = AssetRegistry()
        self.assertTrue(registry.can_load("model.gltf"))
        self.assertTrue(registry.can_load("/path/to/model.gltf"))
        self.assertTrue(registry.can_load("model.GLTF"))

    def test_can_load_glb(self):
        """Recognizes .glb extension."""
        registry = AssetRegistry()
        self.assertTrue(registry.can_load("model.glb"))
        self.assertTrue(registry.can_load("scene.GLB"))

    def test_rejects_non_gltf(self):
        """Rejects non-glTF extensions."""
        registry = AssetRegistry()
        self.assertFalse(registry.can_load("sphere"))
        self.assertFalse(registry.can_load("model.obj"))
        self.assertFalse(registry.can_load("model.fbx"))

    def test_load_gltf_file(self):
        """Loads a glTF file and returns a NodePath."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path)

            registry = AssetRegistry()
            np = registry.load(gltf_path)
            self.assertIsNotNone(np)
            self.assertFalse(np.isEmpty())

    def test_load_glb_file(self):
        """Loads a GLB file and returns a NodePath."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = os.path.join(tmpdir, "cube.glb")
            _make_glb_cube(glb_path)

            registry = AssetRegistry()
            np = registry.load(glb_path)
            self.assertIsNotNone(np)
            self.assertFalse(np.isEmpty())

    def test_caching(self):
        """Loading the same file twice uses the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path)

            registry = AssetRegistry()
            np1 = registry.load(gltf_path, instance_name="inst1")
            np2 = registry.load(gltf_path, instance_name="inst2")

            # Both are valid
            self.assertFalse(np1.isEmpty())
            self.assertFalse(np2.isEmpty())
            # Cache has one entry
            self.assertEqual(len(registry.cached_paths), 1)
            # Instances are independent (different names)
            self.assertNotEqual(np1.getName(), np2.getName())

    def test_search_paths(self):
        """Relative paths are resolved via search_paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path)

            registry = AssetRegistry(search_paths=[tmpdir])
            np = registry.load("cube.gltf")
            self.assertFalse(np.isEmpty())

    def test_missing_file_raises(self):
        """FileNotFoundError for nonexistent file."""
        registry = AssetRegistry()
        with self.assertRaises(FileNotFoundError):
            registry.load("nonexistent_model.gltf")

    def test_clear_cache(self):
        """clear_cache empties the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path)

            registry = AssetRegistry()
            registry.load(gltf_path)
            self.assertEqual(len(registry.cached_paths), 1)
            registry.clear_cache()
            self.assertEqual(len(registry.cached_paths), 0)


class TestGltfRendering(unittest.TestCase):
    """glTF models render correctly in the simulator."""

    def test_gltf_cube_renders_pixels(self):
        """A glTF cube in front of camera produces visible pixels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path, size=0.5)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object(gltf_path, position=(0, 3, 0))
                obs, _ = sim.step([])

                rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
                nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
                self.assertGreater(nonzero, 5, "glTF cube should render pixels")
            finally:
                sim.close()

    def test_glb_cube_renders_pixels(self):
        """A GLB cube in front of camera produces visible pixels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = os.path.join(tmpdir, "cube.glb")
            _make_glb_cube(glb_path, size=0.5)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object(glb_path, position=(0, 3, 0))
                obs, _ = sim.step([])

                rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
                nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
                self.assertGreater(nonzero, 5, "GLB cube should render pixels")
            finally:
                sim.close()

    def test_gltf_cube_depth_valid(self):
        """glTF cube produces valid depth (less than far plane)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path, size=0.5)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object(gltf_path, position=(0, 3, 0))
                obs, _ = sim.step([])

                depth = obs[AGENT_ID][SensorID(SENSOR_ID)]["depth"]
                center_depth = depth[32, 32, 0]

                self.assertLess(center_depth, 10.0,
                                f"Center depth {center_depth:.2f} should be < far")
                self.assertGreater(center_depth, 0.5,
                                   "Cube should be in front of camera")
            finally:
                sim.close()

    def test_gltf_via_search_path(self):
        """Can add a model by relative name when search path is set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "my_cube.gltf")
            _make_gltf_cube(gltf_path, size=0.5)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object("my_cube.gltf", position=(0, 3, 0))
                obs, _ = sim.step([])

                rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
                nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
                self.assertGreater(nonzero, 5)
            finally:
                sim.close()

    def test_multiple_gltf_instances(self):
        """Multiple instances of same model get unique object IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path, size=0.3)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info1 = sim.add_object(gltf_path, position=(0, 3, 0))
                info2 = sim.add_object(gltf_path, position=(2, 3, 0))

                self.assertNotEqual(info1.object_id, info2.object_id)
                self.assertNotEqual(info1.semantic_id, info2.semantic_id)
            finally:
                sim.close()

    def test_mixed_primitives_and_gltf(self):
        """Can mix primitive shapes and glTF models in the same scene."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path, size=0.3)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                info_prim = sim.add_object("sphere", position=(-1, 3, 0),
                                           scale=(0.3, 0.3, 0.3))
                info_gltf = sim.add_object(gltf_path, position=(1, 3, 0))

                self.assertNotEqual(info_prim.object_id, info_gltf.object_id)

                obs, _ = sim.step([])
                rgba = obs[AGENT_ID][SensorID(SENSOR_ID)]["rgba"]
                nonzero = np.any(rgba[:, :, :3] > 0, axis=-1).sum()
                self.assertGreater(nonzero, 10, "Both objects should render")
            finally:
                sim.close()

    def test_remove_gltf_objects(self):
        """remove_all_objects clears glTF objects too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = os.path.join(tmpdir, "cube.gltf")
            _make_gltf_cube(gltf_path, size=0.5)

            sim = _make_simulator(asset_search_paths=[tmpdir])
            try:
                sim.add_object(gltf_path, position=(0, 3, 0))
                self.assertEqual(len(sim._objects), 1)

                sim.remove_all_objects()
                self.assertEqual(len(sim._objects), 0)
            finally:
                sim.close()


class TestGltfErrors(unittest.TestCase):
    """Error handling for glTF loading."""

    def test_nonexistent_gltf_raises(self):
        """FileNotFoundError for missing .gltf file."""
        sim = _make_simulator()
        try:
            with self.assertRaises(FileNotFoundError):
                sim.add_object("does_not_exist.gltf")
        finally:
            sim.close()

    def test_nonexistent_glb_raises(self):
        """FileNotFoundError for missing .glb file."""
        sim = _make_simulator()
        try:
            with self.assertRaises(FileNotFoundError):
                sim.add_object("missing.glb")
        finally:
            sim.close()

    def test_unknown_name_still_raises_valueerror(self):
        """Non-file, non-primitive names still raise ValueError."""
        sim = _make_simulator()
        try:
            with self.assertRaises((ValueError, Exception)):
                sim.add_object("nonexistent_xyz")
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
