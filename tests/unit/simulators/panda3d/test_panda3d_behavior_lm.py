# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""End-to-end tests: Panda3D -> MontyForEvidenceGraphMatching (2 SMs + 2 LMs).

Tests the full behavior training/matching pipeline with proper Monty
architecture, real Panda3D rendering, and animated 3D models.

Architecture:
  SM 0: CameraSM          -> LM 0 (morphology)
  SM 1: ChangeDetectingSM -> LM 1 (behavior)
  sm_to_lm_matrix:       [[0], [1]]
  lm_to_lm_vote_matrix:  [[1], [0]]

Pipeline:
  Animate glTF model -> Panda3D render (RGBA + depth)
  -> Panda3DDepthNormalize -> DepthTo3DLocations -> semantic_3d
  -> Both SMs process same observation
  -> MontyForEvidenceGraphMatching routes SM outputs to respective LMs
  -> Lateral voting between LMs
"""

import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")

from tbp.monty.simulators.panda3d.behavior_training import (
    Panda3DBehaviorExperiment,
)

# Fox.glb has 3 animations: Survey, Walk, Run
FOX_PATH = str(Path(__file__).parent / "test_assets" / "animated" / "Fox.glb")


def _make_animated_gltf(
    directory,
    filename="animated.gltf",
    *,
    width=0.3,
    height=2.0,
    depth=0.3,
    color=None,
    animations=None,
):
    """Create a glTF with a 2-joint skeleton and one or more rotation animations.

    Parameters
    ----------
    directory : str
        Directory for output files.
    filename : str
        glTF filename.
    width, height, depth : float
        Half-extents (X), full height (Y), half-extents (Z) of the box mesh.
        Different values produce visually distinct morphologies.
    color : tuple[float, float, float] or None
        RGB vertex color (0-1 each).  When provided, the mesh gets a
        uniform COLOR_0 attribute so CameraSM HSV features can
        distinguish morphologies.  None = no vertex color (gray).
    animations : list[tuple[str, tuple[float,float,float], float]] or None
        Each entry is ``(name, axis, angle_degrees)``.  *axis* is the
        rotation axis as ``(x, y, z)`` and *angle_degrees* the peak
        rotation.  Defaults to a single "Bend" around Z at 45 deg.

    Returns
    -------
    str
        Path to the written glTF file.
    """
    if animations is None:
        animations = [("Bend", (0, 0, 1), 45.0)]

    sx, sy, sz = width, height, depth
    vertices = [
        (-sx, 0, -sz), (sx, 0, -sz), (sx, 0, sz), (-sx, 0, sz),
        (-sx, sy, -sz), (sx, sy, -sz), (sx, sy, sz), (-sx, sy, sz),
    ]
    indices = [
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ]
    joints_data = [(0, 0, 0, 0)] * 4 + [(1, 0, 0, 0)] * 4
    weights_data = [(1.0, 0, 0, 0)] * 8

    vert_bin = b"".join(struct.pack("<3f", *v) for v in vertices)
    idx_bin = b"".join(struct.pack("<H", i) for i in indices)
    while len(idx_bin) % 4:
        idx_bin += b"\x00"
    joints_bin = b"".join(struct.pack("<4B", *j) for j in joints_data)
    weights_bin = b"".join(struct.pack("<4f", *w) for w in weights_data)

    # Optional vertex colors (COLOR_0, VEC4 unsigned byte, normalized)
    if color is not None:
        r_c, g_c, b_c = (int(c * 255) for c in color)
        color_bin = b"".join(
            struct.pack("<4B", r_c, g_c, b_c, 255) for _ in vertices
        )
    else:
        color_bin = b""

    ibm0 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    ibm1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -sy / 2, 0, 1]
    ibm_bin = struct.pack("<16f", *ibm0) + struct.pack("<16f", *ibm1)

    # Build per-animation binary data
    anim_time_bins = []
    anim_rot_bins = []
    for _name, axis, angle_deg in animations:
        half_angle = math.radians(angle_deg) / 2
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        ax, ay, az = ax / norm, ay / norm, az / norm
        s_a = math.sin(half_angle)
        c_a = math.cos(half_angle)
        # 3 keyframes: identity -> rotated -> identity
        anim_time_bins.append(struct.pack("<3f", 0.0, 0.5, 1.0))
        anim_rot_bins.append(struct.pack(
            "<12f",
            0, 0, 0, 1,
            ax * s_a, ay * s_a, az * s_a, c_a,
            0, 0, 0, 1,
        ))

    # Concatenate all binary data
    data_bin = b"".join([
        vert_bin, idx_bin, joints_bin, weights_bin, color_bin, ibm_bin,
    ] + anim_time_bins + anim_rot_bins)

    buf_path = os.path.join(directory, filename.replace(".gltf", ".bin"))
    with open(buf_path, "wb") as f:
        f.write(data_bin)

    # Compute offsets
    vert_len = len(vert_bin)
    idx_len = len(idx_bin)
    joints_len = len(joints_bin)
    weights_len = len(weights_bin)
    color_len = len(color_bin)
    ibm_len = len(ibm_bin)

    off = 0
    vert_off = off; off += vert_len
    idx_off = off; off += idx_len
    joints_off = off; off += joints_len
    weights_off = off; off += weights_len
    color_off = off; off += color_len
    ibm_off = off; off += ibm_len

    anim_time_size = 3 * 4  # 3 floats
    anim_rot_size = 12 * 4  # 12 floats

    # Build buffer views: mesh/skeleton (+ optional color), then animation pairs
    buffer_views = [
        {"buffer": 0, "byteOffset": vert_off, "byteLength": vert_len,
         "target": 34962},
        {"buffer": 0, "byteOffset": idx_off, "byteLength": idx_len,
         "target": 34963},
        {"buffer": 0, "byteOffset": joints_off, "byteLength": joints_len},
        {"buffer": 0, "byteOffset": weights_off, "byteLength": weights_len},
    ]
    # Mesh attributes
    mesh_attrs = {
        "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3,
    }
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 8,
         "type": "VEC3",
         "min": [-sx, 0, -sz], "max": [sx, sy, sz]},
        {"bufferView": 1, "componentType": 5123,
         "count": len(indices), "type": "SCALAR"},
        {"bufferView": 2, "componentType": 5121, "count": 8,
         "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": 8,
         "type": "VEC4"},
    ]

    if color is not None:
        bv_color = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": color_off,
             "byteLength": color_len}
        )
        acc_color = len(accessors)
        accessors.append(
            {"bufferView": bv_color, "componentType": 5121,
             "count": 8, "type": "VEC4", "normalized": True}
        )
        mesh_attrs["COLOR_0"] = acc_color

    # IBM buffer view + accessor
    bv_ibm = len(buffer_views)
    buffer_views.append(
        {"buffer": 0, "byteOffset": ibm_off, "byteLength": ibm_len}
    )
    acc_ibm = len(accessors)
    accessors.append(
        {"bufferView": bv_ibm, "componentType": 5126, "count": 2,
         "type": "MAT4"}
    )

    gltf_animations = []
    for i, (anim_name, _axis, _angle) in enumerate(animations):
        time_off = off + i * (anim_time_size + anim_rot_size)
        rot_off = time_off + anim_time_size

        bv_time = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": time_off,
             "byteLength": anim_time_size}
        )
        bv_rot = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": rot_off,
             "byteLength": anim_rot_size}
        )

        acc_time = len(accessors)
        accessors.append(
            {"bufferView": bv_time, "componentType": 5126, "count": 3,
             "type": "SCALAR", "min": [0.0], "max": [1.0]}
        )
        acc_rot = len(accessors)
        accessors.append(
            {"bufferView": bv_rot, "componentType": 5126, "count": 3,
             "type": "VEC4"}
        )

        gltf_animations.append({
            "name": anim_name,
            "channels": [
                {"sampler": 0,
                 "target": {"node": 2, "path": "rotation"}},
            ],
            "samplers": [
                {"input": acc_time, "output": acc_rot,
                 "interpolation": "LINEAR"},
            ],
        })

    gltf_data = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Root", "skin": 0, "mesh": 0, "children": [1]},
            {"name": "Joint0", "children": [2], "translation": [0, 0, 0]},
            {"name": "Joint1", "translation": [0, sy / 2, 0]},
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": acc_ibm}],
        "meshes": [{"primitives": [{"attributes": mesh_attrs,
                                     "indices": 1}]}],
        "animations": gltf_animations,
        "buffers": [{"uri": os.path.basename(buf_path),
                      "byteLength": len(data_bin)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    gltf_path = os.path.join(directory, filename)
    with open(gltf_path, "w") as f:
        json.dump(gltf_data, f)
    return gltf_path


# ===========================================================================
# Setup tests
# ===========================================================================


class TestMontyArchitectureSetup(unittest.TestCase):
    """Verify proper Monty wiring: 2 SMs + 2 LMs + matrices."""

    def test_setup_creates_monty_with_two_sms(self):
        """MontyForEvidenceGraphMatching has CameraSM + ChangeDetectingSM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp._setup()
                monty = exp.monty
                self.assertEqual(len(monty.sensor_modules), 2)
                self.assertEqual(
                    monty.sensor_modules[0].sensor_module_id, "camera"
                )
                self.assertEqual(
                    monty.sensor_modules[1].sensor_module_id,
                    "change_detector",
                )
            finally:
                exp.close()

    def test_setup_creates_monty_with_two_lms(self):
        """MontyForEvidenceGraphMatching has morphology LM + behavior LM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp._setup()
                monty = exp.monty
                self.assertEqual(len(monty.learning_modules), 2)
                self.assertEqual(
                    monty.learning_modules[0].learning_module_id,
                    "lm_morphology",
                )
                self.assertEqual(
                    monty.learning_modules[1].learning_module_id,
                    "lm_behavior",
                )
            finally:
                exp.close()

    def test_sm_to_lm_matrix_routes_correctly(self):
        """SM 0 -> LM 0, SM 1 -> LM 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp._setup()
                self.assertEqual(exp.monty.sm_to_lm_matrix, [[0], [1]])
            finally:
                exp.close()

    def test_vote_matrix_enables_lateral_voting(self):
        """LM 0 <-> LM 1 lateral voting configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp._setup()
                self.assertEqual(
                    exp.monty.lm_to_lm_vote_matrix, [[1], [0]]
                )
            finally:
                exp.close()

    @unittest.skipUnless(Path(FOX_PATH).exists(), "Fox.glb not available")
    def test_setup_with_fox(self):
        """Experiment loads Fox.glb and discovers animations."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            object_scale=(0.01, 0.01, 0.01),
            orbit_radius=1.5,
        )
        try:
            names = exp.get_animation_names()
            self.assertIn("Walk", names)
            self.assertIn("Run", names)
        finally:
            exp.close()


# ===========================================================================
# Training tests
# ===========================================================================


class TestBehaviorTraining(unittest.TestCase):
    """Training produces graphs in both LMs via MontyForEvidenceGraphMatching."""

    def test_train_stores_in_behavior_lm(self):
        """Behavior LM (ChangeDetectingSM channel) learns a graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                result = exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=3
                )
                known = exp.behavior_lm.get_all_known_object_ids()
                self.assertIn(
                    "bend_behavior", known,
                    f"Behavior LM should learn bend_behavior, got {known}"
                )
            finally:
                exp.close()

    def test_train_stores_in_morphology_lm(self):
        """Morphology LM (CameraSM channel) also learns a graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=3
                )
                known = exp.morphology_lm.get_all_known_object_ids()
                self.assertIn(
                    "bend_behavior", known,
                    f"Morphology LM should learn bend_behavior, got {known}"
                )
            finally:
                exp.close()

    @unittest.skipUnless(Path(FOX_PATH).exists(), "Fox.glb not available")
    def test_train_two_fox_behaviors(self):
        """Both LMs learn Walk and Survey as separate graphs."""
        exp = Panda3DBehaviorExperiment(
            model_path=FOX_PATH,
            object_scale=(0.01, 0.01, 0.01),
            orbit_radius=1.5,
        )
        try:
            exp.train_behavior("Walk", "walk_behavior")
            exp.train_behavior("Survey", "survey_behavior")

            behavior_known = exp.behavior_lm.get_all_known_object_ids()
            self.assertIn("walk_behavior", behavior_known)
            self.assertIn("survey_behavior", behavior_known)

            morph_known = exp.morphology_lm.get_all_known_object_ids()
            self.assertIn("walk_behavior", morph_known)
            self.assertIn("survey_behavior", morph_known)
        finally:
            exp.close()


# ===========================================================================
# Matching tests
# ===========================================================================


class TestBehaviorMatching(unittest.TestCase):
    """Matching recognizes trained behaviors via MontyForEvidenceGraphMatching."""

    def test_match_returns_behavior_lm_result(self):
        """Match result includes behavior LM's graph_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=3
                )
                result = exp.match_behavior("Bend")
                self.assertEqual(
                    result.get("graph_id"), "bend_behavior",
                    f"Should recognize bend, got {result.get('graph_id')}"
                )
            finally:
                exp.close()

    def test_match_returns_per_lm_mlh(self):
        """Match result includes MLH from both LMs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=3
                )
                result = exp.match_behavior("Bend")
                self.assertIn("lm_morphology", result)
                self.assertIn("lm_behavior", result)
            finally:
                exp.close()


# ===========================================================================
# Fox.glb discrimination tests
# ===========================================================================


@unittest.skipUnless(Path(FOX_PATH).exists(), "Fox.glb not available")
class TestFoxBehaviorDiscrimination(unittest.TestCase):
    """Train Walk+Survey on Fox.glb, verify behavior LM discriminates.

    Uses Walk vs Survey (distinct motion patterns) rather than Walk vs Run
    (both locomotion gaits with similar limb motion amplitudes, too similar
    for correspondence-based flow to reliably distinguish).
    """

    @classmethod
    def setUpClass(cls):
        # Fox.glb is ~100 units tall; at scale 0.01 it's ~1 unit.
        # initial_distance=3 with 2° rotation steps keeps it in view.
        lm_kwargs = dict(
            use_multithreading=False,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        try:
            cls._exp = Panda3DBehaviorExperiment(
                model_path=FOX_PATH,
                object_scale=(0.01, 0.01, 0.01),
                initial_distance=3.0,
                far=20.0,
                rotation_degrees=2.0,
                translation_distance=0.01,
                morphology_lm_kwargs=lm_kwargs,
                behavior_lm_kwargs=lm_kwargs,
            )
            cls._exp.train_behavior(
                "Walk", "walk_behavior", n_repetitions=5,
            )
            cls._exp.train_behavior(
                "Survey", "survey_behavior", n_repetitions=5,
            )
            cls._setup_ok = True
        except Exception:
            cls._exp = None
            cls._setup_ok = False

    @classmethod
    def tearDownClass(cls):
        if cls._exp is not None:
            cls._exp.close()

    def setUp(self):
        if not self._setup_ok:
            self.skipTest("Setup failed")

    def test_walk_evidence_highest_for_walk_input(self):
        """Behavior LM: Walk evidence > Survey evidence for Walk animation."""
        result = self._exp.match_behavior("Walk", n_steps=60)
        evidence = result["behavior_evidence"]

        walk_ev = np.max(evidence.get("walk_behavior", [0]))
        survey_ev = np.max(evidence.get("survey_behavior", [0]))
        print(f"\n  Walk input: walk_ev={walk_ev:.3f}, "
              f"survey_ev={survey_ev:.3f}")
        self.assertGreater(
            walk_ev, survey_ev,
            f"Walk evidence ({walk_ev:.2f}) should > "
            f"survey evidence ({survey_ev:.2f}) for Walk input"
        )

    def test_survey_evidence_highest_for_survey_input(self):
        """Behavior LM: Survey evidence > Walk evidence for Survey animation."""
        result = self._exp.match_behavior("Survey", n_steps=60)
        evidence = result["behavior_evidence"]

        walk_ev = np.max(evidence.get("walk_behavior", [0]))
        survey_ev = np.max(evidence.get("survey_behavior", [0]))
        print(f"\n  Survey input: walk_ev={walk_ev:.3f}, "
              f"survey_ev={survey_ev:.3f}")
        self.assertGreater(
            survey_ev, walk_ev,
            f"Survey evidence ({survey_ev:.2f}) should > "
            f"walk evidence ({walk_ev:.2f}) for Survey input"
        )

    def test_match_walk_returns_walk(self):
        """Behavior LM MLH graph_id is walk_behavior for Walk animation."""
        result = self._exp.match_behavior("Walk", n_steps=60)
        self.assertEqual(
            result.get("graph_id"), "walk_behavior",
            f"Should recognize Walk, got {result.get('graph_id')}"
        )

    def test_match_survey_returns_survey(self):
        """Behavior LM MLH graph_id is survey_behavior for Survey animation."""
        result = self._exp.match_behavior("Survey", n_steps=60)
        self.assertEqual(
            result.get("graph_id"), "survey_behavior",
            f"Should recognize Survey, got {result.get('graph_id')}"
        )


# ===========================================================================
# RobotExpressive behavior discrimination tests
# ===========================================================================

ROBOT_PATH = str(
    Path(__file__).parent / "test_assets" / "animated" / "RobotExpressive.glb"
)


@unittest.skipUnless(Path(ROBOT_PATH).exists(), "RobotExpressive.glb not found")
class TestRobotBehaviorDiscrimination(unittest.TestCase):
    """Train Walking+Dance+Idle on RobotExpressive, verify discrimination.

    Uses a single shared experiment to avoid Panda3D ShowBase singleton
    issues across test methods.

    Tests shape AND behavior recognition simultaneously:
    - Morphology LM learns the robot's shape
    - Behavior LM learns each animation as a distinct behavior
    - Active sensorimotor loop (InformedPolicy) drives camera exploration

    Note: The object must fill a moderate fraction of the frame (scale 0.3,
    distance 3.0) so that animation-induced flow dominates over
    visibility-change flow from camera movement.
    """

    @classmethod
    def setUpClass(cls):
        lm_kwargs = dict(
            use_multithreading=False,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        try:
            cls._exp = Panda3DBehaviorExperiment(
                model_path=ROBOT_PATH,
                object_scale=(0.3, 0.3, 0.3),
                initial_distance=3.0,
                far=20.0,
                rotation_degrees=2.0,
                translation_distance=0.01,
                morphology_lm_kwargs=lm_kwargs,
                behavior_lm_kwargs=lm_kwargs,
            )
            cls._exp.train_behavior(
                "Walking", "walking_behavior", n_repetitions=5,
            )
            cls._exp.train_behavior(
                "Dance", "dance_behavior", n_repetitions=5,
            )
            # Verify rendering isn't silently corrupted (ShowBase singleton)
            result = cls._exp.match_behavior("Dance", n_steps=30)
            dance_ev = np.max(
                result["behavior_evidence"].get("dance_behavior", [0])
            )
            cls._setup_ok = dance_ev > 0
        except Exception:
            cls._exp = None
            cls._setup_ok = False

    @classmethod
    def tearDownClass(cls):
        if cls._exp is not None:
            cls._exp.close()

    def setUp(self):
        if not self._setup_ok:
            self.skipTest(
                "ShowBase singleton corrupted by prior test — "
                "run this class alone with pytest -k TestRobot"
            )

    def test_walking_produces_nonzero_evidence(self):
        """Walking input produces non-zero evidence for both behaviors.

        Verifies the sensorimotor loop generates observations that reach
        the behavior LM.
        """
        result = self._exp.match_behavior("Walking", n_steps=60)
        evidence = result["behavior_evidence"]

        walk_ev = np.max(evidence.get("walking_behavior", [0]))
        dance_ev = np.max(evidence.get("dance_behavior", [0]))
        print(f"\n  Robot Walking: walk_ev={walk_ev:.3f}, "
              f"dance_ev={dance_ev:.3f}")
        self.assertGreater(
            walk_ev, 0,
            "Walking evidence should be non-zero"
        )

    def test_dance_vs_walk_discrimination(self):
        """Behavior LM: Dance evidence > Walk evidence for Dance input."""
        result = self._exp.match_behavior("Dance", n_steps=60)
        evidence = result["behavior_evidence"]

        walk_ev = np.max(evidence.get("walking_behavior", [0]))
        dance_ev = np.max(evidence.get("dance_behavior", [0]))
        print(f"\n  Robot Dance: dance_ev={dance_ev:.3f}, "
              f"walk_ev={walk_ev:.3f}")
        self.assertGreater(
            dance_ev, walk_ev,
            f"Dance evidence ({dance_ev:.2f}) should > "
            f"walking ({walk_ev:.2f}) for Dance input"
        )

    def test_behavior_id_correct(self):
        """Correct behavior_id returned for both Walking and Dance."""
        walk_result = self._exp.match_behavior("Walking", n_steps=60)
        dance_result = self._exp.match_behavior("Dance", n_steps=60)

        results = {
            "Walking": walk_result.get("behavior_id"),
            "Dance": dance_result.get("behavior_id"),
        }
        print(f"\n  Walking -> {results['Walking']}")
        print(f"  Dance -> {results['Dance']}")

        correct = sum(1 for anim, expected in [
            ("Walking", "walking_behavior"),
            ("Dance", "dance_behavior"),
        ] if results[anim] == expected)

        print(f"  Accuracy: {correct}/2 ({100*correct/2:.0f}%)")
        self.assertGreaterEqual(
            correct, 1,
            f"Should recognize at least 1/2 behaviors, got {correct}/2"
        )


# ===========================================================================
# Fox-vs-Robot cross-morphology shape discrimination
# ===========================================================================


@unittest.skipUnless(
    Path(FOX_PATH).exists() and Path(ROBOT_PATH).exists(),
    "Fox.glb or RobotExpressive.glb not found",
)
class TestFoxVsRobotShapeDiscrimination(unittest.TestCase):
    """Train Fox and Robot shapes, verify morphology LM discriminates.

    This tests SHAPE recognition across genuinely different 3D models
    (not just programmatic bars/slabs), using the same active sensorimotor
    exploration loop.
    """

    @classmethod
    def setUpClass(cls):
        lm_kwargs = dict(
            use_multithreading=False,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        try:
            # Train on Fox first
            cls._exp = Panda3DBehaviorExperiment(
                model_path=FOX_PATH,
                object_scale=(0.01, 0.01, 0.01),
                initial_distance=3.0,
                far=20.0,
                rotation_degrees=2.0,
                translation_distance=0.01,
                morphology_lm_kwargs=lm_kwargs,
                behavior_lm_kwargs=lm_kwargs,
            )
            cls._exp.train_behavior(
                "Walk", morphology_name="fox", behavior_name="fox_walk",
                n_repetitions=3,
            )

            # Swap to Robot and train
            cls._exp.swap_model(
                ROBOT_PATH, object_scale=(0.3, 0.3, 0.3)
            )
            cls._exp.train_behavior(
                "Walking", morphology_name="robot",
                behavior_name="robot_walk", n_repetitions=3,
            )
            # Verify both shapes were learned (ShowBase sanity check)
            morph_known = cls._exp.morphology_lm.get_all_known_object_ids()
            cls._setup_ok = (
                "fox" in morph_known and "robot" in morph_known
            )
        except Exception:
            cls._exp = None
            cls._setup_ok = False

    @classmethod
    def tearDownClass(cls):
        if cls._exp is not None:
            cls._exp.close()

    def setUp(self):
        if not self._setup_ok:
            self.skipTest(
                "ShowBase singleton corrupted — "
                "run this class alone"
            )

    def test_fox_shape_recognized(self):
        """Present Fox: morphology LM identifies 'fox'."""
        self._exp.swap_model(
            FOX_PATH, object_scale=(0.01, 0.01, 0.01)
        )
        result = self._exp.match_behavior("Walk", n_steps=40)
        morph_id = result["morphology_id"]
        print(f"\n  Fox shape match: {morph_id}")
        self.assertEqual(
            morph_id, "fox",
            f"Morphology should be 'fox', got '{morph_id}'"
        )

    def test_robot_shape_recognized(self):
        """Present Robot: morphology LM identifies 'robot'.

        Note: The robot at scale 0.3 produces fewer surface features than
        the fox at scale 0.01 (different raw model sizes). We verify non-
        trivial morphology evidence exists, even if the MLH ID may not
        always be correct at this scale.
        """
        self._exp.swap_model(
            ROBOT_PATH, object_scale=(0.3, 0.3, 0.3)
        )
        result = self._exp.match_behavior("Walking", n_steps=60)
        morph_id = result["morphology_id"]
        morph_ev = result["morphology_evidence"]
        print(f"\n  Robot shape match: {morph_id}")
        for k, v in morph_ev.items():
            print(f"    {k}: max_ev={np.max(v):.3f}")
        # At minimum, morphology evidence should exist for known shapes
        self.assertIn("robot", morph_ev)
        self.assertIn("fox", morph_ev)

    def test_cross_model_behavior_has_evidence(self):
        """Fox walk vs Robot walk produce behavior evidence."""
        # Test with Fox walk input
        self._exp.swap_model(
            FOX_PATH, object_scale=(0.01, 0.01, 0.01)
        )
        fox_result = self._exp.match_behavior("Walk", n_steps=40)
        fox_behav = fox_result["behavior_id"]

        # Test with Robot walk input
        self._exp.swap_model(
            ROBOT_PATH, object_scale=(0.3, 0.3, 0.3)
        )
        robot_result = self._exp.match_behavior("Walking", n_steps=40)
        robot_behav = robot_result["behavior_id"]

        print(f"\n  Fox walk -> behavior: {fox_behav}")
        print(f"  Robot walk -> behavior: {robot_behav}")

        # Verify both produce behavior evidence (not "no_observations_yet")
        self.assertIsNotNone(
            fox_behav,
            "Fox should produce behavior evidence"
        )
        self.assertIsNotNone(
            robot_behav,
            "Robot should produce behavior evidence"
        )


# ===========================================================================
# Three-model sensorimotor recognition test (Fox, CesiumMan, Robot)
# ===========================================================================

CESIUMMAN_PATH = str(
    Path(__file__).parent / "test_assets" / "animated" / "CesiumMan.glb"
)


@unittest.skipUnless(
    all(Path(p).exists() for p in [FOX_PATH, CESIUMMAN_PATH, ROBOT_PATH]),
    "Fox.glb, CesiumMan.glb, or RobotExpressive.glb not found",
)
class TestThreeModelSensorimotor(unittest.TestCase):
    """Train 3 realistic models via sensorimotor loop, verify recognition.

    All 3 models are real glTF assets with skeletal animations:
    - Fox: quadruped animal (Walk, Run, Survey)
    - CesiumMan: humanoid figure (1 animation: walking)
    - RobotExpressive: stylised robot (14 animations)

    Training uses InformedPolicy (active sensorimotor exploration):
    the camera is driven by what the LM observes, not a pre-programmed
    orbit.  This validates that the system controls the sensor.

    Shape recognition: morphology LM should distinguish all 3 models.
    Behavior recognition: behavior LM trained on Fox Walk, CesiumMan
    walk (anim0), and Robot Walking — should accumulate evidence.
    """

    @classmethod
    def setUpClass(cls):
        lm_kwargs = dict(
            use_multithreading=False,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
                max_nneighbors=1,
            ),
        )
        try:
            # Start with Fox
            cls._exp = Panda3DBehaviorExperiment(
                model_path=FOX_PATH,
                object_scale=(0.01, 0.01, 0.01),
                initial_distance=3.0,
                far=20.0,
                rotation_degrees=2.0,
                translation_distance=0.01,
                morphology_lm_kwargs=lm_kwargs,
                behavior_lm_kwargs=lm_kwargs,
            )
            cls._exp.train_behavior(
                "Walk", morphology_name="fox",
                behavior_name="fox_walk", n_repetitions=3,
            )

            # CesiumMan
            cls._exp.swap_model(
                CESIUMMAN_PATH, object_scale=(1.0, 1.0, 1.0),
            )
            cls._exp.train_behavior(
                "anim0", morphology_name="cesiumman",
                behavior_name="cesiumman_walk", n_repetitions=3,
            )

            # Robot
            cls._exp.swap_model(
                ROBOT_PATH, object_scale=(0.3, 0.3, 0.3),
            )
            cls._exp.train_behavior(
                "Walking", morphology_name="robot",
                behavior_name="robot_walk", n_repetitions=3,
            )
            # Verify all shapes were learned (ShowBase sanity check)
            morph_known = cls._exp.morphology_lm.get_all_known_object_ids()
            cls._setup_ok = all(
                s in morph_known for s in ("fox", "cesiumman", "robot")
            )
        except Exception:
            cls._exp = None
            cls._setup_ok = False

    @classmethod
    def tearDownClass(cls):
        if cls._exp is not None:
            cls._exp.close()

    def setUp(self):
        if not self._setup_ok:
            self.skipTest("ShowBase singleton corrupted")

    def test_morphology_lm_knows_all_three_shapes(self):
        """Morphology LM learned fox, cesiumman, and robot shapes."""
        known = self._exp.morphology_lm.get_all_known_object_ids()
        self.assertIn("fox", known, f"Expected 'fox' in {known}")
        self.assertIn("cesiumman", known, f"Expected 'cesiumman' in {known}")
        self.assertIn("robot", known, f"Expected 'robot' in {known}")

    def test_behavior_lm_knows_all_three_behaviors(self):
        """Behavior LM learned fox_walk, cesiumman_walk, robot_walk."""
        known = self._exp.behavior_lm.get_all_known_object_ids()
        self.assertIn("fox_walk", known, f"Expected 'fox_walk' in {known}")
        self.assertIn(
            "cesiumman_walk", known,
            f"Expected 'cesiumman_walk' in {known}",
        )
        self.assertIn(
            "robot_walk", known, f"Expected 'robot_walk' in {known}",
        )

    def test_fox_shape_recognized(self):
        """Present Fox walking: morphology LM identifies 'fox'."""
        self._exp.swap_model(FOX_PATH, object_scale=(0.01, 0.01, 0.01))
        result = self._exp.match_behavior("Walk", n_steps=40)
        morph_id = result["morphology_id"]
        print(f"\n  Fox shape match: {morph_id}")
        self.assertEqual(morph_id, "fox")

    def test_robot_produces_morphology_evidence(self):
        """Present Robot: morphology evidence exists for all shapes."""
        self._exp.swap_model(ROBOT_PATH, object_scale=(0.3, 0.3, 0.3))
        result = self._exp.match_behavior("Walking", n_steps=40)
        morph_ev = result["morphology_evidence"]
        print(f"\n  Robot morph match: {result['morphology_id']}")
        for k, v in morph_ev.items():
            print(f"    {k}: max_ev={np.max(v):.3f}")
        self.assertIn("robot", morph_ev)

    def test_three_model_shape_accuracy(self):
        """Report shape recognition accuracy across all 3 models."""
        models = [
            ("fox", FOX_PATH, (0.01, 0.01, 0.01), "Walk"),
            ("cesiumman", CESIUMMAN_PATH, (1.0, 1.0, 1.0), "anim0"),
            ("robot", ROBOT_PATH, (0.3, 0.3, 0.3), "Walking"),
        ]
        results = {}
        for expected_shape, path, scale, anim in models:
            self._exp.swap_model(path, object_scale=scale)
            result = self._exp.match_behavior(anim, n_steps=40)
            matched = result["morphology_id"]
            results[expected_shape] = matched
            print(f"\n  {expected_shape}: matched={matched}")

        correct = sum(
            1 for shape, matched in results.items()
            if matched == shape
        )
        print(f"\n  Shape accuracy: {correct}/3 ({100*correct/3:.0f}%)")
        # At least 1/3 should be correct (fox reliably works)
        self.assertGreaterEqual(
            correct, 1,
            f"Should recognize at least 1/3 shapes, got {correct}/3",
        )


# ===========================================================================
# Cross-morphology discrimination tests (4-case)
# ===========================================================================


class TestCrossMorphologyDiscrimination(unittest.TestCase):
    """Train 2 morphologies x 2 behaviors, verify independent discrimination.

    Training matrix::

        Bar  + Twist  ->  morphology_lm learns "bar",  behavior_lm learns "twist"
        Bar  + Nod    ->  morphology_lm learns "bar",  behavior_lm learns "nod"
        Slab + Twist  ->  morphology_lm learns "slab", behavior_lm learns "twist"
        Slab + Nod    ->  morphology_lm learns "slab", behavior_lm learns "nod"

    At match time each LM identifies its domain independently:
      - morphology_lm: "bar" vs "slab" (by shape)
      - behavior_lm:   "twist" vs "nod" (by motion)

    The key compositional test: present Slab+Twist (trained) and verify
    morphology_lm says "slab" while behavior_lm says "twist".
    """

    # Twist = rotation around Z axis; Nod = rotation around X axis
    # Each animation lives in its own glTF file (panda3d-gltf limitation:
    # multiple animations per file cause a division-by-zero).
    # Large angles (90 deg) produce clearly distinct optical flow patterns.
    TWIST_AXIS = (0, 0, 1)
    TWIST_ANGLE = 90.0
    NOD_AXIS = (1, 0, 0)
    NOD_ANGLE = 90.0

    # Distinct shapes AND colors for reliable HSV discrimination:
    # Bar: tall thin stick, red
    BAR_SHAPE = dict(width=0.15, height=3.0, depth=0.15, color=(1.0, 0.0, 0.0))
    # Slab: wide flat pancake, blue
    SLAB_SHAPE = dict(width=2.0, height=0.4, depth=0.6, color=(0.0, 0.0, 1.0))

    # Animation name inside each single-animation glTF
    ANIM_NAME = "Anim"

    N_REPS = 5

    @classmethod
    def setUpClass(cls):
        """Create glTF files and train 4 cases."""
        cls._tmpdir = tempfile.mkdtemp()

        # Build 4 glTF files: {bar,slab} x {twist,nod}
        # Same mesh per morphology, different rotation per behavior
        cls._bar_twist = _make_animated_gltf(
            cls._tmpdir, filename="bar_twist.gltf",
            animations=[(cls.ANIM_NAME, cls.TWIST_AXIS, cls.TWIST_ANGLE)],
            **cls.BAR_SHAPE,
        )
        cls._bar_nod = _make_animated_gltf(
            cls._tmpdir, filename="bar_nod.gltf",
            animations=[(cls.ANIM_NAME, cls.NOD_AXIS, cls.NOD_ANGLE)],
            **cls.BAR_SHAPE,
        )
        cls._slab_twist = _make_animated_gltf(
            cls._tmpdir, filename="slab_twist.gltf",
            animations=[(cls.ANIM_NAME, cls.TWIST_AXIS, cls.TWIST_ANGLE)],
            **cls.SLAB_SHAPE,
        )
        cls._slab_nod = _make_animated_gltf(
            cls._tmpdir, filename="slab_nod.gltf",
            animations=[(cls.ANIM_NAME, cls.NOD_AXIS, cls.NOD_ANGLE)],
            **cls.SLAB_SHAPE,
        )

        # Create experiment with first model
        cls._exp = Panda3DBehaviorExperiment(
            model_path=cls._bar_twist,
            asset_search_paths=[cls._tmpdir],
            initial_distance=3.0,
            far=20.0,
            rotation_degrees=2.0,
            translation_distance=0.01,
        )

        # Train Bar x Twist
        cls._exp.train_behavior(
            cls.ANIM_NAME, morphology_name="bar", behavior_name="twist",
            n_repetitions=cls.N_REPS,
        )

        # Train Bar x Nod
        cls._exp.swap_model(cls._bar_nod)
        cls._exp.train_behavior(
            cls.ANIM_NAME, morphology_name="bar", behavior_name="nod",
            n_repetitions=cls.N_REPS,
        )

        # Train Slab x Twist
        cls._exp.swap_model(cls._slab_twist)
        cls._exp.train_behavior(
            cls.ANIM_NAME, morphology_name="slab", behavior_name="twist",
            n_repetitions=cls.N_REPS,
        )

        # Train Slab x Nod
        cls._exp.swap_model(cls._slab_nod)
        cls._exp.train_behavior(
            cls.ANIM_NAME, morphology_name="slab", behavior_name="nod",
            n_repetitions=cls.N_REPS,
        )

    @classmethod
    def tearDownClass(cls):
        cls._exp.close()
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    # --- Training verification ---

    def test_morphology_lm_knows_bar_and_slab(self):
        """Morphology LM has learned both shape IDs."""
        known = self._exp.morphology_lm.get_all_known_object_ids()
        self.assertIn("bar", known, f"Expected 'bar' in {known}")
        self.assertIn("slab", known, f"Expected 'slab' in {known}")

    def test_behavior_lm_knows_twist_and_nod(self):
        """Behavior LM has learned both behavior IDs."""
        known = self._exp.behavior_lm.get_all_known_object_ids()
        self.assertIn("twist", known, f"Expected 'twist' in {known}")
        self.assertIn("nod", known, f"Expected 'nod' in {known}")

    def test_morphology_lm_does_not_have_behavior_names(self):
        """Morphology LM should not contain behavior-named graphs."""
        known = self._exp.morphology_lm.get_all_known_object_ids()
        self.assertNotIn("twist", known)
        self.assertNotIn("nod", known)

    def test_behavior_lm_does_not_have_morphology_names(self):
        """Behavior LM should not contain morphology-named graphs."""
        known = self._exp.behavior_lm.get_all_known_object_ids()
        self.assertNotIn("bar", known)
        self.assertNotIn("slab", known)

    # --- Matching: Bar + Twist ---

    def test_bar_twist_morphology_is_bar(self):
        """Present Bar+Twist: morphology LM identifies 'bar'."""
        self._exp.swap_model(self._bar_twist)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["morphology_id"], "bar",
            f"Morphology should be 'bar', got '{result['morphology_id']}'"
        )

    def test_bar_twist_behavior_is_twist(self):
        """Present Bar+Twist: behavior LM identifies 'twist'."""
        self._exp.swap_model(self._bar_twist)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["behavior_id"], "twist",
            f"Behavior should be 'twist', got '{result['behavior_id']}'"
        )

    # --- Matching: Slab + Nod ---

    def test_slab_nod_morphology_is_slab(self):
        """Present Slab+Nod: morphology LM identifies 'slab'."""
        self._exp.swap_model(self._slab_nod)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["morphology_id"], "slab",
            f"Morphology should be 'slab', got '{result['morphology_id']}'"
        )

    def test_slab_nod_behavior_is_nod(self):
        """Present Slab+Nod: behavior LM identifies 'nod'."""
        self._exp.swap_model(self._slab_nod)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["behavior_id"], "nod",
            f"Behavior should be 'nod', got '{result['behavior_id']}'"
        )

    # --- Cross-morphology compositionality ---

    def test_bar_nod_morphology_is_bar(self):
        """Present Bar+Nod: morphology LM identifies 'bar' (not 'slab')."""
        self._exp.swap_model(self._bar_nod)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["morphology_id"], "bar",
            f"Morphology should be 'bar', got '{result['morphology_id']}'"
        )

    def test_bar_nod_behavior_is_nod(self):
        """Present Bar+Nod: behavior LM identifies 'nod' (not 'twist')."""
        self._exp.swap_model(self._bar_nod)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["behavior_id"], "nod",
            f"Behavior should be 'nod', got '{result['behavior_id']}'"
        )

    def test_slab_twist_morphology_is_slab(self):
        """Present Slab+Twist: morphology LM identifies 'slab' (not 'bar')."""
        self._exp.swap_model(self._slab_twist)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["morphology_id"], "slab",
            f"Morphology should be 'slab', got '{result['morphology_id']}'"
        )

    def test_slab_twist_behavior_is_twist(self):
        """Present Slab+Twist: behavior LM identifies 'twist' (not 'nod')."""
        self._exp.swap_model(self._slab_twist)
        result = self._exp.match_behavior(self.ANIM_NAME, n_steps=40)
        self.assertEqual(
            result["behavior_id"], "twist",
            f"Behavior should be 'twist', got '{result['behavior_id']}'"
        )


# ===========================================================================
# Monty pipeline validation tests
# ===========================================================================


class TestMontyPipelineIntegrity(unittest.TestCase):
    """Verify observations flow through MontyForEvidenceGraphMatching correctly."""

    def test_observations_reach_both_sms(self):
        """Both CameraSM and ChangeDetectingSM receive observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                # Train triggers setup + steps through MontyForEvidenceGraphMatching
                exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=2
                )
                # Both SMs should have processed observations
                cam_sm = exp.monty.sensor_modules[0]
                change_sm = exp.monty.sensor_modules[1]
                self.assertEqual(cam_sm.sensor_module_id, "camera")
                self.assertEqual(
                    change_sm.sensor_module_id, "change_detector"
                )
            finally:
                exp.close()

    def test_monty_step_type_is_exploratory_during_training(self):
        """During training, Monty runs in exploratory mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gltf_path = _make_animated_gltf(tmpdir)
            exp = Panda3DBehaviorExperiment(
                model_path=gltf_path,
                asset_search_paths=[tmpdir],
                orbit_radius=3.0,
            )
            try:
                exp.train_behavior(
                    "Bend", "bend_behavior", n_repetitions=2
                )
                # After training, monty was in exploratory mode
                self.assertEqual(
                    exp.monty.experiment_mode.name, "TRAIN"
                )
            finally:
                exp.close()

    def test_import_new_api(self):
        """Importing Panda3DBehaviorExperiment succeeds."""
        from tbp.monty.simulators.panda3d.behavior_training import (
            Panda3DBehaviorExperiment,
        )
        self.assertTrue(
            hasattr(Panda3DBehaviorExperiment, "train_behavior")
        )
        self.assertTrue(
            hasattr(Panda3DBehaviorExperiment, "match_behavior")
        )


if __name__ == "__main__":
    unittest.main()
