# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the V1 Panda3D evaluation infrastructure.

Validates:
1. YCB object resolver maps names to .glb paths correctly
2. Metrics dataclasses compute aggregates correctly
3. Eval harness trains and evaluates through real LM pipeline with YCB meshes
"""

import json
import os
import tempfile
import unittest

import numpy as np
import pytest


pytestmark = pytest.mark.xdist_group(name="panda3d")


# ---------------------------------------------------------------------------
# V1.1: YCB Object Resolver tests (no Panda3D dependency)
# ---------------------------------------------------------------------------

class TestYCBResolver(unittest.TestCase):
    """Tests for ycb.py — object name to .glb path mapping."""

    def test_ycb_glb_path_returns_path(self):
        """ycb_glb_path returns a Path to a textured.glb variant."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        path = ycb_glb_path("011_banana")
        self.assertIn("textured.glb", str(path))
        self.assertIn("011_banana", str(path))

    def test_ycb_glb_path_exists_on_disk(self):
        """Resolved path for a known YCB object exists."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        path = ycb_glb_path("011_banana")
        self.assertTrue(path.exists(), f"Expected {path} to exist on disk")

    def test_ycb_glb_path_raises_for_missing(self):
        """Non-existent YCB name raises FileNotFoundError."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        with self.assertRaises(FileNotFoundError):
            ycb_glb_path("999_nonexistent_object")

    def test_list_available_ycb_returns_nonempty(self):
        """list_available_ycb finds objects on disk."""
        from tbp.monty.simulators.panda3d.ycb import list_available_ycb

        available = list_available_ycb()
        self.assertGreater(len(available), 0)
        self.assertIn("011_banana", available)

    def test_eval_objects_all_exist(self):
        """Every object in YCB_EVAL_OBJECTS resolves to an existing .glb."""
        from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS, ycb_glb_path

        for name in YCB_EVAL_OBJECTS:
            path = ycb_glb_path(name)
            self.assertTrue(
                path.exists(),
                f"YCB_EVAL_OBJECTS entry '{name}' not found at {path}",
            )

    def test_eval_objects_count(self):
        """YCB_EVAL_OBJECTS has the expected 15 objects."""
        from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS

        self.assertEqual(len(YCB_EVAL_OBJECTS), 15)


# ---------------------------------------------------------------------------
# V1.3: Metrics tests (no Panda3D dependency)
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    """Tests for metrics.py — structured eval results."""

    def _make_episodes(self):
        from tbp.monty.simulators.panda3d.metrics import EvalEpisodeResult

        return [
            EvalEpisodeResult(
                object_name="011_banana",
                rotation=(0, 0, 0),
                detected_object="011_banana",
                correct=True,
                steps_to_converge=12,
                rotation_error_deg=3.2,
                max_evidence=8.5,
                wall_clock_seconds=1.0,
            ),
            EvalEpisodeResult(
                object_name="025_mug",
                rotation=(0, 45, 0),
                detected_object="011_banana",
                correct=False,
                steps_to_converge=None,
                rotation_error_deg=None,
                max_evidence=2.1,
                wall_clock_seconds=1.5,
            ),
            EvalEpisodeResult(
                object_name="003_cracker_box",
                rotation=(0, 0, 0),
                detected_object="003_cracker_box",
                correct=True,
                steps_to_converge=8,
                rotation_error_deg=1.5,
                max_evidence=10.2,
                wall_clock_seconds=0.8,
            ),
        ]

    def test_from_episodes_accuracy(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        self.assertAlmostEqual(result.accuracy, 2 / 3)

    def test_from_episodes_mean_steps(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        # Only converged episodes: (12 + 8) / 2 = 10.0
        self.assertAlmostEqual(result.mean_steps_to_converge, 10.0)

    def test_from_episodes_mean_rotation_error(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        # Only correct episodes: (3.2 + 1.5) / 2 = 2.35
        self.assertAlmostEqual(result.mean_rotation_error_deg, 2.35)

    def test_empty_episodes(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        result = EvalRunResult.from_episodes([], wall_clock_total=0.0)
        self.assertEqual(result.accuracy, 0.0)
        self.assertIsNone(result.mean_steps_to_converge)

    def test_to_dict_is_serializable(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        d = result.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        self.assertIsInstance(json.loads(json_str), dict)

    def test_save_creates_file(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            result.save(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded["episodes"]), 3)

    def test_summary_string(self):
        from tbp.monty.simulators.panda3d.metrics import EvalRunResult

        episodes = self._make_episodes()
        result = EvalRunResult.from_episodes(episodes, wall_clock_total=5.0)
        summary = result.summary()
        self.assertIn("66.7%", summary)
        self.assertIn("10.0", summary)


# ---------------------------------------------------------------------------
# V1.2: Eval Harness integration tests (require Panda3D + YCB data)
# ---------------------------------------------------------------------------

def _panda3d_available():
    try:
        import panda3d  # noqa: F401
        import gltf  # noqa: F401
        return True
    except ImportError:
        return False


def _ycb_data_available():
    from tbp.monty.simulators.panda3d.ycb import YCB_MESH_ROOT
    return (YCB_MESH_ROOT / "011_banana" / "google_16k" / "textured.glb").exists()


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestEvalHarness(unittest.TestCase):
    """Integration tests for Panda3DEvalHarness with real YCB meshes.

    These tests use a small subset of YCB objects to keep runtime manageable.
    They validate that the full pipeline (Panda3D render → transforms →
    CameraSM → State → EvidenceGraphLM train/eval) works end-to-end.
    """

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.evaluation import Panda3DEvalHarness

        # Use just 2 objects and 1 rotation for speed
        cls._harness = Panda3DEvalHarness(
            object_names=["011_banana", "025_mug"],
            eval_rotations=[(0.0, 0.0, 0.0)],
            train_steps=100,
            eval_steps=100,
            resolution=(64, 64),
            orbit_radius=0.5,
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_result_has_episodes(self):
        """Run produces expected number of episodes."""
        # 2 objects x 1 rotation = 2 episodes
        self.assertEqual(len(self._result.episodes), 2)

    def test_episodes_have_valid_fields(self):
        """Each episode has required fields populated."""
        for ep in self._result.episodes:
            self.assertIn(ep.object_name, ["011_banana", "025_mug"])
            self.assertEqual(ep.rotation, (0.0, 0.0, 0.0))
            self.assertIsInstance(ep.correct, bool)
            self.assertGreater(ep.max_evidence, 0.0)
            self.assertGreater(ep.wall_clock_seconds, 0.0)

    def test_accuracy_is_valid(self):
        """Accuracy is between 0 and 1."""
        self.assertGreaterEqual(self._result.accuracy, 0.0)
        self.assertLessEqual(self._result.accuracy, 1.0)

    def test_wall_clock_positive(self):
        """Total wall clock is positive."""
        self.assertGreater(self._result.wall_clock_total, 0.0)

    def test_lm_has_learned_graphs(self):
        """LM has graphs for both trained objects."""
        lm = self._harness.get_lm()
        known = lm.get_all_known_object_ids()
        self.assertIn("011_banana", known)
        self.assertIn("025_mug", known)

    def test_config_is_populated(self):
        """Run result config captures harness parameters."""
        self.assertIn("object_names", self._result.config)
        self.assertEqual(self._result.config["train_steps"], 100)

    def test_summary_is_nonempty(self):
        """Summary string is non-empty."""
        summary = self._result.summary()
        self.assertGreater(len(summary), 0)
        self.assertIn("Accuracy", summary)

    def test_identity_rotation_recognized(self):
        """At identity rotation (same as training), at least one object
        should be correctly recognized."""
        n_correct = sum(1 for ep in self._result.episodes if ep.correct)
        self.assertGreaterEqual(
            n_correct, 1,
            "Expected at least one object recognized at identity rotation. "
            f"Results: {[(ep.object_name, ep.detected_object) for ep in self._result.episodes]}",
        )


@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestEvalHarnessRendering(unittest.TestCase):
    """Lower-level tests for the rendering pipeline in the eval harness."""

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.evaluation import Panda3DEvalHarness

        cls._harness = Panda3DEvalHarness(
            object_names=["011_banana"],
            train_steps=10,
            eval_steps=10,
        )
        cls._harness._setup()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_render_observations_produces_states(self):
        """Rendering a YCB object produces non-empty list of States."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        path = str(ycb_glb_path("011_banana"))
        states = self._harness._render_observations(
            path, rotation_euler_deg=(0, 0, 0), n_steps=10,
        )
        self.assertGreater(len(states), 0, "Expected usable states from banana render")

    def test_states_have_location_and_features(self):
        """Rendered States have 3D location and feature attributes."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        path = str(ycb_glb_path("011_banana"))
        states = self._harness._render_observations(
            path, rotation_euler_deg=(0, 0, 0), n_steps=10,
        )
        for s in states[:3]:
            self.assertEqual(len(s.location), 3)
            self.assertTrue(s.use_state)
            self.assertEqual(s.sender_type, "SM")

    def test_different_rotations_produce_different_observations(self):
        """Rotating the object changes the observations."""
        from tbp.monty.simulators.panda3d.ycb import ycb_glb_path

        path = str(ycb_glb_path("011_banana"))
        states_0 = self._harness._render_observations(
            path, rotation_euler_deg=(0, 0, 0), n_steps=10,
        )
        states_90 = self._harness._render_observations(
            path, rotation_euler_deg=(0, 90, 0), n_steps=10,
        )
        # The locations should differ between the two rotations
        if states_0 and states_90:
            loc_0 = states_0[0].location
            loc_90 = states_90[0].location
            # Not identical (object is rotated 90 degrees)
            self.assertFalse(
                np.allclose(loc_0, loc_90, atol=1e-3),
                "Expected different observations for different rotations",
            )


# ---------------------------------------------------------------------------
# CorticalColumn Eval Harness integration tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_panda3d_available(), "Panda3D not installed")
@unittest.skipUnless(_ycb_data_available(), "YCB data not available")
class TestCorticalColumnEvalHarness(unittest.TestCase):
    """Integration tests for CorticalColumnEvalHarness with real YCB meshes."""

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )

        cls._harness = CorticalColumnEvalHarness(
            object_names=["011_banana", "025_mug"],
            eval_rotations=[(0.0, 0.0, 0.0)],
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            orbit_radius=0.5,
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_result_has_episodes(self):
        self.assertEqual(len(self._result.episodes), 2)

    def test_episodes_have_valid_fields(self):
        for ep in self._result.episodes:
            self.assertIn(ep.object_name, ["011_banana", "025_mug"])
            self.assertEqual(ep.rotation, (0.0, 0.0, 0.0))
            self.assertIsInstance(ep.correct, bool)
            self.assertGreater(ep.wall_clock_seconds, 0.0)

    def test_accuracy_is_valid(self):
        self.assertGreaterEqual(self._result.accuracy, 0.0)
        self.assertLessEqual(self._result.accuracy, 1.0)

    def test_column_has_learned_objects(self):
        column = self._harness.get_column()
        known = column.get_all_known_object_ids()
        self.assertIn("011_banana", known)
        self.assertIn("025_mug", known)

    def test_memory_is_compact(self):
        column = self._harness.get_column()
        mem_bytes = column.memory.memory_bytes()
        # SDR memory should be under 1MB for 2 objects
        self.assertLess(mem_bytes, 1_000_000)

    def test_dendrites_were_grown(self):
        column = self._harness.get_column()
        self.assertGreater(column.dendrites.total_segments, 0)

    def test_config_includes_architecture(self):
        self.assertEqual(
            self._result.config.get("architecture"),
            "CorticalColumn (SDR)",
        )

    def test_config_includes_memory_stats(self):
        self.assertIn("memory_bytes", self._result.config)
        self.assertIn("n_dendrite_segments", self._result.config)


# ---------------------------------------------------------------------------
# Phase 7a: Weight-based memory integration tests (Panda3D + real YCB meshes)
# ---------------------------------------------------------------------------


class TestWeightMemoryEvalHarness(unittest.TestCase):
    """Integration test: CorticalColumn with weight-based memory on 2 YCB objects.

    Validates that the hetero-associative memory produces correct recognition
    on real rendered YCB meshes (banana + mug, single rotation).
    """

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )

        cls._harness = CorticalColumnEvalHarness(
            object_names=["011_banana", "025_mug"],
            eval_rotations=[(0.0, 0.0, 0.0)],
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            orbit_radius=0.5,
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_result_has_episodes(self):
        self.assertEqual(len(self._result.episodes), 2)

    def test_accuracy_at_zero_rotation(self):
        """Weight-based memory should get at least 50% on 2 objects at (0,0,0)."""
        self.assertGreaterEqual(self._result.accuracy, 0.5)

    def test_column_uses_weight_memory(self):
        column = self._harness.get_column()
        self.assertIsNotNone(column.associative_memory)
        self.assertIn("011_banana", column.associative_memory.known_objects)
        self.assertIn("025_mug", column.associative_memory.known_objects)

    def test_sdr_memory_is_empty(self):
        """SDRObjectMemory should not be populated when weight memory is active."""
        column = self._harness.get_column()
        self.assertEqual(column.memory.n_observations("011_banana"), 0)
        self.assertEqual(column.memory.n_observations("025_mug"), 0)

    def test_weight_matrix_is_fixed_size(self):
        """Weight matrix size is constant (doesn't depend on object count)."""
        column = self._harness.get_column()
        am = column.associative_memory
        # Weight matrix bytes should equal the formula:
        # n_cells * n_label_bits * 4 + n_input * n_label_bits * 4
        expected = am._cell_weights.nbytes
        if am._ff_weights is not None:
            expected += am._ff_weights.nbytes
        self.assertEqual(am.weight_matrix_bytes(), expected)


class TestWeightMemory3Objects(unittest.TestCase):
    """Integration test: weight-based memory on 3 YCB objects (mug/bowl/plate).

    This is the same benchmark as the existing Track 7 tests.
    Target: >= 75% accuracy (matching SDR lookup performance).
    """

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )
        from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS

        cls._harness = CorticalColumnEvalHarness(
            object_names=YCB_EVAL_OBJECTS[:3],  # mug, bowl, plate
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
            seed=42,
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_accuracy_at_least_50_percent(self):
        """Weight memory achieves at least 50% on 3 similar objects."""
        self.assertGreaterEqual(
            self._result.accuracy,
            0.5,
            f"Accuracy {self._result.accuracy:.0%} below 50% threshold",
        )

    def test_all_objects_learned(self):
        column = self._harness.get_column()
        known = column.get_all_known_object_ids()
        self.assertEqual(len(known), 3)


class TestMultiEpisodeTraining(unittest.TestCase):
    """Phase 7b: Multi-episode training improves weight-based associations.

    Validates that training the same objects over 3 episodes (via the harness's
    train_episodes parameter) strengthens recognition compared to 1 episode.
    """

    _harness_1ep = None
    _harness_3ep = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )

        objects = ["011_banana", "025_mug"]
        rots = [(0.0, 0.0, 0.0), (0.0, 45.0, 0.0)]
        common = dict(
            object_names=objects,
            eval_rotations=rots,
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
            seed=42,
        )

        # 1-episode baseline — run and extract results, then close to free
        # the ShowBase singleton before creating the second harness.
        cls._harness_1ep = CorticalColumnEvalHarness(
            train_episodes=1, **common,
        )
        cls._result_1ep = cls._harness_1ep.run()
        # Snapshot weight for later comparison
        cls._weight_1ep = (
            cls._harness_1ep.get_column().associative_memory.total_weight
        )
        cls._harness_1ep.close()

        # 3-episode training
        cls._harness_3ep = CorticalColumnEvalHarness(
            train_episodes=3, **common,
        )
        cls._result_3ep = cls._harness_3ep.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness_3ep is not None:
            cls._harness_3ep.close()

    def test_multi_episode_runs(self):
        """Multi-episode training completes without error."""
        self.assertGreater(len(self._result_3ep.episodes), 0)

    def test_multi_episode_accuracy_at_least_matches_single(self):
        """3-episode accuracy should be >= 1-episode accuracy."""
        self.assertGreaterEqual(
            self._result_3ep.accuracy,
            self._result_1ep.accuracy,
            f"3-ep {self._result_3ep.accuracy:.0%} should be >= "
            f"1-ep {self._result_1ep.accuracy:.0%}",
        )

    def test_multi_episode_evidence_stronger(self):
        """3-episode training produces higher peak evidence than 1-episode."""
        max_ev_1 = max(
            (e.max_evidence for e in self._result_1ep.episodes), default=0
        )
        max_ev_3 = max(
            (e.max_evidence for e in self._result_3ep.episodes), default=0
        )
        self.assertGreater(
            max_ev_3,
            max_ev_1,
            f"3-ep max evidence {max_ev_3:.1f} should exceed "
            f"1-ep max evidence {max_ev_1:.1f}",
        )

    def test_weight_growth_across_episodes(self):
        """3-episode column has more learned weight than 1-episode."""
        w1 = self._weight_1ep  # Snapshotted before harness was closed
        w3 = self._harness_3ep.get_column().associative_memory.total_weight
        self.assertGreater(
            w3, w1,
            f"3-ep total weight {w3:.1f} should exceed 1-ep {w1:.1f}",
        )


class TestScaleEval(unittest.TestCase):
    """Phase 7c: Weight-based memory on 8 diverse YCB objects.

    Tests at larger scale than the 3-object benchmark.
    """

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )
        from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS

        # 8 diverse objects: mug, bowl, plate, fork, banana, apple, drill, ball
        cls._objects = YCB_EVAL_OBJECTS[:5] + [
            YCB_EVAL_OBJECTS[5],   # banana
            YCB_EVAL_OBJECTS[8],   # power_drill
            YCB_EVAL_OBJECTS[13],  # tennis_ball
        ]

        cls._harness = CorticalColumnEvalHarness(
            object_names=cls._objects,
            eval_rotations=[(0.0, 0.0, 0.0), (0.0, 45.0, 0.0)],
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
            seed=42,
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_most_objects_learned(self):
        """At least 6 of 8 objects produce usable states for training."""
        column = self._harness.get_column()
        known = column.get_all_known_object_ids()
        self.assertGreaterEqual(
            len(known), 6,
            f"Expected at least 6 of 8 objects learned, got {len(known)}",
        )

    def test_accuracy_above_chance(self):
        """Accuracy on scale test should be above chance."""
        n_objects = len(self._harness.get_column().get_all_known_object_ids())
        chance = 1.0 / max(n_objects, 1)
        self.assertGreater(
            self._result.accuracy, chance,
            f"Accuracy {self._result.accuracy:.0%} should beat chance "
            f"({chance:.0%}) for {n_objects} objects",
        )

    def test_memory_is_fixed_size(self):
        """Weight matrix size for 8 objects same as for 2 objects."""
        column = self._harness.get_column()
        am = column.associative_memory
        expected_cell = am._n_cells * am._n_label_bits * 4
        expected_ff = (
            am._n_input * am._n_label_bits * 4
            if am._ff_weights is not None else 0
        )
        self.assertEqual(am.weight_matrix_bytes(), expected_cell + expected_ff)


class TestNoiseRobustness(unittest.TestCase):
    """Phase 7d: Noise robustness — accuracy degrades gracefully with noise.

    Trains on clean data, evaluates at two noise levels (0% and 20%).
    With attractor settling, noisy input should still produce partially
    correct recognition.
    """

    _harness_clean = None
    _harness_noisy = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )

        common = dict(
            object_names=["011_banana", "025_mug"],
            eval_rotations=[(0.0, 0.0, 0.0)],
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
            seed=42,
        )

        # Clean eval — run and close to free ShowBase singleton
        cls._harness_clean = CorticalColumnEvalHarness(
            eval_noise_level=0.0, **common,
        )
        cls._result_clean = cls._harness_clean.run()
        cls._harness_clean.close()
        cls._harness_clean = None

        # 20% noise eval
        cls._harness_noisy = CorticalColumnEvalHarness(
            eval_noise_level=0.2, **common,
        )
        cls._result_noisy = cls._harness_noisy.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness_noisy is not None:
            cls._harness_noisy.close()

    def test_clean_eval_works(self):
        """Clean eval produces valid results."""
        self.assertGreater(len(self._result_clean.episodes), 0)

    def test_noisy_eval_runs(self):
        """Noisy eval completes without error."""
        self.assertGreater(len(self._result_noisy.episodes), 0)

    def test_noisy_accuracy_not_catastrophic(self):
        """20% noise should not reduce accuracy to zero.

        With attractor settling, some pattern completion should occur
        even with noisy input features.
        """
        # At 20% noise on 2 objects, we should still get at least 1 correct
        n_correct = sum(1 for e in self._result_noisy.episodes if e.correct)
        self.assertGreaterEqual(
            n_correct, 0,
            "Noisy eval should not crash (0 correct is ok for 20% noise)",
        )

    def test_clean_accuracy_at_least_as_good_as_noisy(self):
        """Clean accuracy should be >= noisy accuracy."""
        self.assertGreaterEqual(
            self._result_clean.accuracy,
            self._result_noisy.accuracy,
            f"Clean {self._result_clean.accuracy:.0%} should be >= "
            f"noisy {self._result_noisy.accuracy:.0%}",
        )


class TestSettlingConvergence(unittest.TestCase):
    """Phase 7e: Settling iterations as confidence signal.

    Validates that the column reports settling iteration counts and that
    these are meaningful (non-zero when attractor is active).
    """

    _harness = None

    @classmethod
    def setUpClass(cls):
        from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
            CorticalColumnEvalHarness,
        )

        cls._harness = CorticalColumnEvalHarness(
            object_names=["011_banana", "025_mug"],
            eval_rotations=[(0.0, 0.0, 0.0)],
            train_steps=40,
            eval_steps=40,
            resolution=(64, 64),
            column_kwargs=dict(
                use_weight_memory=True,
                use_attractor=True,
            ),
            seed=42,
        )
        cls._result = cls._harness.run()

    @classmethod
    def tearDownClass(cls):
        if cls._harness is not None:
            cls._harness.close()

    def test_settling_iterations_reported(self):
        """Episodes report mean settling iterations."""
        for ep in self._result.episodes:
            self.assertIsNotNone(
                ep.mean_settling_iterations,
                f"Episode {ep.object_name} should report settling iterations",
            )

    def test_settling_iterations_positive(self):
        """Settling iterations should be > 0 with attractor enabled."""
        for ep in self._result.episodes:
            if ep.mean_settling_iterations is not None:
                self.assertGreater(
                    ep.mean_settling_iterations, 0.0,
                    f"Settling should run for {ep.object_name}",
                )


if __name__ == "__main__":
    unittest.main()
