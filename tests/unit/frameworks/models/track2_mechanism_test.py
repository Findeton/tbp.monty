"""Track 2 mechanism validation tests (Tier 1).

Tests 1-4 validate the HPC->evidence modulation pathway in isolation
using synthetic data. No Habitat, no training runs, no eval runs.
Run time: seconds.

See docs/track-2-temporal-world-model.md for context.
"""

import copy
import unittest

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.hippocampal_module import HippocampalModule
from tbp.monty.frameworks.models.states import State
from tests.unit.resources.unit_test_utils import BaseGraphTest


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from hippocampal_test.py)
# ---------------------------------------------------------------------------


def _make_ctx(global_step=0, episode_step=0):
    """Create a minimal RuntimeContext with a fake timer."""

    class _FakeTimer:
        pass

    t = _FakeTimer()
    t.global_step = global_step
    t.episode_step = episode_step
    rng = np.random.RandomState(42)
    return RuntimeContext(rng=rng, timer=t)


def _make_hpc_state(obj_id, confidence=0.9, location=None, use_state=True):
    """Create a minimal mock State for HPC input."""

    class _S:
        pass

    s = _S()
    s.use_state = use_state
    s.confidence = confidence
    s.non_morphological_features = {"object_id": obj_id}
    s.location = (
        location if location is not None else np.array([0.0, 0.0, 0.0])
    )
    s.sender_id = "lm_0"
    s.sender_type = "LM"
    return s


# ---------------------------------------------------------------------------
# Test 1: HPC association strengths bias evidence in EvidenceGraphLM
# ---------------------------------------------------------------------------


class TestHPCAssociationBiasesEvidence(BaseGraphTest):
    """Test 1: HPC context with strong association boosts target evidence.

    Creates an EvidenceGraphLM with two graphs, sets equal evidence via
    matching, creates an HPC context signal with strong association for one
    graph, calls receive_context(), verifies that graph's evidence is boosted
    and the other's is not.
    """

    def _make_lm_with_two_objects(self):
        """Create EvidenceGraphLM with two trained objects and category bias."""
        taxonomy = {
            "ball_obj": "ball",
            "fruit_obj": "fruit",
        }

        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            max_graph_size=10,
            category_taxonomy=taxonomy,
            category_bias_strength=0.5,
        )

        # Train first object (ball)
        lm.mode = ExperimentMode.TRAIN
        for obs in self.fake_obs_learn:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "ball_obj"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        # Train second object (fruit) - use different geometry
        target = {"object": "fruit_obj", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        for obs in self.fake_obs_symmetric:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "fruit_obj"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        self.assertEqual(
            len(lm.get_all_known_object_ids()), 2,
            "Should have two objects in memory",
        )
        return lm

    def test_hpc_association_biases_evidence(self):
        """When HPC has strong ball association, ball evidence is boosted."""
        lm = self._make_lm_with_two_objects()

        # Switch to eval and run matching to initialize evidence
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)
        lm.matching_step(self.ctx, [self.fake_obs_learn[0]])

        # Record evidence before HPC bias
        ball_ev_before = lm.evidence["ball_obj"].copy()
        fruit_ev_before = lm.evidence["fruit_obj"].copy()

        # Create HPC context with strong ball association
        context = {
            "context_vector": np.zeros(32),
            "active_concepts": ["ball_concept"],
            "association_strengths": {
                "ball_obj": 0.8,  # Direct match to graph ID
            },
            "episode_count": 5,
        }

        lm.receive_context(**context)

        ball_ev_after = lm.evidence["ball_obj"]
        fruit_ev_after = lm.evidence["fruit_obj"]

        # Ball should have increased evidence
        self.assertTrue(
            np.all(ball_ev_after > ball_ev_before),
            f"Ball evidence should be boosted by HPC bias. "
            f"Max before: {np.max(ball_ev_before):.4f}, "
            f"Max after: {np.max(ball_ev_after):.4f}",
        )

        # Fruit should NOT have increased evidence
        np.testing.assert_array_equal(
            fruit_ev_after,
            fruit_ev_before,
            "Fruit evidence should NOT be boosted when only ball has association",
        )


# ---------------------------------------------------------------------------
# Test 2: Cross-episode association accumulates in HPC
# ---------------------------------------------------------------------------


class TestCrossEpisodeAssociationAccumulates(unittest.TestCase):
    """Test 2: HPC co-occurrence associations build across episodes.

    Episode 1: ball + bouncy co-active.
    Episode 2: fruit + edible co-active.
    Verify distinct, non-zero associations and no cross-contamination.
    """

    def test_cross_episode_association_accumulates(self):
        hpc = HippocampalModule(context_dim=16, max_episodes=100)
        ctx = _make_ctx()

        # Episode 1: ball + bouncy
        hpc.pre_episode()
        hpc.matching_step(ctx, [
            _make_hpc_state("ball", confidence=0.9),
            _make_hpc_state("bouncy", confidence=0.9),
        ])
        hpc.post_episode()

        # Episode 2: fruit + edible
        hpc.pre_episode()
        hpc.matching_step(ctx, [
            _make_hpc_state("fruit", confidence=0.9),
            _make_hpc_state("edible", confidence=0.9),
        ])
        hpc.post_episode()

        # Verify ball<->bouncy association
        ball_assoc = hpc.recall_associations("ball")
        self.assertIn("bouncy", ball_assoc)
        self.assertGreater(
            ball_assoc["bouncy"], 0.0,
            "ball should have non-zero association with bouncy",
        )

        # Verify fruit<->edible association
        fruit_assoc = hpc.recall_associations("fruit")
        self.assertIn("edible", fruit_assoc)
        self.assertGreater(
            fruit_assoc["edible"], 0.0,
            "fruit should have non-zero association with edible",
        )

        # ball should have zero association with edible (never co-occurred)
        self.assertAlmostEqual(
            ball_assoc.get("edible", 0.0), 0.0, places=5,
            msg="ball should have zero association with edible",
        )

        # fruit should have zero association with bouncy
        self.assertAlmostEqual(
            fruit_assoc.get("bouncy", 0.0), 0.0, places=5,
            msg="fruit should have zero association with bouncy",
        )

        # Episodic memory should have 2 episodes
        self.assertEqual(len(hpc.episodic_memory), 2)

    def test_association_strengthens_with_repetition(self):
        """Repeated co-occurrence should not decrease association strength."""
        hpc = HippocampalModule(context_dim=16, max_episodes=100)
        ctx = _make_ctx()

        # See ball+bouncy once
        hpc.pre_episode()
        hpc.matching_step(ctx, [
            _make_hpc_state("ball"),
            _make_hpc_state("bouncy"),
        ])
        hpc.post_episode()

        strength_after_1 = hpc.recall_associations("ball").get("bouncy", 0)

        # See ball+bouncy two more times
        for _ in range(2):
            hpc.pre_episode()
            hpc.matching_step(ctx, [
                _make_hpc_state("ball"),
                _make_hpc_state("bouncy"),
            ])
            hpc.post_episode()

        strength_after_3 = hpc.recall_associations("ball").get("bouncy", 0)

        # Cosine normalization saturates at 1.0, but should never decrease
        self.assertGreaterEqual(
            strength_after_3, strength_after_1,
            "Association should not decrease with more co-occurrences",
        )


# ---------------------------------------------------------------------------
# Test 3: Dynamic feature registration enables non-geometric discrimination
# ---------------------------------------------------------------------------


class TestDynamicFeatureBreaksTie(BaseGraphTest):
    """Test 3: material_class feature breaks tie between identical geometries.

    Register a "material_class" feature via register_dynamic_feature().
    Train two objects with same geometry but different material_class.
    In eval, present observation matching one object's material_class.
    Verify different evidence accumulation.
    """

    def _make_obs_with_material(self, base_obs, material_class):
        """Clone a State and add material_class to non_morphological_features."""
        obs_args = dict(
            location=base_obs.location.copy(),
            morphological_features={
                "pose_vectors": (
                    base_obs.morphological_features["pose_vectors"].copy()
                ),
                "pose_fully_defined": (
                    base_obs.morphological_features["pose_fully_defined"]
                ),
                "on_object": base_obs.morphological_features.get("on_object", 1),
            },
            non_morphological_features={
                **{
                    k: (v.copy() if hasattr(v, "copy") else v)
                    for k, v in base_obs.non_morphological_features.items()
                },
                "material_class": np.array([material_class]),
            },
            confidence=base_obs.confidence,
            use_state=base_obs.use_state,
            sender_id=base_obs.sender_id,
            sender_type=base_obs.sender_type,
        )
        return State(**obs_args)

    def test_dynamic_feature_breaks_tie(self):
        """Objects with same geometry but different material_class diverge."""
        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                    "material_class": [0.3],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                    "material_class": np.array([2.0]),
                }
            },
            max_graph_size=10,
        )

        # Train object 0 with material_class=0.0 (hard/ball)
        obs_hard = [
            self._make_obs_with_material(o, 0.0) for o in self.fake_obs_learn
        ]
        lm.mode = ExperimentMode.TRAIN
        for obs in obs_hard:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "hard_object"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        # Train object 1 with material_class=1.0 (soft/fruit), same geometry
        target = {"object": "soft_object", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        obs_soft = [
            self._make_obs_with_material(o, 1.0) for o in self.fake_obs_learn
        ]
        for obs in obs_soft:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "soft_object"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        self.assertEqual(len(lm.get_all_known_object_ids()), 2)

        # Eval: present observation with material_class=0.0 (matches hard)
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)

        test_obs = self._make_obs_with_material(self.fake_obs_learn[0], 0.0)
        lm.matching_step(self.ctx, [test_obs])

        hard_ev = np.max(lm.evidence.get("hard_object", np.array([0])))
        soft_ev = np.max(lm.evidence.get("soft_object", np.array([0])))

        self.assertGreater(
            hard_ev, soft_ev,
            f"Hard object (matching material_class=0) should have more "
            f"evidence than soft object. Hard: {hard_ev:.4f}, Soft: {soft_ev:.4f}",
        )

    def test_register_dynamic_feature_api(self):
        """register_dynamic_feature() creates valid tolerance and weight entries."""
        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=10,
        )

        lm.register_dynamic_feature(
            "patch", "behavioral_tag", dim=3, weight=1.5, tolerance=0.25,
        )

        self.assertIn("behavioral_tag", lm.tolerances["patch"])
        self.assertIn("behavioral_tag", lm.feature_weights["patch"])
        np.testing.assert_array_almost_equal(
            lm.tolerances["patch"]["behavioral_tag"],
            np.array([0.25, 0.25, 0.25]),
        )
        np.testing.assert_array_almost_equal(
            lm.feature_weights["patch"]["behavioral_tag"],
            np.array([1.5, 1.5, 1.5]),
        )

    def test_set_feature_weight_api(self):
        """set_feature_weight() updates weight for an existing feature."""
        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=10,
        )

        lm.set_feature_weight("patch", "hsv", np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(
            lm.feature_weights["patch"]["hsv"],
            np.array([0, 0, 0]),
        )

        # Scalar weight broadcasts
        lm.set_feature_weight("patch", "hsv", 2.0)
        self.assertEqual(lm.feature_weights["patch"]["hsv"][0], 2.0)


# ---------------------------------------------------------------------------
# Test 4: HPC context resolves geometric ambiguity (core mechanism test)
# ---------------------------------------------------------------------------


class TestCategoryContextResolvesGeometricAmbiguity(BaseGraphTest):
    """Test 4: HPC context breaks tie between geometrically identical objects.

    This is the core mechanism test simulating the ball=fruit case.
    Without HPC context: both graphs have equal evidence.
    With HPC context: the associated graph wins.
    """

    def _make_lm_with_identical_objects(self):
        """Create two objects with IDENTICAL geometry (simulates ball=fruit)."""
        taxonomy = {
            "ball_graph": "ball",
            "fruit_graph": "fruit",
        }

        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            max_graph_size=10,
            category_taxonomy=taxonomy,
            category_bias_strength=0.5,
        )

        # Train ball_graph
        lm.mode = ExperimentMode.TRAIN
        for obs in self.fake_obs_learn:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "ball_graph"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        # Train fruit_graph with SAME observations (identical geometry)
        target = {"object": "fruit_graph", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        for obs in self.fake_obs_learn:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "fruit_graph"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        self.assertEqual(len(lm.get_all_known_object_ids()), 2)
        return lm

    def test_without_hpc_equal_evidence(self):
        """Without HPC context, identical geometry -> equal evidence."""
        lm = self._make_lm_with_identical_objects()

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)

        # Run matching steps with the training observations
        for obs in self.fake_obs_learn[:2]:
            lm.matching_step(self.ctx, [obs])

        ball_ev = np.max(lm.evidence.get("ball_graph", np.array([0])))
        fruit_ev = np.max(lm.evidence.get("fruit_graph", np.array([0])))

        self.assertAlmostEqual(
            ball_ev, fruit_ev, places=3,
            msg=(
                f"Without HPC context, evidence should be equal. "
                f"Ball: {ball_ev:.4f}, Fruit: {fruit_ev:.4f}"
            ),
        )

    def test_with_hpc_context_resolves_ambiguity(self):
        """With HPC context, the associated graph wins the tie."""
        lm = self._make_lm_with_identical_objects()

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)

        for obs in self.fake_obs_learn[:2]:
            lm.matching_step(self.ctx, [obs])

        # Record baseline evidence (should be equal)
        ball_ev_before = np.max(lm.evidence["ball_graph"])
        fruit_ev_before = np.max(lm.evidence["fruit_graph"])
        self.assertAlmostEqual(ball_ev_before, fruit_ev_before, places=3)

        # Apply HPC context favoring ball
        context = {
            "context_vector": np.zeros(32),
            "active_concepts": ["bouncy_concept"],
            "association_strengths": {
                "ball_graph": 0.9,  # Strong direct association
            },
            "episode_count": 10,
        }
        lm.receive_context(**context)

        # Ball should now have more evidence
        ball_ev_after = np.max(lm.evidence["ball_graph"])
        fruit_ev_after = np.max(lm.evidence["fruit_graph"])

        self.assertGreater(
            ball_ev_after, fruit_ev_after,
            f"With HPC context favoring ball, ball should exceed fruit. "
            f"Ball: {ball_ev_after:.4f}, Fruit: {fruit_ev_after:.4f}",
        )

        # Verify margin is at least category_bias_strength * association
        expected_min_margin = 0.5 * 0.9  # 0.45
        actual_margin = ball_ev_after - fruit_ev_after
        self.assertGreaterEqual(
            actual_margin, expected_min_margin * 0.9,
            f"Evidence margin ({actual_margin:.4f}) should be close to "
            f"expected ({expected_min_margin:.4f})",
        )

    def test_category_match_biases_unseen_instances(self):
        """HPC association with one ball biases ALL balls via category taxonomy."""
        lm = self._make_lm_with_identical_objects()

        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)

        for obs in self.fake_obs_learn[:2]:
            lm.matching_step(self.ctx, [obs])

        # HPC associates with a DIFFERENT ball concept (not ball_graph itself)
        # but shares the "ball" category via taxonomy
        lm.category_taxonomy["other_ball_id"] = "ball"

        context = {
            "context_vector": np.zeros(32),
            "active_concepts": ["other_ball"],
            "association_strengths": {
                "other_ball_id": 0.7,  # Not in graph memory, but in taxonomy
            },
            "episode_count": 5,
        }
        lm.receive_context(**context)

        # ball_graph should be boosted via category match
        ball_ev = np.max(lm.evidence["ball_graph"])
        fruit_ev = np.max(lm.evidence["fruit_graph"])

        self.assertGreater(
            ball_ev, fruit_ev,
            f"Category match should bias ball_graph via shared 'ball' category. "
            f"Ball: {ball_ev:.4f}, Fruit: {fruit_ev:.4f}",
        )


# ---------------------------------------------------------------------------
# Test 5 (bonus): HPC produces valid context signals after learning
# ---------------------------------------------------------------------------


class TestHPCContextSignalGeneration(unittest.TestCase):
    """Verify HPC produces usable context signals for downstream LMs."""

    def test_learned_association_produces_context_signal(self):
        """After co-occurrence learning, HPC produces a valid context signal."""
        hpc = HippocampalModule(context_dim=16)
        ctx = _make_ctx()

        # Learn ball+bouncy co-occurrence
        hpc.pre_episode()
        hpc.matching_step(ctx, [
            _make_hpc_state("ball"),
            _make_hpc_state("bouncy"),
        ])
        hpc.post_episode()

        # New episode: present ball alone
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball")])

        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal, "HPC should produce context after seeing ball")
        self.assertIn("association_strengths", signal)
        self.assertIn("active_concepts", signal)
        self.assertIn("ball", signal["active_concepts"])
        self.assertIn("context_vector", signal)
        self.assertEqual(signal["context_vector"].shape, (16,))

    def test_context_signal_contains_association_strengths(self):
        """Context signal has non-zero association_strengths for known concepts."""
        hpc = HippocampalModule(context_dim=16)
        ctx = _make_ctx()

        # Build associations across multiple episodes
        for _ in range(3):
            hpc.pre_episode()
            hpc.matching_step(ctx, [
                _make_hpc_state("ball"),
                _make_hpc_state("bouncy"),
            ])
            hpc.post_episode()

        # Present ball - should have non-zero association strength
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball")])

        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)

        # association_strengths maps active concept -> mean strength
        strengths = signal["association_strengths"]
        self.assertIn("ball", strengths)
        self.assertGreater(
            strengths["ball"], 0.0,
            "Ball should have non-zero association strength after co-occurrences",
        )


# ---------------------------------------------------------------------------
# Test 6: Multi-episode learning (Phase 4 - full cross-episode pipeline)
# ---------------------------------------------------------------------------


class TestMultiEpisodeCrossEpisodeLearning(BaseGraphTest):
    """Test 6: HPC learns associations naturally, then biases subsequent episodes.

    This is the core Phase 4 test: run multiple episodes where HPC
    accumulates associations, then verify that later episodes benefit
    from earlier experience.
    """

    def _make_lm_with_identical_objects(self):
        """Same as Test 4: two objects with identical geometry."""
        taxonomy = {
            "ball_graph": "ball",
            "fruit_graph": "fruit",
        }

        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={
                "patch": {
                    "hsv": [0.1, 1, 1],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            max_graph_size=10,
            category_taxonomy=taxonomy,
            category_bias_strength=0.5,
        )

        # Train both objects with identical geometry
        lm.mode = ExperimentMode.TRAIN
        for obs in self.fake_obs_learn:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "ball_graph"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        target = {"object": "fruit_graph", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        for obs in self.fake_obs_learn:
            lm.exploratory_step(self.ctx, [obs])
        lm.detected_object = "fruit_graph"
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

        return lm

    def test_multi_episode_hpc_accumulation(self):
        """HPC accumulates associations across episodes and biases LM.

        Episode 1-3: HPC observes ball_graph + bouncy co-occurring.
        Episode 4: LM evaluates on ambiguous object. HPC context from
        accumulated associations biases toward ball_graph.
        """
        hpc = HippocampalModule(context_dim=16, min_confidence_to_bind=0.5)
        ctx = _make_ctx()

        # Episodes 1-3: HPC learns ball_graph ↔ bouncy
        for ep in range(3):
            hpc.pre_episode()
            hpc.matching_step(ctx, [
                _make_hpc_state("ball_graph", confidence=0.9),
                _make_hpc_state("bouncy", confidence=0.9),
            ])
            hpc.post_episode()

        # Verify HPC has accumulated 3 episodes
        self.assertEqual(len(hpc.episodic_memory), 3)

        # Verify association was learned
        assoc = hpc.recall_associations("ball_graph")
        self.assertIn("bouncy", assoc)
        self.assertGreater(assoc["bouncy"], 0)

        # Episode 4: LM evaluates on ambiguous geometry
        lm = self._make_lm_with_identical_objects()
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(primary_target=self.placeholder_target)

        for obs in self.fake_obs_learn[:2]:
            lm.matching_step(self.ctx, [obs])

        # Baseline: equal evidence
        ball_ev_before = np.max(lm.evidence["ball_graph"])
        fruit_ev_before = np.max(lm.evidence["fruit_graph"])
        self.assertAlmostEqual(ball_ev_before, fruit_ev_before, places=3)

        # HPC sees ball_graph in this episode
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball_graph")])

        # HPC generates context signal from accumulated associations
        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)

        # Apply to LM
        lm.receive_context(**signal)

        # Ball should be boosted by accumulated associations
        ball_ev_after = np.max(lm.evidence["ball_graph"])
        fruit_ev_after = np.max(lm.evidence["fruit_graph"])

        self.assertGreater(
            ball_ev_after, fruit_ev_after,
            f"After multi-episode HPC learning, ball should exceed fruit. "
            f"Ball: {ball_ev_after:.4f}, Fruit: {fruit_ev_after:.4f}",
        )


class TestPreloadAssociations(unittest.TestCase):
    """Test preload_associations() for Phase 3 behavioral injection."""

    def test_preload_creates_associations(self):
        """preload_associations seeds co-occurrence counts."""
        hpc = HippocampalModule(context_dim=16)

        pairs = [
            ("ball", "bouncy"),
            ("fruit", "edible"),
        ]
        hpc.preload_associations(pairs, count=10)

        # Verify associations exist
        ball_assoc = hpc.recall_associations("ball")
        self.assertIn("bouncy", ball_assoc)
        self.assertGreater(ball_assoc["bouncy"], 0)

        fruit_assoc = hpc.recall_associations("fruit")
        self.assertIn("edible", fruit_assoc)
        self.assertGreater(fruit_assoc["edible"], 0)

    def test_preloaded_hpc_produces_useful_context(self):
        """Pre-loaded HPC generates context signals that bias downstream LMs."""
        hpc = HippocampalModule(context_dim=16)
        hpc.preload_associations([
            ("ball_graph", "bouncy"),
            ("ball_graph", "round"),
        ], count=5)

        ctx = _make_ctx()

        # HPC sees ball_graph
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball_graph")])

        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)

        # association_strengths should include bouncy and round
        strengths = signal["association_strengths"]
        self.assertIn("bouncy", strengths)
        self.assertGreater(strengths["bouncy"], 0)
        self.assertIn("round", strengths)
        self.assertGreater(strengths["round"], 0)

    def test_preloaded_associations_persist_across_episodes(self):
        """Pre-loaded associations are available in all subsequent episodes."""
        hpc = HippocampalModule(context_dim=16)
        hpc.preload_associations([("ball", "bouncy")], count=10)

        ctx = _make_ctx()

        # Episode 1
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball")])
        signal1 = hpc.get_context_signal()
        hpc.post_episode()

        # Episode 2
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state("ball")])
        signal2 = hpc.get_context_signal()

        # Both signals should have bouncy association
        self.assertIsNotNone(signal1)
        self.assertIsNotNone(signal2)
        self.assertIn("bouncy", signal1["association_strengths"])
        self.assertIn("bouncy", signal2["association_strengths"])


# ---------------------------------------------------------------------------
# Test 7: Training-mode get_output emits target identity for downstream LMs
# ---------------------------------------------------------------------------


class TestTrainingModeGetOutput(BaseGraphTest):
    """Verify that LMs emit target identity during supervised training.

    During supervised pretraining, the LM knows the object identity
    (via stepwise_target_object). get_output() should return a State
    with that identity so downstream modules (HPC) can learn associations.
    """

    def test_get_output_returns_target_during_training(self):
        """get_output() returns State with graph_id during training mode."""
        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=10,
        )
        lm.mode = ExperimentMode.TRAIN
        lm.learning_module_id = "lm_0"
        lm.stepwise_target_object = "055_baseball"

        output = lm.get_output()
        # EvidenceGraphLM overrides get_output, so training-mode
        # identity comes from the parent class method only when
        # MLH is not available.  To test the parent behavior, call
        # the parent directly.
        from tbp.monty.frameworks.models.graph_matching import GraphLM
        parent_output = GraphLM.get_output(lm)

        self.assertIsNotNone(parent_output)
        self.assertTrue(parent_output.use_state)
        self.assertEqual(parent_output.confidence, 1.0)
        self.assertEqual(
            parent_output.non_morphological_features["graph_id"],
            "055_baseball",
        )
        self.assertEqual(parent_output.sender_type, "LM")

    def test_get_output_returns_none_during_eval(self):
        """get_output() returns None during eval for base GraphMatchingLM."""
        from tbp.monty.frameworks.models.graph_matching import GraphLM

        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=10,
        )
        lm.mode = ExperimentMode.EVAL
        lm.learning_module_id = "lm_0"
        lm.stepwise_target_object = "055_baseball"

        parent_output = GraphLM.get_output(lm)
        self.assertIsNone(parent_output)

    def test_get_output_returns_none_for_no_label(self):
        """get_output() returns None when target is 'no_label'."""
        from tbp.monty.frameworks.models.graph_matching import GraphLM

        lm = EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={"patch": {"hsv": [0.1, 1, 1]}},
            feature_weights={"patch": {"hsv": np.array([1, 0, 0])}},
            max_graph_size=10,
        )
        lm.mode = ExperimentMode.TRAIN
        lm.learning_module_id = "lm_0"
        lm.stepwise_target_object = "no_label"

        parent_output = GraphLM.get_output(lm)
        self.assertIsNone(parent_output)


# ---------------------------------------------------------------------------
# Test 8: HPC learns from training-mode LM outputs
# ---------------------------------------------------------------------------


class TestHPCLearnsFromTraining(unittest.TestCase):
    """Verify HPC accumulates associations when receiving LM training outputs.

    Simulates the supervised training pipeline where the LM emits its
    target identity via get_output() and HPC receives those States.
    """

    def test_hpc_learns_object_sequence_during_training(self):
        """HPC learns co-occurrence from consecutive training episodes.

        Simulates: Episode 1: LM trains on baseball → HPC sees baseball.
        Episode 2: LM trains on banana → HPC sees banana.
        After both, HPC should have episodic memory of both objects.
        """
        hpc = HippocampalModule(context_dim=16, min_confidence_to_bind=0.5)
        ctx = _make_ctx()

        # Episode 1: LM training on baseball
        hpc.pre_episode()
        baseball_state = _make_hpc_state("055_baseball", confidence=1.0)
        baseball_state.non_morphological_features["graph_id"] = "055_baseball"
        hpc.exploratory_step(ctx, [baseball_state])
        hpc.post_episode()

        # Episode 2: LM training on banana
        hpc.pre_episode()
        banana_state = _make_hpc_state("011_banana", confidence=1.0)
        banana_state.non_morphological_features["graph_id"] = "011_banana"
        hpc.exploratory_step(ctx, [banana_state])
        hpc.post_episode()

        # HPC should have recorded both episodes
        self.assertEqual(len(hpc.episodic_memory), 2)

    def test_hpc_learns_temporal_transitions_during_training(self):
        """HPC learns A→B transition from training episode sequence.

        After several episodes, the HPC should learn the temporal transition
        pattern and be able to predict the next object.
        """
        hpc = HippocampalModule(
            context_dim=16,
            min_confidence_to_bind=0.5,
            temporal_dim=128,
            temporal_sparsity=0.04,
        )
        ctx = _make_ctx()

        # Simulate training sequence: ball → ball → banana → banana → ball...
        objects = ["055_baseball", "055_baseball",
                   "011_banana", "011_banana",
                   "055_baseball", "055_baseball"]
        for obj_name in objects:
            hpc.pre_episode()
            state = _make_hpc_state(obj_name, confidence=1.0)
            state.non_morphological_features["graph_id"] = obj_name
            hpc.exploratory_step(ctx, [state])
            hpc.post_episode()

        # After training, HPC should have recorded all episodes
        self.assertEqual(len(hpc.episodic_memory), 6)

        # Temporal predictions should be available
        preds = hpc.get_temporal_predictions("055_baseball")
        # After baseball, both baseball and banana have appeared
        self.assertIsNotNone(preds)

    def test_combine_inputs_handles_none_lm_output(self):
        """_combine_inputs should not crash when LM output is None.

        This happens during training when DisplacementGraphLM.get_output()
        returns None (it doesn't override get_output with training-mode
        identity emission).
        """
        from tbp.monty.frameworks.models.monty_base import MontyBase

        # Create a minimal MontyBase-like object to test _combine_inputs
        class _FakeMonty:
            _combine_inputs = MontyBase._combine_inputs

        monty = _FakeMonty()

        # HPC has no SM connections, receives None from upstream LM
        result = monty._combine_inputs([], [None])
        # Should return None (no useful inputs), not crash
        self.assertIsNone(result)

    def test_combine_inputs_passes_valid_lm_output(self):
        """_combine_inputs should pass through valid LM States."""
        from tbp.monty.frameworks.models.monty_base import MontyBase

        class _FakeMonty:
            _combine_inputs = MontyBase._combine_inputs

        monty = _FakeMonty()

        state = State(
            location=np.zeros(3),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
            },
            non_morphological_features={"graph_id": "ball"},
            confidence=1.0,
            use_state=True,
            sender_id="lm_0",
            sender_type="LM",
        )

        # HPC has no SM connections, receives valid State from upstream LM
        result = monty._combine_inputs([], [state])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].non_morphological_features["graph_id"], "ball"
        )


if __name__ == "__main__":
    unittest.main()
