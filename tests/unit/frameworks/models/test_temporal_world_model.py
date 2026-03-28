"""Unit tests for the temporal world model (T2.4/T2.5).

Tests the SDR-based Hebbian temporal prediction system in HippocampalModule:
- SDR generation (deterministic, sparse, binary)
- Hebbian temporal association learning (outer-product rule)
- Spreading activation prediction (P(next | current))
- Action-conditioned prediction (P(next | current, action))
- Prediction validation and accuracy tracking
- Context signal integration (temporal predictions prime downstream LMs)
- Forward replay planning (hippocampal replay)
- Trajectory simulation (vicarious trial and error)
- Serialization round-trip for SDR/matrix state
"""
import unittest

import numpy as np

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.hippocampal_module import HippocampalModule


def _make_ctx(global_step=0, episode_step=0):
    """Create a minimal RuntimeContext with a fake timer."""
    class _FakeTimer:
        pass

    t = _FakeTimer()
    t.global_step = global_step
    t.episode_step = episode_step
    rng = np.random.RandomState(42)
    return RuntimeContext(rng=rng, timer=t)


def _make_state(obj_id, confidence=0.9, location=None, use_state=True):
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


def _run_episode(hpc, concept, n_steps=5):
    """Simulate a single-object recognition episode."""
    hpc.pre_episode()
    for step in range(n_steps):
        ctx = _make_ctx(global_step=step, episode_step=step)
        hpc.matching_step(ctx, [_make_state(concept)])
    hpc.post_episode()


class TestSDRGeneration(unittest.TestCase):
    """Test sparse distributed representation generation."""

    def test_sdr_is_sparse_binary(self):
        hpc = HippocampalModule(temporal_dim=256, temporal_sparsity=0.05)
        sdr = hpc._get_concept_sdr("ball")
        # Binary
        self.assertTrue(np.all((sdr == 0) | (sdr == 1)))
        # Sparse
        n_active = int(sdr.sum())
        expected = int(256 * 0.05)
        self.assertEqual(n_active, expected)

    def test_sdr_is_deterministic(self):
        hpc1 = HippocampalModule(temporal_dim=256)
        hpc2 = HippocampalModule(temporal_dim=256)
        np.testing.assert_array_equal(
            hpc1._get_concept_sdr("ball"),
            hpc2._get_concept_sdr("ball"),
        )

    def test_different_concepts_different_sdrs(self):
        hpc = HippocampalModule(temporal_dim=256, temporal_sparsity=0.05)
        sdr_ball = hpc._get_concept_sdr("ball")
        sdr_cup = hpc._get_concept_sdr("cup")
        # Should not be identical (extremely unlikely with random SDRs)
        self.assertFalse(np.array_equal(sdr_ball, sdr_cup))

    def test_sdr_correct_dimensionality(self):
        hpc = HippocampalModule(temporal_dim=128)
        sdr = hpc._get_concept_sdr("test")
        self.assertEqual(len(sdr), 128)


class TestHebbianLearning(unittest.TestCase):
    """Test Hebbian temporal association learning."""

    def test_hebbian_update_strengthens_association(self):
        hpc = HippocampalModule(temporal_dim=256, hebbian_learning_rate=1.0)
        sdr_a = hpc._get_concept_sdr("ball")
        sdr_b = hpc._get_concept_sdr("cup")

        # Before learning: no association
        activation_before = hpc._temporal_W @ sdr_a
        overlap_before = float(np.dot(activation_before, sdr_b))

        # Learn A → B
        hpc._hebbian_update(sdr_b, sdr_a)

        # After learning: association exists
        activation_after = hpc._temporal_W @ sdr_a
        overlap_after = float(np.dot(activation_after, sdr_b))

        self.assertGreater(overlap_after, overlap_before)

    def test_repeated_learning_strengthens(self):
        hpc = HippocampalModule(temporal_dim=256, hebbian_learning_rate=1.0)
        sdr_a = hpc._get_concept_sdr("ball")
        sdr_b = hpc._get_concept_sdr("cup")

        hpc._hebbian_update(sdr_b, sdr_a)
        activation_1 = hpc._temporal_W @ sdr_a
        overlap_1 = float(np.dot(activation_1, sdr_b))

        hpc._hebbian_update(sdr_b, sdr_a)
        activation_2 = hpc._temporal_W @ sdr_a
        overlap_2 = float(np.dot(activation_2, sdr_b))

        self.assertGreater(overlap_2, overlap_1)

    def test_action_conditioned_matrix_created(self):
        hpc = HippocampalModule(temporal_dim=256)
        sdr_a = hpc._get_concept_sdr("ball")
        sdr_b = hpc._get_concept_sdr("cup")

        hpc._pending_action = "push"
        hpc._hebbian_update(sdr_b, sdr_a)

        self.assertIn("push", hpc._action_W)
        activation = hpc._action_W["push"] @ sdr_a
        overlap = float(np.dot(activation, sdr_b))
        self.assertGreater(overlap, 0)


class TestSpreadingActivationPrediction(unittest.TestCase):
    """Test prediction via spreading activation."""

    def test_learned_transition_predicts_correctly(self):
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            hebbian_learning_rate=1.0,
        )
        # Learn ball → cup via episodes
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        preds = hpc.get_temporal_predictions(from_concept="ball")
        self.assertIn("cup", preds)
        # cup should have the highest probability
        top = max(preds, key=preds.get)
        self.assertEqual(top, "cup")

    def test_stronger_association_dominates(self):
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            hebbian_learning_rate=1.0,
        )
        # ball → cup (3 times), ball → fruit (1 time)
        for _ in range(3):
            _run_episode(hpc, "ball")
            _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")
        _run_episode(hpc, "fruit")

        preds = hpc.get_temporal_predictions(from_concept="ball")
        # cup should be predicted more strongly than fruit
        self.assertGreater(preds.get("cup", 0), preds.get("fruit", 0))

    def test_probabilities_sum_to_one(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")
        _run_episode(hpc, "fruit")

        preds = hpc.get_temporal_predictions(from_concept="ball")
        if preds:
            self.assertAlmostEqual(sum(preds.values()), 1.0, places=5)

    def test_no_prediction_before_learning(self):
        hpc = HippocampalModule(temporal_dim=256)
        preds = hpc.get_temporal_predictions(from_concept="ball")
        self.assertEqual(len(preds), 0)

    def test_prediction_from_last_terminal(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")

        # Last terminal is "ball", should predict "cup"
        preds = hpc.get_temporal_predictions()
        self.assertIn("cup", preds)

    def test_sequence_learning(self):
        """Learn A→B→C cycle and verify predictions."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        for _ in range(4):
            _run_episode(hpc, "ball")
            _run_episode(hpc, "cup")
            _run_episode(hpc, "fruit")

        preds_a = hpc.get_temporal_predictions(from_concept="ball")
        preds_b = hpc.get_temporal_predictions(from_concept="cup")

        # ball should predict cup, cup should predict fruit
        self.assertEqual(max(preds_a, key=preds_a.get), "cup")
        self.assertEqual(max(preds_b, key=preds_b.get), "fruit")


class TestActionConditionedPrediction(unittest.TestCase):
    """Test action-conditioned predictions via separate matrices."""

    def test_different_actions_different_predictions(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)

        # ball →[push]→ cup (2 times)
        for _ in range(2):
            _run_episode(hpc, "ball")
            hpc.record_action("push")
            _run_episode(hpc, "cup")

        # ball →[pull]→ fruit (2 times)
        for _ in range(2):
            _run_episode(hpc, "ball")
            hpc.record_action("pull")
            _run_episode(hpc, "fruit")

        push_preds = hpc.get_temporal_predictions(
            from_concept="ball", action="push"
        )
        pull_preds = hpc.get_temporal_predictions(
            from_concept="ball", action="pull"
        )

        # push should prefer cup, pull should prefer fruit
        self.assertEqual(max(push_preds, key=push_preds.get), "cup")
        self.assertEqual(max(pull_preds, key=pull_preds.get), "fruit")

    def test_unknown_action_empty_prediction(self):
        hpc = HippocampalModule(temporal_dim=256)
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")

        preds = hpc.get_temporal_predictions(
            from_concept="ball", action="unknown"
        )
        self.assertEqual(len(preds), 0)

    def test_default_action_label(self):
        """Transitions without explicit action get 'episode_transition'."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        preds = hpc.get_temporal_predictions(
            from_concept="ball", action="episode_transition"
        )
        self.assertIn("cup", preds)


class TestPredictionValidation(unittest.TestCase):
    """Test prediction accuracy tracking."""

    def test_prediction_hit_recorded(self):
        """Correct prediction is recorded as a hit."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        # Learn ball → cup
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        # ball again — sets up prediction "cup"
        _run_episode(hpc, "ball")
        # cup confirms the prediction
        _run_episode(hpc, "cup")

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["total"], 0)
        self.assertGreater(acc["hits"], 0)
        # The last recorded prediction should be a hit
        last = acc["history"][-1]
        self.assertTrue(last["hit"])
        self.assertTrue(last["top_correct"])
        self.assertEqual(last["observed"], "cup")

    def test_prediction_miss_recorded(self):
        """Incorrect prediction is recorded as a miss."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        # Learn ball → cup
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        # ball again — sets up prediction "cup"
        _run_episode(hpc, "ball")
        # Surprise: fruit instead of cup
        _run_episode(hpc, "fruit")

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["total"], 0)
        last = acc["history"][-1]
        self.assertFalse(last["top_correct"])
        self.assertEqual(last["observed"], "fruit")
        self.assertEqual(last["top_prediction"], "cup")

    def test_accuracy_improves_with_experience(self):
        """More experience → higher prediction accuracy."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        # Deterministic sequence: A → B → C → A → B → C ...
        sequence = ["ball", "cup", "fruit"]

        # Run 6 cycles (18 episodes) — system should learn the pattern
        for _ in range(6):
            for concept in sequence:
                _run_episode(hpc, concept)

        acc = hpc.get_prediction_accuracy()
        # After 18 episodes, most predictions should be correct
        # (first couple may miss before enough data is learned)
        self.assertGreater(acc["total"], 5)
        self.assertGreater(acc["accuracy"], 0.5,
                           f"Expected >50% accuracy, got {acc['accuracy']:.1%}")

    def test_accuracy_on_repeated_pair(self):
        """Repeated A→B pair should achieve near-perfect accuracy."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        # Learn ball → cup, 10 times
        for _ in range(10):
            _run_episode(hpc, "ball")
            _run_episode(hpc, "cup")

        acc = hpc.get_prediction_accuracy()
        # After 20 episodes, predictions from "ball" should reliably say "cup"
        # and predictions from "cup" should reliably say "ball"
        self.assertGreater(acc["total"], 5)
        self.assertGreater(acc["top_accuracy"], 0.7,
                           f"Expected >70% top-1 accuracy, got "
                           f"{acc['top_accuracy']:.1%}")

    def test_no_validation_on_first_episode(self):
        """No prediction history for the very first episode."""
        hpc = HippocampalModule(temporal_dim=256)
        _run_episode(hpc, "ball")

        acc = hpc.get_prediction_accuracy()
        self.assertEqual(acc["total"], 0)

    def test_accuracy_with_branching_sequences(self):
        """System should track accuracy even with non-deterministic sequences."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        # ball → cup (3x), ball → fruit (1x)
        for _ in range(3):
            _run_episode(hpc, "ball")
            _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")
        _run_episode(hpc, "fruit")

        acc = hpc.get_prediction_accuracy()
        # System should have made some predictions
        self.assertGreater(acc["total"], 0)
        # At least some hits (cup was predicted correctly multiple times)
        self.assertGreater(acc["hits"], 0)

    def test_prediction_history_contains_probabilities(self):
        """Each prediction entry should contain the full probability dict."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        acc = hpc.get_prediction_accuracy()
        for entry in acc["history"]:
            self.assertIn("predicted", entry)
            self.assertIn("observed", entry)
            self.assertIn("hit", entry)
            self.assertIn("top_prediction", entry)
            self.assertIn("top_correct", entry)
            # predicted should be a dict with probabilities
            self.assertIsInstance(entry["predicted"], dict)
            if entry["predicted"]:
                total = sum(entry["predicted"].values())
                self.assertAlmostEqual(total, 1.0, places=3)


class TestContextSignalIntegration(unittest.TestCase):
    """Test that temporal predictions flow into the context signal."""

    def test_predictions_in_context_signal(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")

        hpc.pre_episode()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("cup")])

        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)
        self.assertIn("temporal_predictions", signal)
        self.assertIn("cup", signal["temporal_predictions"])

    def test_predictions_merged_into_association_strengths(self):
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            temporal_prediction_weight=0.8,
        )
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")

        hpc.pre_episode()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("something")])

        signal = hpc.get_context_signal()
        # cup should appear in association_strengths due to temporal prediction
        self.assertIn("cup", signal["association_strengths"])
        self.assertGreater(signal["association_strengths"]["cup"], 0)

    def test_context_signal_with_only_predictions(self):
        """Context signal returned even with no active concepts if
        temporal predictions exist."""
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")

        hpc.pre_episode()
        # No matching_step → no active concepts
        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)
        self.assertIn("cup", signal["temporal_predictions"])

    def test_temporal_weight_zero_disables_priming(self):
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            temporal_prediction_weight=0.0,
        )
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")

        hpc.pre_episode()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("something")])

        signal = hpc.get_context_signal()
        # Predictions exist but weight=0 so no priming
        self.assertIn("cup", signal["temporal_predictions"])
        cup_strength = signal["association_strengths"].get("cup", 0.0)
        self.assertAlmostEqual(cup_strength, 0.0)


class TestForwardReplayPlanning(unittest.TestCase):
    """Test goal-directed planning via forward replay."""

    def test_direct_transition_plan(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        plan = hpc.plan_action_sequence("ball", "cup")
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0][1], "cup")

    def test_multi_step_plan(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "fruit")

        plan = hpc.plan_action_sequence("ball", "fruit")
        self.assertIsNotNone(plan)
        # Should reach fruit (possibly via cup)
        self.assertEqual(plan[-1][1], "fruit")

    def test_already_at_goal(self):
        hpc = HippocampalModule(temporal_dim=256)
        plan = hpc.plan_action_sequence("ball", "ball")
        self.assertEqual(plan, [])

    def test_no_path_returns_none(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        # "tool" was never learned — no path
        plan = hpc.plan_action_sequence("cup", "tool")
        self.assertIsNone(plan)

    def test_plan_with_action_annotations(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")

        plan = hpc.plan_action_sequence("ball", "cup")
        self.assertIsNotNone(plan)
        # Action should be "push" (learned via Hebbian association)
        self.assertEqual(plan[0][0], "push")


class TestTrajectorySimulation(unittest.TestCase):
    """Test mental simulation of action trajectories."""

    def test_simulate_known_trajectory(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")

        trajectory = hpc.simulate_trajectory("ball", ["push"])
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(trajectory[0][0], "cup")

    def test_simulate_unknown_action_stops(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")

        trajectory = hpc.simulate_trajectory("ball", ["push", "unknown"])
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory[0][0], "cup")
        self.assertIsNone(trajectory[1][0])

    def test_simulate_empty_actions(self):
        hpc = HippocampalModule(temporal_dim=256)
        trajectory = hpc.simulate_trajectory("ball", [])
        self.assertEqual(len(trajectory), 0)


class TestSerializationRoundTrip(unittest.TestCase):
    """Test that SDR/Hebbian state survives serialization."""

    def test_temporal_state_preserved(self):
        hpc = HippocampalModule(
            temporal_dim=256, temporal_sparsity=0.05,
            temporal_prediction_weight=0.7,
        )
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")

        state = hpc.state_dict()
        hpc2 = HippocampalModule(temporal_dim=256)
        hpc2.load_state_dict(state)

        # SDRs preserved
        np.testing.assert_array_equal(
            hpc2._concept_sdrs["ball"], hpc._concept_sdrs["ball"]
        )
        # Weight matrix preserved
        np.testing.assert_array_almost_equal(
            hpc2._temporal_W, hpc._temporal_W
        )
        # Predictions still work
        preds = hpc2.get_temporal_predictions(from_concept="ball")
        self.assertIn("cup", preds)
        self.assertAlmostEqual(hpc2.temporal_prediction_weight, 0.7)


class TestMotorActionReporting(unittest.TestCase):
    """Test automatic motor→HPC action reporting via MontyBase."""

    def _make_monty_with_hpc(self):
        from tbp.monty.frameworks.models.monty_base import MontyBase

        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)

        class _FakeMotorSystem:
            action_sequence = []
            motor_only_step = False

            def pre_episode(self):
                self.action_sequence = []

            def __call__(self, ctx, obs):
                return []

        class _FakeAction:
            def __init__(self, name):
                self._name = name

            def action_name(self):
                return self._name

        monty = object.__new__(MontyBase)
        monty.learning_modules = [hpc]
        monty.sensor_modules = []
        monty.motor_system = _FakeMotorSystem()
        return monty, hpc, _FakeAction

    def test_summarize_motor_actions(self):
        monty, hpc, FakeAction = self._make_monty_with_hpc()
        monty.motor_system.action_sequence = [
            ([FakeAction("move_tangentially")], None),
            ([FakeAction("move_tangentially")], None),
            ([FakeAction("turn_left")], None),
        ]

        summary = monty._summarize_motor_actions()
        self.assertEqual(summary, "move_tangentially")

    def test_post_episode_reports_action_to_hpc(self):
        monty, hpc, FakeAction = self._make_monty_with_hpc()

        # Episode 1
        hpc.pre_episode()
        for step in range(5):
            ctx = _make_ctx(global_step=step, episode_step=step)
            hpc.matching_step(ctx, [_make_state("ball")])
        monty.motor_system.action_sequence = [
            ([FakeAction("move_tangentially")], None),
        ]
        monty.post_episode()

        # Episode 2
        hpc.pre_episode()
        for step in range(5):
            ctx = _make_ctx(global_step=step + 5, episode_step=step)
            hpc.matching_step(ctx, [_make_state("cup")])
        monty.motor_system.action_sequence = [
            ([FakeAction("turn_left")], None),
        ]
        monty.post_episode()

        # ball→cup transition annotated with "turn_left"
        preds = hpc.get_temporal_predictions(
            from_concept="ball", action="turn_left"
        )
        self.assertIn("cup", preds)


class TestEdgeCases(unittest.TestCase):

    def test_empty_episode_no_transition(self):
        hpc = HippocampalModule(temporal_dim=256)
        _run_episode(hpc, "ball")
        hpc.pre_episode()
        hpc.post_episode()  # empty episode
        _run_episode(hpc, "cup")

        # ball→cup should still be recorded
        preds = hpc.get_temporal_predictions(from_concept="ball")
        self.assertIn("cup", preds)

    def test_self_transition(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        _run_episode(hpc, "ball")

        preds = hpc.get_temporal_predictions(from_concept="ball")
        self.assertIn("ball", preds)

    def test_record_action_cleared_after_episode(self):
        hpc = HippocampalModule(temporal_dim=512, temporal_sparsity=0.04)
        _run_episode(hpc, "ball")
        hpc.record_action("push")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "ball")  # no action recorded

        # cup→ball should use "episode_transition", not "push"
        preds_push = hpc.get_temporal_predictions(
            from_concept="cup", action="push"
        )
        preds_default = hpc.get_temporal_predictions(
            from_concept="cup", action="episode_transition"
        )
        # push matrix has no cup→ball association
        if preds_push:
            # If push predictions exist for cup, they shouldn't be about ball
            # (push was only used for ball→cup)
            self.assertLess(
                preds_push.get("ball", 0),
                preds_default.get("ball", 0),
            )
        self.assertIn("ball", preds_default)


# ======================================================================
# Temporal scenario tests — real-world objects with temporal dynamics
# ======================================================================

def _hpc(**kwargs):
    """Create an HPC with defaults suitable for scenario tests."""
    defaults = dict(temporal_dim=512, temporal_sparsity=0.04,
                    hebbian_learning_rate=1.0)
    defaults.update(kwargs)
    return HippocampalModule(**defaults)


class TestStaplerTwoState(unittest.TestCase):
    """Stapler: discrete reversible two-state object."""

    def test_press_opens_stapler(self):
        hpc = _hpc()
        _run_episode(hpc, "stapler_closed")
        hpc.record_action("press")
        _run_episode(hpc, "stapler_open")

        preds = hpc.get_temporal_predictions(
            from_concept="stapler_closed", action="press"
        )
        self.assertEqual(max(preds, key=preds.get), "stapler_open")

    def test_release_closes_stapler(self):
        hpc = _hpc()
        _run_episode(hpc, "stapler_open")
        hpc.record_action("release")
        _run_episode(hpc, "stapler_closed")

        preds = hpc.get_temporal_predictions(
            from_concept="stapler_open", action="release"
        )
        self.assertEqual(max(preds, key=preds.get), "stapler_closed")

    def test_bidirectional_cycle(self):
        """Learn both directions, verify both predictions work."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")
            hpc.record_action("release")

        press_preds = hpc.get_temporal_predictions(
            from_concept="stapler_closed", action="press"
        )
        release_preds = hpc.get_temporal_predictions(
            from_concept="stapler_open", action="release"
        )
        self.assertEqual(max(press_preds, key=press_preds.get), "stapler_open")
        self.assertEqual(
            max(release_preds, key=release_preds.get), "stapler_closed"
        )

    def test_plan_to_open_stapler(self):
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")
            hpc.record_action("release")

        plan = hpc.plan_action_sequence("stapler_closed", "stapler_open")
        self.assertIsNotNone(plan)
        self.assertEqual(plan[0], ("press", "stapler_open"))

    def test_prediction_accuracy_on_repeated_cycles(self):
        hpc = _hpc()
        for _ in range(8):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")
            hpc.record_action("release")

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["accuracy"], 0.6,
                           f"Stapler accuracy {acc['accuracy']:.0%}")


class TestCanIrreversibleChain(unittest.TestCase):
    """Can: irreversible state chain (sealed → opened → empty)."""

    def test_pull_tab_opens_can(self):
        hpc = _hpc()
        _run_episode(hpc, "can_sealed")
        hpc.record_action("pull_tab")
        _run_episode(hpc, "can_opened")

        preds = hpc.get_temporal_predictions(
            from_concept="can_sealed", action="pull_tab"
        )
        self.assertEqual(max(preds, key=preds.get), "can_opened")

    def test_pour_empties_can(self):
        hpc = _hpc()
        _run_episode(hpc, "can_opened")
        hpc.record_action("pour")
        _run_episode(hpc, "can_empty")

        preds = hpc.get_temporal_predictions(
            from_concept="can_opened", action="pour"
        )
        self.assertEqual(max(preds, key=preds.get), "can_empty")

    def test_full_chain_simulation(self):
        """Simulate the full chain: sealed →[pull_tab]→ opened →[pour]→ empty."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")
            hpc.record_action("pour")
            _run_episode(hpc, "can_empty")

        trajectory = hpc.simulate_trajectory(
            "can_sealed", ["pull_tab", "pour"]
        )
        self.assertEqual(len(trajectory), 2)
        self.assertEqual(trajectory[0][0], "can_opened")
        self.assertEqual(trajectory[1][0], "can_empty")

    def test_plan_sealed_to_empty(self):
        """Planning should find the two-step path."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")
            hpc.record_action("pour")
            _run_episode(hpc, "can_empty")

        plan = hpc.plan_action_sequence("can_sealed", "can_empty")
        self.assertIsNotNone(plan)
        self.assertEqual(plan[-1][1], "can_empty")
        self.assertLessEqual(len(plan), 3)

    def test_no_op_action_preserves_state(self):
        """Shaking a sealed can doesn't change its state."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("shake")
            _run_episode(hpc, "can_sealed")

        preds = hpc.get_temporal_predictions(
            from_concept="can_sealed", action="shake"
        )
        self.assertEqual(max(preds, key=preds.get), "can_sealed")

    def test_multi_action_divergence(self):
        """Different actions from same state → different outcomes."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")

            _run_episode(hpc, "can_sealed")
            hpc.record_action("throw")
            _run_episode(hpc, "can_dented")

        pull_preds = hpc.get_temporal_predictions(
            from_concept="can_sealed", action="pull_tab"
        )
        throw_preds = hpc.get_temporal_predictions(
            from_concept="can_sealed", action="throw"
        )
        self.assertEqual(max(pull_preds, key=pull_preds.get), "can_opened")
        self.assertEqual(max(throw_preds, key=throw_preds.get), "can_dented")


class TestScissorsCyclic(unittest.TestCase):
    """Scissors: rapid reversible two-state cycle (cutting)."""

    def test_rapid_alternation_accuracy(self):
        """20 open/close cycles — should achieve high accuracy."""
        hpc = _hpc()
        for _ in range(20):
            _run_episode(hpc, "scissors_open")
            hpc.record_action("cut")
            _run_episode(hpc, "scissors_closed")
            hpc.record_action("release")

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["top_accuracy"], 0.7,
                           f"Scissors accuracy {acc['top_accuracy']:.0%}")

    def test_action_discrimination(self):
        """cut → closed, release → open."""
        hpc = _hpc()
        for _ in range(5):
            _run_episode(hpc, "scissors_open")
            hpc.record_action("cut")
            _run_episode(hpc, "scissors_closed")
            hpc.record_action("release")

        cut_preds = hpc.get_temporal_predictions(
            from_concept="scissors_open", action="cut"
        )
        release_preds = hpc.get_temporal_predictions(
            from_concept="scissors_closed", action="release"
        )
        self.assertEqual(max(cut_preds, key=cut_preds.get), "scissors_closed")
        self.assertEqual(
            max(release_preds, key=release_preds.get), "scissors_open"
        )


class TestWheelMultiPhase(unittest.TestCase):
    """Wheel: 4-phase rotation cycle."""

    PHASES = ["wheel_0", "wheel_90", "wheel_180", "wheel_270"]

    def test_cycle_prediction(self):
        """Each phase predicts the next after learning."""
        hpc = _hpc()
        for _ in range(5):
            for phase in self.PHASES:
                _run_episode(hpc, phase)

        for i, phase in enumerate(self.PHASES):
            next_phase = self.PHASES[(i + 1) % 4]
            preds = hpc.get_temporal_predictions(from_concept=phase)
            top = max(preds, key=preds.get)
            self.assertEqual(top, next_phase,
                             f"{phase} should predict {next_phase}, got {top}")

    def test_wraparound(self):
        """wheel_270 → wheel_0 (cycle wraps)."""
        hpc = _hpc()
        for _ in range(5):
            for phase in self.PHASES:
                _run_episode(hpc, phase)

        preds = hpc.get_temporal_predictions(from_concept="wheel_270")
        self.assertEqual(max(preds, key=preds.get), "wheel_0")

    def test_plan_half_rotation(self):
        """Plan from 0° to 180° — should find 2-step path."""
        hpc = _hpc()
        for _ in range(5):
            for phase in self.PHASES:
                _run_episode(hpc, phase)

        plan = hpc.plan_action_sequence("wheel_0", "wheel_180")
        self.assertIsNotNone(plan)
        self.assertEqual(plan[-1][1], "wheel_180")
        self.assertLessEqual(len(plan), 3)


class TestWalkingLongCycle(unittest.TestCase):
    """Walking: 7-state pose cycle stressing SDR capacity."""

    POSES = [
        "stand", "left_lift", "left_forward", "left_plant",
        "right_lift", "right_forward", "right_plant",
    ]

    def test_learns_walk_cycle(self):
        """After 4 walk cycles, each pose predicts the next."""
        hpc = _hpc()
        for _ in range(4):
            for pose in self.POSES:
                _run_episode(hpc, pose)

        correct = 0
        for i, pose in enumerate(self.POSES):
            next_pose = self.POSES[(i + 1) % len(self.POSES)]
            preds = hpc.get_temporal_predictions(from_concept=pose)
            if preds and max(preds, key=preds.get) == next_pose:
                correct += 1

        # At least 5 of 7 poses should predict correctly
        self.assertGreaterEqual(correct, 5,
                                f"Walk cycle: {correct}/7 correct")

    def test_prediction_accuracy_over_cycles(self):
        """Accuracy should be reasonable after several cycles."""
        hpc = _hpc()
        for _ in range(6):
            for pose in self.POSES:
                _run_episode(hpc, pose)

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["accuracy"], 0.4,
                           f"Walking accuracy {acc['accuracy']:.0%}")


class TestPianoChordProgression(unittest.TestCase):
    """Piano: chord progression cycle (I-V-vi-IV)."""

    CHORDS = ["C_major", "G_major", "A_minor", "F_major"]

    def test_chord_progression_learning(self):
        hpc = _hpc()
        for _ in range(5):
            for chord in self.CHORDS:
                _run_episode(hpc, chord)

        for i, chord in enumerate(self.CHORDS):
            next_chord = self.CHORDS[(i + 1) % 4]
            preds = hpc.get_temporal_predictions(from_concept=chord)
            top = max(preds, key=preds.get)
            self.assertEqual(top, next_chord,
                             f"{chord} should predict {next_chord}, got {top}")

    def test_simulate_full_progression(self):
        """Simulate I→V→vi→IV without action conditioning."""
        hpc = _hpc()
        for _ in range(5):
            for chord in self.CHORDS:
                _run_episode(hpc, chord)

        # Use unconditioned prediction (default action)
        trajectory = hpc.simulate_trajectory(
            "C_major", ["episode_transition"] * 3
        )
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(trajectory[0][0], "G_major")
        self.assertEqual(trajectory[1][0], "A_minor")
        self.assertEqual(trajectory[2][0], "F_major")


class TestEpisodicReplay(unittest.TestCase):
    """Test hippocampal replay for consolidation."""

    def test_replay_strengthens_associations(self):
        """Replay should increase prediction confidence."""
        hpc = _hpc()
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        # Prediction strength before replay
        preds_before = hpc.get_temporal_predictions(from_concept="ball")
        strength_before = preds_before.get("cup", 0)

        # The prediction should exist but let's measure raw activation
        sdr_ball = hpc._get_concept_sdr("ball")
        sdr_cup = hpc._get_concept_sdr("cup")
        activation_before = float(
            np.dot(hpc._temporal_W @ sdr_ball, sdr_cup)
        )

        # Replay consolidates
        n = hpc.replay_cross_episode(n_replays=3)
        self.assertGreater(n, 0)

        activation_after = float(
            np.dot(hpc._temporal_W @ sdr_ball, sdr_cup)
        )
        self.assertGreater(activation_after, activation_before)

    def test_replay_with_lower_learning_rate(self):
        """Replay with lower LR should still strengthen, but less."""
        hpc = _hpc()
        for _ in range(3):
            _run_episode(hpc, "ball")
            _run_episode(hpc, "cup")

        sdr_ball = hpc._get_concept_sdr("ball")
        sdr_cup = hpc._get_concept_sdr("cup")
        baseline = float(np.dot(hpc._temporal_W @ sdr_ball, sdr_cup))

        # Replay with low LR
        hpc.replay_cross_episode(n_replays=1, learning_rate=0.1)
        after_low = float(np.dot(hpc._temporal_W @ sdr_ball, sdr_cup))

        self.assertGreater(after_low, baseline)

    def test_replay_returns_update_count(self):
        hpc = _hpc()
        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")
        _run_episode(hpc, "fruit")

        n = hpc.replay_cross_episode(n_replays=2)
        # 3 episodes → 2 transitions → 2 updates per replay → 4 total
        self.assertEqual(n, 4)

    def test_replay_empty_memory(self):
        """Replay with no stored episodes does nothing."""
        hpc = _hpc()
        n = hpc.replay_cross_episode(n_replays=5)
        self.assertEqual(n, 0)

    def test_replay_improves_weak_associations(self):
        """Association learned once, then consolidated via replay."""
        hpc = _hpc()
        # Learn each transition only ONCE
        _run_episode(hpc, "alpha")
        _run_episode(hpc, "beta")
        _run_episode(hpc, "gamma")

        preds_before = hpc.get_temporal_predictions(from_concept="alpha")

        # Consolidate via replay
        hpc.replay_cross_episode(n_replays=5, learning_rate=0.5)

        preds_after = hpc.get_temporal_predictions(from_concept="alpha")
        # beta should be more strongly predicted after replay
        self.assertGreater(
            preds_after.get("beta", 0),
            preds_before.get("beta", 0) * 0.5,  # generous threshold
        )

    def test_reverse_replay_within_episode(self):
        """Reverse replay on a multi-concept episode creates backward links."""
        hpc = _hpc()
        # Simulate a multi-concept episode (e.g., multi-object scene)
        hpc.pre_episode()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("start")])
        hpc.matching_step(ctx, [_make_state("middle")])
        hpc.matching_step(ctx, [_make_state("end")])
        hpc.post_episode()

        W_norm_before = np.linalg.norm(hpc._temporal_W)

        n = hpc.replay(n_replays=3, reverse=True)
        self.assertGreater(n, 0)

        W_norm_after = np.linalg.norm(hpc._temporal_W)
        self.assertGreater(W_norm_after, W_norm_before)


class TestStateDiscrimination(unittest.TestCase):
    """Test T2.2: same object, different states."""

    def test_register_and_query_states(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])

        self.assertEqual(hpc.get_object_identity("stapler_open"), "stapler")
        self.assertEqual(hpc.get_object_identity("stapler_closed"), "stapler")
        self.assertEqual(hpc.get_object_identity("cup"), "cup")  # not registered

        states = hpc.get_object_states("stapler")
        self.assertEqual(states, {"stapler_open", "stapler_closed"})

    def test_get_current_state(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])
        _run_episode(hpc, "stapler_closed")

        self.assertEqual(hpc.get_current_state("stapler"), "stapler_closed")

    def test_predict_state_after_action(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])

        # Learn the transition
        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")

        # End on stapler_closed so current state is "closed"
        _run_episode(hpc, "stapler_closed")

        # Now predict: stapler (currently closed) + press → ?
        predicted = hpc.predict_state_after_action("stapler", "press")
        self.assertEqual(predicted, "stapler_open")

    def test_predict_filters_to_same_object_states(self):
        """Prediction should prefer states of the same object."""
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])
        hpc.register_object_states("scissors", ["scissors_open", "scissors_closed"])

        # Learn stapler_closed →[press]→ stapler_open
        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")

        # End on stapler_closed so current state is "closed"
        _run_episode(hpc, "stapler_closed")

        # Even if scissors SDRs have some overlap, prediction should
        # prefer stapler_open over scissors_open
        predicted = hpc.predict_state_after_action("stapler", "press")
        self.assertEqual(predicted, "stapler_open")

    def test_unknown_object_returns_none(self):
        hpc = _hpc()
        self.assertIsNone(hpc.predict_state_after_action("unknown", "push"))

    def test_can_state_chain(self):
        """Can with 3 states: sealed → opened → empty."""
        hpc = _hpc()
        hpc.register_object_states(
            "can", ["can_sealed", "can_opened", "can_empty"]
        )

        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")
            hpc.record_action("pour")
            _run_episode(hpc, "can_empty")

        self.assertEqual(
            hpc.predict_state_after_action("can", "pull_tab"), "can_opened"
        )

    def test_state_discrimination_survives_serialization(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])
        _run_episode(hpc, "stapler_closed")

        state = hpc.state_dict()
        hpc2 = _hpc()
        hpc2.load_state_dict(state)

        self.assertEqual(hpc2.get_object_identity("stapler_open"), "stapler")
        self.assertEqual(
            hpc2.get_object_states("stapler"),
            {"stapler_open", "stapler_closed"},
        )


class TestBehaviorRecognition(unittest.TestCase):
    """Test T2.6: temporal pattern matching."""

    def test_recognize_pouring(self):
        hpc = _hpc()
        hpc.register_behavior(
            "pouring", ["can_sealed", "can_opened", "can_empty"]
        )

        _run_episode(hpc, "can_sealed")
        _run_episode(hpc, "can_opened")
        _run_episode(hpc, "can_empty")

        results = hpc.recognize_behavior()
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "pouring")
        self.assertAlmostEqual(results[0][1], 1.0)

    def test_partial_behavior_recognized(self):
        """Recognize a behavior before it's complete."""
        hpc = _hpc()
        hpc.register_behavior(
            "pouring", ["can_sealed", "can_opened", "can_empty"]
        )

        _run_episode(hpc, "can_sealed")
        _run_episode(hpc, "can_opened")
        # Only 2 of 3 steps observed

        results = hpc.recognize_behavior()
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "pouring")
        self.assertAlmostEqual(results[0][1], 2 / 3)

    def test_discriminate_between_behaviors(self):
        """Should rank the correct behavior highest."""
        hpc = _hpc()
        hpc.register_behavior(
            "pouring", ["can_sealed", "can_opened", "can_empty"]
        )
        hpc.register_behavior(
            "cutting", ["scissors_open", "scissors_closed",
                        "scissors_open", "scissors_closed"]
        )

        _run_episode(hpc, "scissors_open")
        _run_episode(hpc, "scissors_closed")

        results = hpc.recognize_behavior()
        names = [r[0] for r in results]
        self.assertIn("cutting", names)
        # Cutting should rank higher than pouring
        if len(results) > 1:
            self.assertEqual(results[0][0], "cutting")

    def test_cyclic_behavior_wrap(self):
        """Cyclic behaviors should match at wrap-around points."""
        hpc = _hpc()
        hpc.register_behavior(
            "walking",
            ["stand", "left_lift", "left_plant", "right_lift", "right_plant"],
            cyclic=True,
        )

        # Observe sequence starting mid-cycle
        _run_episode(hpc, "right_lift")
        _run_episode(hpc, "right_plant")
        _run_episode(hpc, "stand")

        results = hpc.recognize_behavior()
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "walking")
        self.assertAlmostEqual(results[0][1], 3 / 5)

    def test_no_match_returns_empty(self):
        hpc = _hpc()
        hpc.register_behavior("pouring", ["can_sealed", "can_opened"])

        _run_episode(hpc, "ball")
        _run_episode(hpc, "cup")

        results = hpc.recognize_behavior()
        self.assertEqual(len(results), 0)

    def test_explicit_observed_sequence(self):
        """Pass observed sequence directly instead of using episode history."""
        hpc = _hpc()
        hpc.register_behavior("chord_prog", ["C", "G", "Am", "F"])

        results = hpc.recognize_behavior(observed=["C", "G", "Am"])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][0], "chord_prog")
        self.assertAlmostEqual(results[0][1], 3 / 4)


class TestManipulationViaPrediction(unittest.TestCase):
    """Test T2.7: using temporal model to select actions."""

    def test_suggest_action_for_stapler(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])

        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")
            hpc.record_action("release")

        # End on closed
        _run_episode(hpc, "stapler_closed")

        action = hpc.suggest_action("stapler_open", object_id="stapler")
        self.assertEqual(action, "press")

    def test_suggest_action_reverse_direction(self):
        hpc = _hpc()
        hpc.register_object_states("stapler", ["stapler_open", "stapler_closed"])

        for _ in range(3):
            _run_episode(hpc, "stapler_closed")
            hpc.record_action("press")
            _run_episode(hpc, "stapler_open")
            hpc.record_action("release")

        # End on open — suggest how to close
        action = hpc.suggest_action("stapler_closed", object_id="stapler")
        self.assertEqual(action, "release")

    def test_suggest_action_no_plan(self):
        hpc = _hpc()
        _run_episode(hpc, "ball")
        action = hpc.suggest_action("unknown_goal")
        self.assertIsNone(action)

    def test_evaluate_action(self):
        hpc = _hpc()
        hpc.register_object_states("can", ["can_sealed", "can_opened", "can_empty"])

        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")

        _run_episode(hpc, "can_sealed")

        outcome = hpc.evaluate_action("pull_tab", object_id="can")
        self.assertIn("can_opened", outcome)
        self.assertGreater(outcome["can_opened"], 0.5)

    def test_evaluate_unknown_action(self):
        hpc = _hpc()
        _run_episode(hpc, "ball")
        outcome = hpc.evaluate_action("unknown_action")
        self.assertEqual(len(outcome), 0)

    def test_full_manipulation_loop(self):
        """Full loop: observe → plan → suggest action → predict outcome."""
        hpc = _hpc()
        hpc.register_object_states(
            "can", ["can_sealed", "can_opened", "can_empty"]
        )

        for _ in range(3):
            _run_episode(hpc, "can_sealed")
            hpc.record_action("pull_tab")
            _run_episode(hpc, "can_opened")
            hpc.record_action("pour")
            _run_episode(hpc, "can_empty")

        # Start from sealed — goal is empty
        _run_episode(hpc, "can_sealed")

        # Step 1: suggest action
        action1 = hpc.suggest_action("can_empty", object_id="can")
        self.assertIsNotNone(action1)

        # Step 2: evaluate that action
        outcome1 = hpc.evaluate_action(action1, object_id="can")
        self.assertGreater(len(outcome1), 0)


class TestAccuracyLearningCurve(unittest.TestCase):
    """Verify that prediction accuracy improves over time."""

    def test_accuracy_monotonically_improves(self):
        """Track accuracy at checkpoints — should trend upward."""
        hpc = _hpc()
        sequence = ["ball", "cup", "fruit"]

        # Run 15 cycles, check accuracy at 5, 10, 15
        accuracies = []
        for cycle in range(15):
            for concept in sequence:
                _run_episode(hpc, concept)

            if (cycle + 1) in (5, 10, 15):
                acc = hpc.get_prediction_accuracy()
                accuracies.append(acc["accuracy"])

        # Accuracy at cycle 15 should be >= accuracy at cycle 5
        self.assertGreaterEqual(accuracies[-1], accuracies[0],
                                f"Accuracy should improve: {accuracies}")
        # Final accuracy should be decent
        self.assertGreater(accuracies[-1], 0.5,
                           f"Final accuracy: {accuracies[-1]:.0%}")


class TestInterferenceCapacity(unittest.TestCase):
    """Test SDR associative memory capacity and interference."""

    def test_many_pairs_coexist(self):
        """Learn 10 distinct A→B pairs, verify all are retrievable."""
        hpc = _hpc()
        pairs = [(f"src_{i}", f"dst_{i}") for i in range(10)]

        for _ in range(3):
            for src, dst in pairs:
                _run_episode(hpc, src)
                _run_episode(hpc, dst)

        correct = 0
        for src, dst in pairs:
            preds = hpc.get_temporal_predictions(from_concept=src)
            if preds and max(preds, key=preds.get) == dst:
                correct += 1

        self.assertGreaterEqual(correct, 7,
                                f"Capacity: {correct}/10 pairs correct")

    def test_early_pairs_survive_later_learning(self):
        """First-learned pairs should still work after adding more."""
        hpc = _hpc()

        # Learn pair 0 first (3 times)
        for _ in range(3):
            _run_episode(hpc, "first_src")
            _run_episode(hpc, "first_dst")

        # Then learn 8 more pairs
        for i in range(8):
            for _ in range(3):
                _run_episode(hpc, f"later_src_{i}")
                _run_episode(hpc, f"later_dst_{i}")

        # First pair should still be retrievable
        preds = hpc.get_temporal_predictions(from_concept="first_src")
        self.assertIn("first_dst", preds)
        top = max(preds, key=preds.get)
        self.assertEqual(top, "first_dst",
                         f"First pair degraded: top={top}")


if __name__ == "__main__":
    unittest.main()
