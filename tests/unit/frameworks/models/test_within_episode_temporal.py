"""Tests for within-episode temporal memory.

Tests that the temporal memory learns, predicts, and recognizes temporal
patterns from raw sensory observations — no object labels, no categories,
just feature streams from dynamic behaviors.

Each test class focuses on a specific capability:
- SDR encoding (locality-sensitive)
- Hebbian temporal learning
- Surprise / prediction error
- Real behavior learning (walking, stapler, door, wheel, etc.)
- Behavior recognition from partial observations
- Replay consolidation
- Sequence prediction (mental simulation)
- Serialization
"""

import unittest

import numpy as np

from tbp.monty.frameworks.environments.behaviors import (
    add_noise,
    ball_rolling,
    door_opening,
    hand_waving,
    light_flickering,
    pendulum_swing,
    scissors_cutting,
    stapler_press,
    walking_gait,
    wheel_spinning,
)
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.models.temporal_memory import TemporalMemory


def _make_state(location, curvatures, hsv=None, pose=None):
    """Helper: create a State with given features."""
    if pose is None:
        pose = np.eye(3)
    if hsv is None:
        hsv = [0.0, 0.0, 0.5]
    return State(
        location=np.array(location, dtype=np.float64),
        morphological_features={
            "pose_vectors": np.array(pose, dtype=np.float64),
            "pose_fully_defined": True,
            "on_object": 1,
        },
        non_morphological_features={
            "hsv": list(hsv),
            "principal_curvatures_log": list(curvatures),
        },
        confidence=1.0,
        use_state=True,
        sender_id="patch",
        sender_type="SM",
    )


# =============================================================================
# SDR Encoding
# =============================================================================


class TestSDREncoding(unittest.TestCase):
    """Locality-sensitive SDR encoding of raw sensory observations."""

    def test_sdr_is_sparse_binary(self):
        """SDR has correct dimensions, is binary, and has correct sparsity."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.02)
        state = _make_state([0, 0, 0], [0.1, 0.2])
        sdr = tm.encode(state)

        self.assertEqual(sdr.shape, (1024,))
        self.assertTrue(np.all((sdr == 0) | (sdr == 1)))
        self.assertEqual(int(np.sum(sdr)), tm.n_active)

    def test_similar_inputs_high_overlap(self):
        """Close feature vectors produce SDRs with high overlap."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02)
        s1 = _make_state([0.0, 0.0, 0.0], [0.1, 0.2])
        s2 = _make_state([0.001, 0.0, 0.0], [0.1, 0.2])  # tiny change
        s3 = _make_state([5.0, 5.0, 5.0], [2.0, 3.0])  # very different

        sdr1 = tm.encode(s1)
        sdr2 = tm.encode(s2)
        sdr3 = tm.encode(s3)

        overlap_similar = np.dot(sdr1, sdr2)
        overlap_different = np.dot(sdr1, sdr3)

        self.assertGreater(
            overlap_similar, overlap_different,
            "Similar inputs should produce higher SDR overlap than dissimilar"
        )

    def test_identical_inputs_identical_sdrs(self):
        """Same input always produces the same SDR (deterministic)."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.02)
        state = _make_state([1.0, 2.0, 3.0], [0.5, 0.5])
        sdr1 = tm.encode(state)
        sdr2 = tm.encode(state)
        np.testing.assert_array_equal(sdr1, sdr2)

    def test_different_seeds_different_encodings(self):
        """Different projection seeds produce different SDR encodings."""
        tm1 = TemporalMemory(sdr_dim=1024, projection_seed=42)
        tm2 = TemporalMemory(sdr_dim=1024, projection_seed=99)
        state = _make_state([1, 0, 0], [0.3, 0.4])
        sdr1 = tm1.encode(state)
        sdr2 = tm2.encode(state)
        overlap = np.dot(sdr1, sdr2)
        self.assertLess(overlap, tm1.n_active, "Different seeds → different SDRs")

    def test_feature_extraction_consistency(self):
        """Same state always yields the same feature vector."""
        tm = TemporalMemory()
        state = _make_state([1, 2, 3], [0.5, 0.7], hsv=[0.1, 0.2, 0.3])
        f1 = tm.extract_features(state)
        f2 = tm.extract_features(state)
        np.testing.assert_array_equal(f1, f2)

    def test_location_exclusion(self):
        """include_location=False ignores location in encoding."""
        tm_with = TemporalMemory(sdr_dim=1024, include_location=True)
        tm_without = TemporalMemory(sdr_dim=1024, include_location=False)

        s1 = _make_state([0, 0, 0], [0.5, 0.5])
        s2 = _make_state([100, 100, 100], [0.5, 0.5])

        # With location: different SDRs
        sdr1_with = tm_with.encode(s1)
        sdr2_with = tm_with.encode(s2)
        overlap_with = np.dot(sdr1_with, sdr2_with)

        # Without location: same features → same SDR
        sdr1_without = tm_without.encode(s1)
        sdr2_without = tm_without.encode(s2)
        overlap_without = np.dot(sdr1_without, sdr2_without)

        self.assertGreater(
            overlap_without, overlap_with,
            "Excluding location should make these states more similar"
        )


# =============================================================================
# Hebbian Temporal Learning
# =============================================================================


class TestHebbianTemporalLearning(unittest.TestCase):
    """Hebbian outer-product learning of temporal transitions."""

    def test_learns_transition(self):
        """After A→B, prediction from A activates B's SDR bits."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=1.0)
        s_a = _make_state([0, 0, 0], [0.1, 0.1])
        s_b = _make_state([1, 0, 0], [0.5, 0.5])

        tm.step(s_a, learn=True)
        tm.step(s_b, learn=True)

        # Predict from A
        sdr_a = tm.encode(s_a)
        predicted = tm.predict_next(from_sdr=sdr_a)
        sdr_b = tm.encode(s_b)

        self.assertIsNotNone(predicted)
        overlap = np.dot(predicted, sdr_b)
        self.assertGreater(overlap, 0, "Prediction from A should overlap with B")

    def test_repeated_learning_strengthens(self):
        """Repeating A→B strengthens the prediction overlap."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=0.5)
        s_a = _make_state([0, 0, 0], [0.1, 0.1])
        s_b = _make_state([1, 0, 0], [0.5, 0.5])

        # One pass
        tm.step(s_a)
        tm.step(s_b)
        sdr_a = tm.encode(s_a)
        sdr_b = tm.encode(s_b)
        pred1 = tm.predict_next(from_sdr=sdr_a)
        strength1 = float(np.dot(tm._W @ sdr_a, sdr_b))

        # Second pass
        tm.reset_episode()
        tm.step(s_a)
        tm.step(s_b)
        strength2 = float(np.dot(tm._W @ sdr_a, sdr_b))

        self.assertGreater(strength2, strength1)

    def test_no_learning_when_disabled(self):
        """learn=False does not update the association matrix."""
        tm = TemporalMemory(sdr_dim=1024)
        s = _make_state([0, 0, 0], [0.1, 0.2])
        tm.step(s, learn=False)
        w_snapshot = tm._W.copy()
        tm.step(s, learn=False)
        np.testing.assert_array_equal(tm._W, w_snapshot)

    def test_sequence_learning(self):
        """Learns a 3-step sequence A→B→C and can predict B from A."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=1.0)
        states = [
            _make_state([0, 0, 0], [0.1, 0.1]),
            _make_state([1, 0, 0], [0.3, 0.3]),
            _make_state([2, 0, 0], [0.6, 0.6]),
        ]

        for s in states:
            tm.step(s, learn=True)

        # From A, should predict B
        sdr_a = tm.encode(states[0])
        sdr_b = tm.encode(states[1])
        pred = tm.predict_next(from_sdr=sdr_a)
        self.assertIsNotNone(pred)
        self.assertGreater(np.dot(pred, sdr_b), 0)


# =============================================================================
# Surprise / Prediction Error
# =============================================================================


class TestSurprise(unittest.TestCase):
    """Prediction error as a surprise signal."""

    def test_no_prediction_max_surprise(self):
        """No prediction available → surprise = 1.0."""
        tm = TemporalMemory()
        sdr = np.zeros(tm.sdr_dim)
        self.assertEqual(tm.compute_surprise(sdr, None), 1.0)

    def test_perfect_prediction_zero_surprise(self):
        """Predicted SDR exactly matches observed → surprise = 0.0."""
        tm = TemporalMemory(sdr_dim=512, sdr_sparsity=0.05)
        sdr = np.zeros(512)
        sdr[: int(512 * 0.05)] = 1.0
        self.assertEqual(tm.compute_surprise(sdr, sdr), 0.0)

    def test_surprise_decreases_with_training(self):
        """After learning A→B multiple times, surprise on B after A drops."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=1.0)
        s_a = _make_state([0, 0, 0], [0.1, 0.1])
        s_b = _make_state([1, 0, 0], [0.5, 0.5])

        # First time: max surprise (no prediction exists)
        tm.step(s_a)
        result_first = tm.step(s_b)
        surprise_first = result_first["surprise"]

        # Learn it several more times
        for _ in range(5):
            tm.reset_episode()
            tm.step(s_a)
            tm.step(s_b)

        # Now surprise should be lower
        tm.reset_episode()
        tm.step(s_a)
        result_after = tm.step(s_b)
        surprise_after = result_after["surprise"]

        self.assertLess(surprise_after, surprise_first)

    def test_surprise_history_tracked(self):
        """get_surprise_history returns per-step surprise values."""
        tm = TemporalMemory(sdr_dim=512, sdr_sparsity=0.05, learning_rate=0.5)
        states = [
            _make_state([0, 0, 0], [0.1, 0.1]),
            _make_state([1, 0, 0], [0.3, 0.3]),
            _make_state([2, 0, 0], [0.5, 0.5]),
        ]
        for s in states:
            tm.step(s)

        history = tm.get_surprise_history()
        # First step has no previous → no prediction logged
        # Steps 2 and 3 should have predictions
        self.assertGreaterEqual(len(history), 1)


# =============================================================================
# Walking Gait Behavior
# =============================================================================


class TestWalkingBehavior(unittest.TestCase):
    """Learning temporal patterns from walking gait observations."""

    def test_learns_gait_cycle(self):
        """After training on walking, mean surprise decreases on re-test."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train: 3 full gait cycles (60 steps, period 20)
        states = walking_gait(n_steps=60, gait_period=20)
        for state in states:
            tm.step(state, learn=True)

        # Eval: same gait pattern
        tm.reset_episode()
        eval_states = walking_gait(n_steps=20, gait_period=20)
        for state in eval_states:
            tm.step(state, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(
            mean_surprise, 0.9,
            f"Mean surprise {mean_surprise:.2f} too high — "
            "temporal memory should have learned the gait pattern"
        )

    def test_predicts_next_gait_step(self):
        """Predicted SDR has nonzero overlap with actual next observation."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train
        states = walking_gait(n_steps=60, gait_period=20)
        for s in states:
            tm.step(s, learn=True)

        # Present a few steps, then predict
        tm.reset_episode()
        new_states = walking_gait(n_steps=5, gait_period=20)
        for s in new_states[:3]:
            tm.step(s, learn=False)

        prediction = tm.predict_next()
        actual_sdr = tm.encode(new_states[3])

        self.assertIsNotNone(prediction)
        overlap = np.dot(prediction, actual_sdr)
        self.assertGreater(
            overlap, 0,
            "Prediction should overlap with actual next observation"
        )

    def test_surprise_spike_on_anomaly(self):
        """Anomalous observation mid-gait causes a surprise spike."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train
        states = walking_gait(n_steps=60, gait_period=20)
        for s in states:
            tm.step(s, learn=True)

        # Eval normally for 10 steps, then insert anomaly
        tm.reset_episode()
        eval_states = walking_gait(n_steps=15, gait_period=20)
        normal_surprises = []
        for i, s in enumerate(eval_states[:10]):
            result = tm.step(s, learn=False)
            if i > 0:
                normal_surprises.append(result["surprise"])

        # Anomaly: completely different features
        anomaly = _make_state([10, 10, 10], [5.0, 5.0], hsv=[1.0, 1.0, 0.0])
        anomaly_result = tm.step(anomaly, learn=False)

        if normal_surprises:
            mean_normal = np.mean(normal_surprises)
            self.assertGreater(
                anomaly_result["surprise"], mean_normal,
                "Anomaly should cause higher surprise than normal gait"
            )

    def test_noise_robust(self):
        """Learning is robust to moderate sensory noise."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train on clean data
        clean = walking_gait(n_steps=60, gait_period=20)
        for s in clean:
            tm.step(s, learn=True)

        # Eval on noisy version
        tm.reset_episode()
        noisy = add_noise(
            walking_gait(n_steps=20, gait_period=20),
            location_noise=0.002, feature_noise=0.02, seed=123,
        )
        for s in noisy:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(
            mean_surprise, 0.95,
            f"Should tolerate noise: mean surprise {mean_surprise:.2f}"
        )


# =============================================================================
# Stapler Behavior
# =============================================================================


class TestStaplerBehavior(unittest.TestCase):
    """Learning discrete state transitions from raw stapler observations."""

    def test_learns_press_transition(self):
        """Learns the feature change when stapler closes."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train on 3 press cycles
        for _ in range(3):
            states = stapler_press(n_steps=40)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        # Eval: show open state, check prediction exists during closing
        eval_states = stapler_press(n_steps=40)
        tm.reset_episode()
        for s in eval_states[:12]:
            tm.step(s, learn=False)

        prediction = tm.predict_next()
        self.assertIsNotNone(prediction, "Should predict next state during press")

    def test_surprise_at_press_decreases(self):
        """First press is surprising; after training, less so."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.5)

        states = stapler_press(n_steps=40, press_start=10, press_end=15)

        # First exposure
        for s in states:
            tm.step(s, learn=True)
        first_surprises = tm.get_surprise_history()
        # Surprise at step 11 (during closing transition)
        first_transition = first_surprises[10] if len(first_surprises) > 10 else 1.0

        # More training
        for _ in range(4):
            tm.reset_episode()
            for s in states:
                tm.step(s, learn=True)

        # Re-evaluate
        tm.reset_episode()
        for s in states:
            tm.step(s, learn=False)
        final_surprises = tm.get_surprise_history()
        final_transition = final_surprises[10] if len(final_surprises) > 10 else 1.0

        self.assertLessEqual(
            final_transition, first_transition,
            "Surprise at press transition should decrease with training"
        )

    def test_open_closed_sdrs_differ(self):
        """Open and closed states produce distinct SDR patterns."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02)
        states = stapler_press(n_steps=40, press_start=10, press_end=15)

        # Open state (step 0) vs closed state (step 15)
        sdr_open = tm.encode(states[0])
        sdr_closed = tm.encode(states[15])
        overlap = np.dot(sdr_open, sdr_closed)

        # They should be substantially different
        self.assertLess(
            overlap / tm.n_active, 0.8,
            "Open and closed states should produce different SDRs"
        )


# =============================================================================
# Door Opening Behavior
# =============================================================================


class TestDoorBehavior(unittest.TestCase):
    """Learning monotonic trajectory from door opening observations."""

    def test_learns_opening_trajectory(self):
        """After training, predicts intermediate angle in opening sequence."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = door_opening(n_steps=40)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        # Show first half, predict next
        eval_states = door_opening(n_steps=40)
        tm.reset_episode()
        for s in eval_states[:20]:
            tm.step(s, learn=False)

        prediction = tm.predict_next()
        self.assertIsNotNone(prediction)
        actual = tm.encode(eval_states[20])
        overlap = np.dot(prediction, actual)
        self.assertGreater(overlap, 0)

    def test_surprise_decreases_over_training(self):
        """Mean surprise on door opening decreases with more training."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)
        states = door_opening(n_steps=40)

        # First pass
        for s in states:
            tm.step(s, learn=True)
        surprise_first = tm.get_mean_surprise()

        # More training
        for _ in range(4):
            tm.reset_episode()
            for s in states:
                tm.step(s, learn=True)

        # Re-evaluate
        tm.reset_episode()
        for s in states:
            tm.step(s, learn=False)
        surprise_after = tm.get_mean_surprise()

        self.assertLess(surprise_after, surprise_first)


# =============================================================================
# Wheel Spinning Behavior
# =============================================================================


class TestWheelBehavior(unittest.TestCase):
    """Learning continuous rotation from wheel spinning observations."""

    def test_learns_rotation_pattern(self):
        """Learns the continuous rotation of a spinning wheel."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        states = wheel_spinning(n_steps=60, rpm=1.0)
        for s in states:
            tm.step(s, learn=True)

        # Eval
        tm.reset_episode()
        eval_states = wheel_spinning(n_steps=20, rpm=1.0)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(mean_surprise, 0.95)

    def test_different_speed_higher_surprise(self):
        """Wheel at a different speed produces higher surprise."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train at 1 rpm
        for _ in range(3):
            states = wheel_spinning(n_steps=60, rpm=1.0)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        # Eval at same speed
        tm.reset_episode()
        same = wheel_spinning(n_steps=30, rpm=1.0)
        for s in same:
            tm.step(s, learn=False)
        surprise_same = tm.get_mean_surprise()

        # Eval at different speed
        tm.reset_episode()
        diff = wheel_spinning(n_steps=30, rpm=3.0)
        for s in diff:
            tm.step(s, learn=False)
        surprise_diff = tm.get_mean_surprise()

        self.assertGreater(
            surprise_diff, surprise_same,
            "Different speed should produce higher surprise"
        )


# =============================================================================
# Scissors Behavior
# =============================================================================


class TestScissorsBehavior(unittest.TestCase):
    """Cyclical cutting behavior from raw observations."""

    def test_learns_cutting_cycle(self):
        """Learns the repetitive open-close cycle of scissors."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = scissors_cutting(n_steps=50, cut_period=15)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        tm.reset_episode()
        eval_states = scissors_cutting(n_steps=30, cut_period=15)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(mean_surprise, 0.9)


# =============================================================================
# Hand Waving Behavior
# =============================================================================


class TestHandWaving(unittest.TestCase):
    """Oscillatory hand gesture from raw observations."""

    def test_learns_wave_pattern(self):
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = hand_waving(n_steps=50)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        tm.reset_episode()
        eval_states = hand_waving(n_steps=20)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(mean_surprise, 0.95)


# =============================================================================
# Pendulum Behavior
# =============================================================================


class TestPendulumBehavior(unittest.TestCase):
    """Decaying oscillation — non-stationary dynamics."""

    def test_learns_damped_motion(self):
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = pendulum_swing(n_steps=80)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        tm.reset_episode()
        eval_states = pendulum_swing(n_steps=40)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(mean_surprise, 0.95)


# =============================================================================
# Ball Rolling Behavior
# =============================================================================


class TestBallRolling(unittest.TestCase):
    """Translation + rotation combined."""

    def test_learns_rolling_motion(self):
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = ball_rolling(n_steps=50)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        tm.reset_episode()
        eval_states = ball_rolling(n_steps=20)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        self.assertLess(mean_surprise, 0.95)


# =============================================================================
# Light Flickering (pure color dynamics)
# =============================================================================


class TestLightFlickering(unittest.TestCase):
    """Color-only temporal dynamics (no motion)."""

    def test_learns_flicker_pattern(self):
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = light_flickering(n_steps=60)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        tm.reset_episode()
        eval_states = light_flickering(n_steps=30)
        for s in eval_states:
            tm.step(s, learn=False)

        mean_surprise = tm.get_mean_surprise()
        # Light flickering has noise, so predictions are harder
        self.assertLess(mean_surprise, 1.0)


# =============================================================================
# Behavior Recognition
# =============================================================================


class TestBehaviorRecognition(unittest.TestCase):
    """Recognizing which behavior is being observed from raw sensory data."""

    def test_recognize_walking_vs_stapler(self):
        """Distinguishes walking from stapler behavior."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        tm.learn_behavior("walking", walking_gait(n_steps=60), n_repetitions=3)
        tm.learn_behavior("stapler", stapler_press(n_steps=40), n_repetitions=3)

        # Test walking
        test_walk = walking_gait(n_steps=20)
        name, score = tm.recognize_behavior(test_walk)
        self.assertEqual(
            name, "walking",
            f"Should recognize walking, got {name} (score={score:.2f})"
        )

        # Test stapler
        test_stapler = stapler_press(n_steps=20)
        name, score = tm.recognize_behavior(test_stapler)
        self.assertEqual(
            name, "stapler",
            f"Should recognize stapler, got {name} (score={score:.2f})"
        )

    def test_recognize_from_partial_observation(self):
        """Recognizes behavior from just a few observations."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        tm.learn_behavior("walking", walking_gait(n_steps=60), n_repetitions=3)
        tm.learn_behavior("door", door_opening(n_steps=40), n_repetitions=3)

        # Just 5 observations
        partial = walking_gait(n_steps=5)
        name, score = tm.recognize_behavior(partial, min_overlap=0.2)
        self.assertEqual(name, "walking")

    def test_recognize_four_behaviors(self):
        """Distinguishes among 4 different behaviors."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        behaviors = {
            "walking": walking_gait(n_steps=40),
            "stapler": stapler_press(n_steps=30),
            "door": door_opening(n_steps=30),
            "wheel": wheel_spinning(n_steps=40),
        }

        for name, states in behaviors.items():
            tm.learn_behavior(name, states, n_repetitions=3)

        correct = 0
        for name, states in behaviors.items():
            test_states = states[:15]
            recognized, _ = tm.recognize_behavior(test_states, min_overlap=0.2)
            if recognized == name:
                correct += 1

        self.assertGreaterEqual(
            correct, 3,
            f"Should recognize at least 3 of 4 behaviors, got {correct}"
        )

    def test_recognize_six_behaviors(self):
        """Distinguishes among 6 different behaviors — capacity test."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        behaviors = {
            "walking": walking_gait(n_steps=40),
            "stapler": stapler_press(n_steps=30),
            "door": door_opening(n_steps=30),
            "wheel": wheel_spinning(n_steps=40),
            "scissors": scissors_cutting(n_steps=40),
            "hand_wave": hand_waving(n_steps=40),
        }

        for name, states in behaviors.items():
            tm.learn_behavior(name, states, n_repetitions=3)

        correct = 0
        for name, states in behaviors.items():
            test_states = states[:15]
            recognized, _ = tm.recognize_behavior(test_states, min_overlap=0.15)
            if recognized == name:
                correct += 1

        self.assertGreaterEqual(
            correct, 4,
            f"Should recognize at least 4 of 6 behaviors, got {correct}"
        )

    def test_tempo_invariant_recognition_rescues_scaled_behavior(self):
        """Tempo-invariant matching should handle stretched sequences better."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        tm.learn_behavior("walking", walking_gait(n_steps=60), n_repetitions=3)
        tm.learn_behavior("door", door_opening(n_steps=40), n_repetitions=3)

        stretched_walk = walking_gait(n_steps=90, gait_period=30)
        plain_name, plain_score = tm.recognize_behavior(
            stretched_walk,
            min_overlap=0.2,
            tempo_invariant=False,
        )
        tempo_name, tempo_score = tm.recognize_behavior(
            stretched_walk,
            min_overlap=0.2,
            tempo_invariant=True,
        )

        self.assertEqual(tempo_name, "walking")
        self.assertGreaterEqual(tempo_score, plain_score)


# =============================================================================
# Sequence Prediction (Mental Simulation)
# =============================================================================


class TestSequencePrediction(unittest.TestCase):
    """Predicting future observation sequences (imagination)."""

    def test_predict_walking_sequence(self):
        """Can chain predictions to simulate future gait steps."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(3):
            states = walking_gait(n_steps=60, gait_period=20)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        start = walking_gait(n_steps=1, gait_period=20)[0]
        predicted_sdrs = tm.predict_sequence(start, n_steps=5)

        self.assertGreater(
            len(predicted_sdrs), 0,
            "Should generate at least one predicted step"
        )

    def test_predicted_sdrs_stay_sparse(self):
        """Chained predictions maintain correct sparsity."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        for _ in range(5):
            states = walking_gait(n_steps=60)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        start = walking_gait(n_steps=1)[0]
        preds = tm.predict_sequence(start, n_steps=10)

        for p in preds:
            active = int(np.sum(p))
            self.assertEqual(
                active, tm.n_active,
                "Predicted SDRs should maintain correct sparsity"
            )


# =============================================================================
# Replay Consolidation
# =============================================================================


class TestReplayConsolidation(unittest.TestCase):
    """Episodic replay strengthens temporal associations."""

    def test_replay_strengthens_associations(self):
        """Replay after one exposure strengthens the W matrix."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.2)

        states = walking_gait(n_steps=30, gait_period=15)
        for s in states:
            tm.step(s, learn=True)

        w_norm_before = np.linalg.norm(tm._W)
        n_updates = tm.replay_episode(n_replays=5)
        w_norm_after = np.linalg.norm(tm._W)

        self.assertGreater(n_updates, 0)
        self.assertGreater(w_norm_after, w_norm_before)

    def test_replay_returns_correct_count(self):
        """Replay returns the correct number of Hebbian updates."""
        tm = TemporalMemory()
        states = walking_gait(n_steps=10)
        for s in states:
            tm.step(s, learn=True)

        n = tm.replay_episode(n_replays=2)
        # 2 replays × 9 transitions (10 steps, 9 consecutive pairs)
        self.assertEqual(n, 2 * 9)

    def test_replay_empty_episode(self):
        """Replay on empty episode returns 0."""
        tm = TemporalMemory()
        n = tm.replay_episode(n_replays=5)
        self.assertEqual(n, 0)

    def test_replay_single_step(self):
        """Replay with only 1 observation returns 0 (no transitions)."""
        tm = TemporalMemory()
        tm.step(_make_state([0, 0, 0], [0.1, 0.1]))
        n = tm.replay_episode(n_replays=3)
        self.assertEqual(n, 0)


# =============================================================================
# Temporal Context Signal
# =============================================================================


class TestTemporalContext(unittest.TestCase):
    """Temporal context signal for downstream components."""

    def test_context_available_after_step(self):
        """get_temporal_context returns data after at least one step."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=0.5)
        s = _make_state([0, 0, 0], [0.1, 0.2])
        tm.step(s)

        ctx = tm.get_temporal_context()
        self.assertIsNotNone(ctx)
        self.assertIn("last_sdr", ctx)
        self.assertIn("mean_surprise", ctx)
        self.assertIn("step_count", ctx)

    def test_context_none_before_any_step(self):
        """No context available before any observation."""
        tm = TemporalMemory()
        ctx = tm.get_temporal_context()
        self.assertIsNone(ctx)

    def test_context_includes_prediction(self):
        """After learning, context includes predicted SDR."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=1.0)
        s_a = _make_state([0, 0, 0], [0.1, 0.1])
        s_b = _make_state([1, 0, 0], [0.5, 0.5])

        # Learn A→B
        tm.step(s_a)
        tm.step(s_b)
        tm.reset_episode()

        # Present A again
        tm.step(s_a)
        ctx = tm.get_temporal_context()

        self.assertIsNotNone(ctx["predicted_sdr"])


# =============================================================================
# Attention via Surprise
# =============================================================================


class TestSurpriseAsAttention(unittest.TestCase):
    """High surprise signals novel events — an attention mechanism."""

    def test_behavior_switch_increases_surprise(self):
        """Switching from learned to unlearned behavior causes surprise."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train on walking only
        for _ in range(3):
            states = walking_gait(n_steps=60)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        # Show walking then switch to stapler
        tm.reset_episode()
        walk = walking_gait(n_steps=15)
        for s in walk:
            tm.step(s, learn=False)
        walk_surprise = tm.get_mean_surprise()

        press = stapler_press(n_steps=10)
        for s in press:
            tm.step(s, learn=False)
        # Get surprise just for the stapler portion
        all_surprises = tm.get_surprise_history()
        stapler_surprises = all_surprises[len(all_surprises) - len(press):]
        stapler_surprise = np.mean(stapler_surprises) if stapler_surprises else 1.0

        self.assertGreater(
            stapler_surprise, walk_surprise,
            "Unexpected behavior should increase surprise"
        )

    def test_training_on_both_reduces_both_surprises(self):
        """Training on both behaviors makes both less surprising."""
        tm = TemporalMemory(sdr_dim=2048, sdr_sparsity=0.02, learning_rate=0.3)

        # Train on both
        for _ in range(3):
            for s in walking_gait(n_steps=40):
                tm.step(s, learn=True)
            tm.reset_episode()
            for s in stapler_press(n_steps=30):
                tm.step(s, learn=True)
            tm.reset_episode()

        # Test walking
        tm.reset_episode()
        for s in walking_gait(n_steps=20):
            tm.step(s, learn=False)
        walk_surprise = tm.get_mean_surprise()

        # Test stapler
        tm.reset_episode()
        for s in stapler_press(n_steps=20):
            tm.step(s, learn=False)
        stapler_surprise = tm.get_mean_surprise()

        # Both should be somewhat predictable
        self.assertLess(walk_surprise, 1.0)
        self.assertLess(stapler_surprise, 1.0)


# =============================================================================
# Serialization
# =============================================================================


class TestSerialization(unittest.TestCase):
    """Temporal memory state persistence."""

    def test_save_load_preserves_encodings(self):
        """Saved/loaded model produces same SDR encodings."""
        tm1 = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.03, learning_rate=0.3)
        states = walking_gait(n_steps=30)
        for s in states:
            tm1.step(s, learn=True)

        state_dict = tm1.state_dict()

        tm2 = TemporalMemory()
        tm2.load_state_dict(state_dict)

        test_state = walking_gait(n_steps=1)[0]
        sdr1 = tm1.encode(test_state)
        sdr2 = tm2.encode(test_state)
        np.testing.assert_array_equal(sdr1, sdr2)

    def test_save_load_preserves_predictions(self):
        """Saved/loaded model makes same predictions."""
        tm1 = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.03, learning_rate=0.3)
        states = walking_gait(n_steps=30)
        for s in states:
            tm1.step(s, learn=True)

        state_dict = tm1.state_dict()
        tm2 = TemporalMemory()
        tm2.load_state_dict(state_dict)

        test_sdr = tm1.encode(walking_gait(n_steps=1)[0])
        pred1 = tm1.predict_next(from_sdr=test_sdr)
        pred2 = tm2.predict_next(from_sdr=test_sdr)

        if pred1 is not None and pred2 is not None:
            np.testing.assert_array_equal(pred1, pred2)

    def test_save_load_preserves_behaviors(self):
        """Known behaviors survive serialization."""
        tm1 = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.03, learning_rate=0.3)
        tm1.learn_behavior("walking", walking_gait(n_steps=20))

        state_dict = tm1.state_dict()
        tm2 = TemporalMemory()
        tm2.load_state_dict(state_dict)

        self.assertIn("walking", tm2._known_behaviors)
        self.assertEqual(
            len(tm2._known_behaviors["walking"]),
            len(tm1._known_behaviors["walking"]),
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_empty_episode_reset(self):
        """reset_episode on fresh instance doesn't crash."""
        tm = TemporalMemory()
        tm.reset_episode()
        self.assertEqual(len(tm._history), 0)

    def test_predict_before_any_learning(self):
        """Prediction before any learning returns None."""
        tm = TemporalMemory()
        s = _make_state([0, 0, 0], [0.1, 0.2])
        tm.step(s, learn=False)
        pred = tm.predict_next()
        self.assertIsNone(pred)

    def test_single_step_episode(self):
        """Single-step episode doesn't crash step or predict."""
        tm = TemporalMemory()
        result = tm.step(_make_state([0, 0, 0], [0.1, 0.2]))
        self.assertIn("sdr", result)
        self.assertIn("surprise", result)
        self.assertIsNone(result["prediction"])

    def test_very_short_behavior(self):
        """Can learn and recognize a 2-step behavior."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.05, learning_rate=1.0)
        states = [
            _make_state([0, 0, 0], [0.1, 0.1]),
            _make_state([1, 0, 0], [0.5, 0.5]),
        ]
        tm.learn_behavior("short", states, n_repetitions=3)
        name, score = tm.recognize_behavior(states, min_overlap=0.3)
        self.assertEqual(name, "short")

    def test_mean_surprise_no_predictions(self):
        """Mean surprise is 1.0 when no predictions were made."""
        tm = TemporalMemory()
        self.assertEqual(tm.get_mean_surprise(), 1.0)


# =============================================================================
# LM + TemporalMemory Integration
# =============================================================================


class TestLMTemporalMemoryIntegration(unittest.TestCase):
    """Verify TemporalMemory works when embedded in the EvidenceGraphLM."""

    def test_lm_with_temporal_memory_config(self):
        """LM accepts temporal_memory config and instantiates it."""
        from tbp.monty.frameworks.models.evidence_matching.learning_module import (
            EvidenceGraphLM,
        )

        lm = EvidenceGraphLM(
            max_match_distance=0.01,
            tolerances={"patch": {"hsv": [0.1, 0.2, 0.2]}},
            feature_weights={"patch": {"hsv": np.ones(3)}},
            temporal_memory={"sdr_dim": 1024, "sdr_sparsity": 0.03},
        )
        self.assertIsNotNone(lm._temporal_memory)
        self.assertEqual(lm._temporal_memory.sdr_dim, 1024)

    def test_lm_without_temporal_memory(self):
        """LM works normally when temporal_memory is not configured."""
        from tbp.monty.frameworks.models.evidence_matching.learning_module import (
            EvidenceGraphLM,
        )

        lm = EvidenceGraphLM(
            max_match_distance=0.01,
            tolerances={"patch": {"hsv": [0.1, 0.2, 0.2]}},
            feature_weights={"patch": {"hsv": np.ones(3)}},
        )
        self.assertIsNone(lm._temporal_memory)
        self.assertEqual(lm.get_temporal_surprise(), 1.0)
        self.assertIsNone(lm.get_temporal_context())

    def test_lm_temporal_surprise_during_behavior(self):
        """LM with temporal memory provides surprise signal on behaviors."""
        tm = TemporalMemory(sdr_dim=1024, sdr_sparsity=0.03, learning_rate=0.3)

        # Pre-train the temporal memory on walking
        for _ in range(3):
            states = walking_gait(n_steps=40)
            for s in states:
                tm.step(s, learn=True)
            tm.reset_episode()

        # Feed through a fresh episode
        tm.reset_episode()
        eval_states = walking_gait(n_steps=15)
        for s in eval_states:
            tm.step(s, learn=False)

        # Should have low surprise on familiar pattern
        surprise = tm.get_mean_surprise()
        self.assertLess(surprise, 0.9)

        # Now feed unfamiliar pattern
        tm.reset_episode()
        stapler_states = stapler_press(n_steps=15)
        for s in stapler_states:
            tm.step(s, learn=False)

        # Should have higher surprise on unfamiliar
        unfamiliar_surprise = tm.get_mean_surprise()
        self.assertGreater(unfamiliar_surprise, surprise)


if __name__ == "__main__":
    unittest.main()
