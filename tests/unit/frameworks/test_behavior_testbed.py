# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for Phase 6: Behavior Testbed.

Tests the SyntheticBehaviorEnvironment and behavior config builders
at testbed difficulty levels 1-3.
"""

import unittest

import numpy as np

from tbp.monty.frameworks.config_utils.behavior_configs import (
    TESTBED_LEVELS,
    build_behavior_testbed,
    get_behavior_names,
)
from tbp.monty.frameworks.environments.synthetic_behavior_env import (
    SyntheticBehaviorEnvironment,
    make_behavior_sequences,
)
from tbp.monty.frameworks.models.states import State


class TestSyntheticBehaviorEnvironment(unittest.TestCase):
    """Test the synthetic behavior environment."""

    def _make_simple_sequence(self, name, n=5, x_offset=0.0):
        """Create a simple test sequence."""
        states = []
        for i in range(n):
            states.append(
                State(
                    location=np.array([x_offset + i * 0.01, 0.0, 0.0]),
                    morphological_features={
                        "pose_vectors": np.eye(3),
                        "pose_fully_defined": True,
                    },
                    non_morphological_features={"step": i},
                    confidence=1.0,
                    use_state=True,
                    sender_id=f"SM_{name}",
                    sender_type="SM",
                )
            )
        return states

    def test_add_and_list_sequences(self):
        env = SyntheticBehaviorEnvironment()
        env.add_sequence("walk", self._make_simple_sequence("walk", n=10))
        env.add_sequence("run", self._make_simple_sequence("run", n=5))
        self.assertEqual(sorted(env.get_sequence_names()), ["run", "walk"])

    def test_step_returns_states_in_order(self):
        env = SyntheticBehaviorEnvironment()
        seq = self._make_simple_sequence("test", n=3)
        env.add_sequence("test", seq)
        env.set_active_sequence("test")

        for i in range(3):
            state = env.step()
            self.assertIsNotNone(state)
            self.assertEqual(state.non_morphological_features["step"], i)

        # Exhausted
        self.assertIsNone(env.step())

    def test_reset_replays_from_start(self):
        env = SyntheticBehaviorEnvironment()
        env.add_sequence("test", self._make_simple_sequence("test", n=3))
        env.set_active_sequence("test")

        env.step()
        env.step()
        env.reset()

        state = env.step()
        self.assertEqual(state.non_morphological_features["step"], 0)

    def test_is_done(self):
        env = SyntheticBehaviorEnvironment()
        env.add_sequence("test", self._make_simple_sequence("test", n=2))
        env.set_active_sequence("test")

        self.assertFalse(env.is_done)
        env.step()
        self.assertFalse(env.is_done)
        env.step()
        self.assertTrue(env.is_done)

    def test_get_remaining_steps(self):
        env = SyntheticBehaviorEnvironment()
        env.add_sequence("test", self._make_simple_sequence("test", n=5))
        env.set_active_sequence("test")

        self.assertEqual(env.get_remaining_steps(), 5)
        env.step()
        self.assertEqual(env.get_remaining_steps(), 4)

    def test_set_unknown_sequence_raises(self):
        env = SyntheticBehaviorEnvironment()
        with self.assertRaises(KeyError):
            env.set_active_sequence("nonexistent")

    def test_step_without_active_returns_none(self):
        env = SyntheticBehaviorEnvironment()
        self.assertIsNone(env.step())

    def test_switch_between_sequences(self):
        env = SyntheticBehaviorEnvironment()
        env.add_sequence("a", self._make_simple_sequence("a", n=3, x_offset=0.0))
        env.add_sequence("b", self._make_simple_sequence("b", n=3, x_offset=1.0))

        env.set_active_sequence("a")
        state_a = env.step()
        self.assertAlmostEqual(state_a.location[0], 0.0)

        env.set_active_sequence("b")
        state_b = env.step()
        self.assertAlmostEqual(state_b.location[0], 1.0)


class TestMakeBehaviorSequences(unittest.TestCase):
    """Test the make_behavior_sequences helper."""

    def test_generates_sequences(self):
        from tbp.monty.frameworks.environments.behaviors import (
            walking_gait,
            stapler_press,
        )

        sequences = make_behavior_sequences(
            behaviors={"walk": walking_gait, "stapler": stapler_press},
            n_steps=20,
        )
        self.assertEqual(sorted(sequences.keys()), ["stapler", "walk"])
        self.assertEqual(len(sequences["walk"]), 20)
        self.assertEqual(len(sequences["stapler"]), 20)

    def test_each_state_is_valid(self):
        from tbp.monty.frameworks.environments.behaviors import walking_gait

        sequences = make_behavior_sequences(
            behaviors={"walk": walking_gait},
            n_steps=10,
        )
        for state in sequences["walk"]:
            self.assertIsInstance(state, State)
            self.assertTrue(state.use_state)
            self.assertEqual(state.location.shape, (3,))


class TestBehaviorTestbedLevel1(unittest.TestCase):
    """Level 1: 3 simple repeated behaviors → recognition."""

    def test_build_level_1(self):
        env = build_behavior_testbed(level=1)
        names = env.get_sequence_names()
        self.assertEqual(len(names), 3)
        self.assertIn("walking", names)
        self.assertIn("stapler", names)
        self.assertIn("door", names)

    def test_level_1_sequences_not_empty(self):
        env = build_behavior_testbed(level=1, n_steps=20)
        for name in env.get_sequence_names():
            self.assertGreater(env.get_sequence_length(name), 0)

    def test_level_1_behaviors_are_distinct(self):
        """Different behaviors should produce distinct State sequences."""
        env = build_behavior_testbed(level=1, n_steps=20)

        # Collect locations from each behavior
        behavior_locations = {}
        for name in env.get_sequence_names():
            env.set_active_sequence(name)
            locs = []
            while not env.is_done:
                state = env.step()
                locs.append(state.location.copy())
            behavior_locations[name] = np.array(locs)

        # Walking should have monotonically increasing x
        walk_locs = behavior_locations["walking"]
        self.assertGreater(walk_locs[-1, 0], walk_locs[0, 0])

        # Stapler should return to start (periodic)
        stapler_locs = behavior_locations["stapler"]
        self.assertLess(
            np.linalg.norm(stapler_locs[-1] - stapler_locs[0]),
            np.linalg.norm(walk_locs[-1] - walk_locs[0]),
        )


class TestBehaviorTestbedLevel2(unittest.TestCase):
    """Level 2: Same behavior on 2 morphologies."""

    def test_build_level_2(self):
        env = build_behavior_testbed(level=2)
        names = env.get_sequence_names()
        self.assertEqual(len(names), 2)

    def test_level_2_replay_full(self):
        """Level 2 should replay all steps."""
        env = build_behavior_testbed(level=2, n_steps=15)
        for name in env.get_sequence_names():
            env.set_active_sequence(name)
            count = 0
            while not env.is_done:
                state = env.step()
                self.assertIsNotNone(state)
                count += 1
            self.assertEqual(count, 15)


class TestBehaviorTestbedLevel3(unittest.TestCase):
    """Level 3: 1 object with 2 behaviors at different locations."""

    def test_build_level_3(self):
        env = build_behavior_testbed(level=3)
        names = env.get_sequence_names()
        self.assertIn("walking_normal", names)
        self.assertIn("walking_fast", names)


class TestBehaviorTestbedConfig(unittest.TestCase):
    """Test config helpers."""

    def test_get_behavior_names(self):
        names = get_behavior_names(level=1)
        self.assertIn("walking", names)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            build_behavior_testbed(level=99)

    def test_custom_n_steps(self):
        env = build_behavior_testbed(level=1, n_steps=5)
        for name in env.get_sequence_names():
            self.assertEqual(env.get_sequence_length(name), 5)


class TestBehaviorRecognitionWithLM(unittest.TestCase):
    """Test that EvidenceGraphLM can learn and distinguish behaviors.

    Trains the LM on walking and stapler behaviors, then tests that it
    correctly identifies which behavior is being observed.
    """

    def _make_lm(self):
        """Create an EvidenceGraphLM configured for behavior recognition.

        Uses max_graph_size=1.0 since behaviors span ~0.2-0.6 units.
        With num_model_voxels_per_dim=100, voxel size = 0.01, giving
        enough spatial resolution to learn distinct behavior patterns.
        """
        from tbp.monty.frameworks.models.evidence_matching.learning_module import (
            EvidenceGraphLM,
        )

        return EvidenceGraphLM(
            max_match_distance=0.01,
            tolerances={
                "patch": {
                    "hsv": [0.2, 1, 1],
                    "principal_curvatures_log": [2, 2],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1, 0, 0]),
                }
            },
            max_graph_size=1.0,
            num_model_voxels_per_dim=100,
            hypotheses_updater_args=dict(
                initial_possible_poses="informed",
            ),
        )

    def _train_behavior(self, lm, obs_states, object_name):
        """Train the LM on a behavior sequence."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.experiments.mode import ExperimentMode

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        lm.mode = ExperimentMode.TRAIN
        lm.pre_episode(
            primary_target={
                "object": object_name,
                "quat_rotation": [1, 0, 0, 0],
            }
        )

        for obs in obs_states:
            lm.exploratory_step(ctx, [obs])

        lm.detected_object = object_name
        lm.detected_rotation_r = None
        lm.buffer.stats["detected_location_rel_body"] = (
            lm.buffer.get_current_location(input_channel="first")
        )
        lm.post_episode()

    def _match_behavior(self, lm, obs_states, n_steps=None):
        """Run matching on a behavior sequence and return MLH."""
        from tbp.monty.context import RuntimeContext
        from tbp.monty.frameworks.experiments.mode import ExperimentMode

        ctx = RuntimeContext(rng=np.random.RandomState(42))
        lm.mode = ExperimentMode.EVAL
        lm.pre_episode(
            primary_target={
                "object": "placeholder",
                "quat_rotation": [1, 0, 0, 0],
            }
        )

        n = n_steps or len(obs_states)
        for obs in obs_states[:n]:
            lm.add_lm_processing_to_buffer_stats(lm_processed=True)
            lm.matching_step(ctx, [obs])

        return lm.get_current_mlh()

    def test_lm_learns_two_behaviors(self):
        """LM can learn walking and stapler as separate objects."""
        from tbp.monty.frameworks.environments.behaviors import (
            walking_gait,
            stapler_press,
        )

        lm = self._make_lm()
        walk_states = walking_gait(n_steps=20)
        stapler_states = stapler_press(n_steps=20)

        self._train_behavior(lm, walk_states, "walking")
        self._train_behavior(lm, stapler_states, "stapler")

        known = lm.get_all_known_object_ids()
        self.assertEqual(len(known), 2)
        self.assertIn("walking", known)
        self.assertIn("stapler", known)

    def test_lm_recognizes_walking(self):
        """LM correctly identifies walking behavior."""
        from tbp.monty.frameworks.environments.behaviors import (
            walking_gait,
            stapler_press,
        )

        lm = self._make_lm()
        self._train_behavior(lm, walking_gait(n_steps=20), "walking")
        self._train_behavior(lm, stapler_press(n_steps=20), "stapler")

        # Test with a new walking sequence (same generator, should match)
        test_walk = walking_gait(n_steps=10)
        mlh = self._match_behavior(lm, test_walk)
        self.assertEqual(
            mlh["graph_id"], "walking",
            f"Should recognize walking, got {mlh['graph_id']}"
        )

    def test_lm_walking_not_confused_with_stapler(self):
        """Walking input is never confused for stapler.

        Stapler (fixed location) can't be reliably identified by
        location-based matching alone — that requires the behavior
        column with ChangeDetectingSM. But walking, with its clear
        spatial structure, should always be recognized even when
        stapler is in memory.
        """
        from tbp.monty.frameworks.environments.behaviors import (
            walking_gait,
            stapler_press,
        )

        lm = self._make_lm()
        self._train_behavior(lm, walking_gait(n_steps=20), "walking")
        self._train_behavior(lm, stapler_press(n_steps=20), "stapler")

        # Walking input → walking recognized (not confused with stapler)
        test_walk = walking_gait(n_steps=10)
        mlh = self._match_behavior(lm, test_walk)
        self.assertEqual(
            mlh["graph_id"], "walking",
            f"Should recognize walking, got {mlh['graph_id']}"
        )

    def test_lm_distinguishes_spatially_distinct_behaviors(self):
        """LM distinguishes behaviors with distinct spatial signatures.

        Walking (translational motion in x) and door opening (rotational
        arc in x-y) have distinct spatial structure that the location-based
        LM can resolve. Behaviors at a fixed location (like stapler) need
        a ChangeDetectingSM-based behavior column for discrimination —
        that is tested separately in the multi-column testbed.
        """
        from tbp.monty.frameworks.environments.behaviors import (
            door_opening,
            walking_gait,
        )

        lm = self._make_lm()
        self._train_behavior(lm, walking_gait(n_steps=40), "walking")
        self._train_behavior(lm, door_opening(n_steps=40), "door")

        self.assertEqual(len(lm.get_all_known_object_ids()), 2)

        # Walking input → walking recognized
        walk_test = walking_gait(n_steps=15)
        mlh_walk = self._match_behavior(lm, walk_test)
        self.assertEqual(
            mlh_walk["graph_id"], "walking",
            f"Should recognize walking, got {mlh_walk['graph_id']}"
        )

        # Door input → door recognized
        door_test = door_opening(n_steps=15)
        mlh_door = self._match_behavior(lm, door_test)
        self.assertEqual(
            mlh_door["graph_id"], "door",
            f"Should recognize door, got {mlh_door['graph_id']}"
        )

    def test_walking_evidence_higher_than_stapler_for_walk_input(self):
        """Walking evidence should exceed stapler evidence for walk input."""
        from tbp.monty.frameworks.environments.behaviors import (
            walking_gait,
            stapler_press,
        )

        lm = self._make_lm()
        self._train_behavior(lm, walking_gait(n_steps=20), "walking")
        self._train_behavior(lm, stapler_press(n_steps=20), "stapler")

        # Match walking sequence
        test_walk = walking_gait(n_steps=10)
        mlh = self._match_behavior(lm, test_walk)

        # Check that walking evidence exceeds stapler evidence
        walk_evidence = np.max(lm.evidence["walking"])
        stapler_evidence = np.max(lm.evidence["stapler"])
        self.assertGreater(
            walk_evidence, stapler_evidence,
            "Walking evidence should exceed stapler evidence for walking input"
        )


if __name__ == "__main__":
    unittest.main()
