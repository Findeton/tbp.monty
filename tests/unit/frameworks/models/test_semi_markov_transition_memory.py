"""Tests for the lightweight semi-Markov transition memory."""

import unittest

from tbp.monty.frameworks.models.semi_markov_transition_memory import (
    SemiMarkovTransitionMemory,
)


class TestSemiMarkovTransitionMemory(unittest.TestCase):
    def test_learns_transition_and_duration(self):
        memory = SemiMarkovTransitionMemory(min_stable_steps=1)

        for label in ["hinge:0", "hinge:0", "hinge:1", "hinge:1"]:
            memory.observe(label, learn=True)
        memory.finalize_episode(learn=True)

        self.assertEqual(memory.get_most_likely_next("hinge:0"), "hinge:1")
        self.assertGreaterEqual(memory.get_expected_duration("hinge:0"), 2.0)

    def test_prediction_switches_after_expected_duration(self):
        memory = SemiMarkovTransitionMemory(min_stable_steps=1)

        for label in ["hinge:0", "hinge:0", "hinge:1", "hinge:1"]:
            memory.observe(label, learn=True)
        memory.finalize_episode(learn=True)
        memory.reset_episode()

        info = memory.observe("hinge:0", learn=False)
        self.assertEqual(info["predicted_label"], "hinge:0")

        info = memory.observe("hinge:0", learn=False)
        self.assertEqual(info["predicted_label"], "hinge:1")

    def test_prediction_status_reports_confused_on_wrong_next_label(self):
        memory = SemiMarkovTransitionMemory(min_stable_steps=1)

        for label in ["hinge:0", "hinge:0", "hinge:1", "hinge:1"]:
            memory.observe(label, learn=True)
        memory.finalize_episode(learn=True)
        memory.reset_episode()

        memory.observe("hinge:0", learn=False)
        memory.observe("hinge:0", learn=False)
        info = memory.observe("hinge:0", learn=False)

        self.assertEqual(info["prediction_status"], "confused")
        self.assertFalse(info["event_detected"])

    def test_state_dict_round_trip(self):
        memory = SemiMarkovTransitionMemory(min_stable_steps=1)
        for label in ["a", "a", "b", "b"]:
            memory.observe(label, learn=True)
        memory.finalize_episode(learn=True)

        restored = SemiMarkovTransitionMemory()
        restored.load_state_dict(memory.state_dict())

        self.assertEqual(restored.get_most_likely_next("a"), "b")
        self.assertEqual(
            restored.get_transition_distribution("a"),
            memory.get_transition_distribution("a"),
        )


if __name__ == "__main__":
    unittest.main()