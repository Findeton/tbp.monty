"""Tests for the forward-only self-supervised semi-Markov memory."""

import unittest

import torch

from tbp.monty.frameworks.models.self_supervised_semi_markov_memory import (
    SelfSupervisedSemiMarkovMemory,
)


class TestSelfSupervisedSemiMarkovMemory(unittest.TestCase):
    def test_discovers_and_reuses_latent_states(self):
        memory = SelfSupervisedSemiMarkovMemory(
            match_threshold=0.8,
            new_state_threshold=0.55,
            min_segment_steps=1,
        )

        embeddings = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.95, 0.05]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.05, 0.95]),
            torch.tensor([1.0, 0.0]),
        ]
        labels = []

        for embedding in embeddings:
            info = memory.observe(embedding, learn=True, surprise=0.1)
            labels.append(info["current_label"])

        self.assertEqual(len(memory.get_known_states()), 2)
        self.assertEqual(labels[0], labels[-1])
        self.assertNotEqual(labels[0], labels[2])

    def test_learns_duration_and_predicts_next_state(self):
        memory = SelfSupervisedSemiMarkovMemory(
            match_threshold=0.8,
            new_state_threshold=0.55,
            min_segment_steps=1,
            default_duration=2.0,
        )

        train_embeddings = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.95, 0.05]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.05, 0.95]),
        ]
        for embedding in train_embeddings:
            memory.observe(embedding, learn=True, surprise=0.1)
        memory.finalize_episode(learn=True)
        memory.reset_episode()

        info = memory.observe(torch.tensor([1.0, 0.0]), learn=False, surprise=0.1)
        first_label = info["current_label"]
        self.assertEqual(info["predicted_label"], first_label)

        info = memory.observe(torch.tensor([0.95, 0.05]), learn=False, surprise=0.1)
        predicted_next = info["predicted_label"]
        self.assertNotEqual(predicted_next, first_label)

        info = memory.observe(torch.tensor([0.0, 1.0]), learn=False, surprise=0.1)

        self.assertEqual(info["prediction_status"], "confident")
        self.assertTrue(info["event_detected"])
        self.assertEqual(info["current_label"], predicted_next)

    def test_state_dict_round_trip(self):
        memory = SelfSupervisedSemiMarkovMemory(
            match_threshold=0.8,
            new_state_threshold=0.55,
            min_segment_steps=1,
        )

        for embedding in (
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.95, 0.05]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.05, 0.95]),
        ):
            memory.observe(embedding, learn=True, surprise=0.1)
        memory.finalize_episode(learn=True)

        restored = SelfSupervisedSemiMarkovMemory()
        restored.load_state_dict(memory.state_dict())

        self.assertEqual(restored.get_known_states(), memory.get_known_states())
        first_label = memory.get_known_states()[0]
        self.assertEqual(
            restored.get_transition_distribution(first_label),
            memory.get_transition_distribution(first_label),
        )


if __name__ == "__main__":
    unittest.main()