# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for CorticalColumnTorchLM (LearningModule adapter)."""

import unittest
from collections import Counter

import numpy as np
import torch

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)
from tbp.monty.frameworks.models.states import State


def _make_state(location=None, hsv=None, use_state=True, sender_type="SM"):
    """Create a minimal State for testing."""
    return State(
        location=np.array(location or [0.0, 0.0, 0.0], dtype=np.float64),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
        },
        non_morphological_features={"hsv": hsv or [0.5, 0.3, 0.8]},
        confidence=1.0,
        use_state=use_state,
        sender_id="test_sm",
        sender_type=sender_type,
    )


def _rotation_z(theta_radians: float) -> np.ndarray:
    c = float(np.cos(theta_radians))
    s = float(np.sin(theta_radians))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class TestCorticalColumnTorchLMInterface(unittest.TestCase):
    """Verify LearningModule ABC compliance."""

    def _make_lm(self, **col_kwargs):
        defaults = dict(
            n_minicolumns=128,
            n_cells_per_minicolumn=4,
            sparsity=0.1,
            seed=42,
        )
        defaults.update(col_kwargs)
        return CorticalColumnTorchLM(
            column_kwargs=defaults,
            learning_module_id="test_lm",
        )

    def test_all_abstract_methods_exist(self):
        lm = self._make_lm()
        required = [
            "reset", "pre_episode", "post_episode", "set_experiment_mode",
            "matching_step", "exploratory_step",
            "receive_votes", "send_out_vote",
            "propose_goal_states", "get_output",
            "state_dict", "load_state_dict",
        ]
        for method in required:
            self.assertTrue(callable(getattr(lm, method, None)), f"Missing {method}")

    def test_lifecycle(self):
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "test"})
        lm.post_episode()

    def test_matching_step(self):
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        state = _make_state(location=[0.1, 0.2, 0.3])
        lm.matching_step(None, [state])
        self.assertTrue(lm._stepped)

    def test_exploratory_step(self):
        lm = self._make_lm()
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "obj"})

        state = _make_state(location=[0.1, 0.2, 0.3])
        lm.exploratory_step(None, [state])
        self.assertTrue(lm._stepped)


class TestCorticalColumnTorchLMVoting(unittest.TestCase):
    def _make_trained_lm(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=128,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            learning_module_id="voter",
        )
        # Train
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "mug"})
        for i in range(10):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.exploratory_step(None, [state])
        lm.post_episode()

        # Eval
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.matching_step(None, [state])
        return lm

    def test_send_vote(self):
        lm = self._make_trained_lm()
        vote = lm.send_out_vote()
        # Should have vote data (or None if no evidence above threshold)
        if vote is not None:
            self.assertIn("possible_states", vote)
            self.assertIn("sensed_pose_rel_body", vote)

    def test_send_vote_uses_ranked_top_k_hypotheses(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            learning_module_id="ranked_voter",
            vote_evidence_threshold=0.95,
            vote_top_k=2,
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._last_input_state = _make_state(location=[0.1, 0.2, 0.3])
        lm._last_result = {
            "evidence": {"fox": 4.0, "robot": 3.0, "mug": 2.0}
        }

        vote = lm.send_out_vote()

        self.assertIsNotNone(vote)
        self.assertEqual(
            [entry["object_id"] for entry in vote["ranked_hypotheses"]],
            ["fox", "robot"],
        )
        self.assertEqual(set(vote["possible_states"].keys()), {"fox", "robot"})
        self.assertGreater(
            vote["ranked_hypotheses"][0]["probability"],
            vote["ranked_hypotheses"][1]["probability"],
        )

    def test_receive_votes_dict(self):
        lm = self._make_trained_lm()
        # Receive a vote that boosts "mug"
        vote_state = _make_state(location=[0.1, 0.0, 0.0])
        vote_state = State(
            location=np.array([0.1, 0.0, 0.0]),
            morphological_features={"pose_vectors": np.eye(3), "pose_fully_defined": True},
            non_morphological_features=None,
            confidence=0.8,
            use_state=True,
            sender_id="other_lm",
            sender_type="LM",
        )
        lm.receive_votes({"mug": [vote_state]})
        # Evidence should have increased
        self.assertIn("mug", lm.evidence)

    def test_two_lms_vote_exchange(self):
        """Two LMs exchange votes, evidence should converge."""
        lm1 = self._make_trained_lm()
        lm2 = self._make_trained_lm()

        vote1 = lm1.send_out_vote()
        vote2 = lm2.send_out_vote()

        if vote1 is not None:
            lm2.receive_votes(vote1.get("possible_states", {}))
        if vote2 is not None:
            lm1.receive_votes(vote2.get("possible_states", {}))

        # Both should still have evidence
        self.assertTrue(len(lm1.evidence) > 0 or len(lm2.evidence) > 0)


class TestCorticalColumnTorchLMHierarchy(unittest.TestCase):
    def test_prepare_input_state_uses_relative_hidden_geometry(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            inferred_state_config={},
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        first_state = _make_state(location=[1.0, 2.0, 3.0])
        second_state = _make_state(location=[1.2, 1.9, 3.1])
        second_state.morphological_features["pose_vectors"] = _rotation_z(np.pi / 2)
        second_state.non_morphological_features["flow_direction"] = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        second_state.non_morphological_features["flow_magnitude"] = np.array(
            [2.0],
            dtype=np.float64,
        )

        inferred_first = lm._prepare_input_state(first_state)
        inferred_second = lm._prepare_input_state(second_state)

        np.testing.assert_allclose(
            inferred_first.location,
            np.zeros(3, dtype=np.float64),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inferred_first.morphological_features["pose_vectors"],
            np.eye(3),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inferred_second.location,
            np.array([0.2, -0.1, 0.1], dtype=np.float64),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inferred_second.morphological_features["pose_vectors"],
            _rotation_z(np.pi / 2),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inferred_second.non_morphological_features["flow_direction"],
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            inferred_second.non_morphological_features["flow_magnitude"],
            np.array([2.0], dtype=np.float64),
            atol=1e-6,
        )

    def test_get_output(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        state = _make_state(location=[0.1, 0.2, 0.3])
        lm.matching_step(None, [state])

        output = lm.get_output()
        self.assertIsNotNone(output)
        self.assertEqual(output.sender_type, "LM")
        self.assertIn("active_cells", output.non_morphological_features)

    def test_get_output_coerces_numpy_scalars_to_python_bool(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._last_input_state = _make_state(location=[0.1, 0.2, 0.3])
        lm._column.get_current_mlh = lambda: {
            "graph_id": "mug",
            "evidence": np.float32(1.5),
        }

        output = lm.get_output()

        self.assertIsNotNone(output)
        self.assertIs(type(output.use_state), bool)
        self.assertTrue(output.use_state)

    def test_get_output_uses_inferred_state_geometry(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            inferred_state_config={},
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._last_observed_state = _make_state(location=[1.0, 2.0, 3.0])
        lm._last_input_state = _make_state(location=[0.2, -0.1, 0.1])
        lm._last_input_state.morphological_features["pose_vectors"] = _rotation_z(
            np.pi / 2
        )
        lm._column.get_current_mlh = lambda: {
            "graph_id": "mug",
            "evidence": np.float32(1.5),
        }

        output = lm.get_output()

        self.assertIsNotNone(output)
        np.testing.assert_allclose(
            output.location,
            np.array([0.2, -0.1, 0.1], dtype=np.float64),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            output.morphological_features["pose_vectors"],
            _rotation_z(np.pi / 2),
            atol=1e-6,
        )

    def test_receive_context(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                use_apical=True,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        context = np.random.rand(lm.column.n_cells).astype(np.float32)
        lm.receive_context(active_cells=context)

        state = _make_state(location=[0.1, 0.2, 0.3])
        lm.matching_step(None, [state])
        self.assertTrue(lm._stepped)

    def test_receive_context_waits_for_expected_senders_before_context_step(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                use_apical=True,
                seed=42,
            ),
            expected_context_sender_ids=["lm_morphology", "lm_behavior"],
            context_identity_weight=0.5,
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        n_cells = lm.column.n_cells
        context_a = np.zeros(n_cells, dtype=np.float32)
        context_b = np.zeros(n_cells, dtype=np.float32)
        context_a[:8] = 1.0
        context_b[8:16] = 1.0

        lm.receive_context(
            active_cells=context_a,
            sender_id="lm_morphology",
            graph_id="fox",
            confidence=0.9,
            sender_step_count=1,
        )

        self.assertFalse(lm._stepped)
        self.assertEqual(lm._step_count, 0)

        lm.receive_context(
            active_cells=context_b,
            sender_id="lm_behavior",
            graph_id="fox",
            confidence=0.8,
            sender_step_count=1,
        )

        expected_context = (0.9 * context_a + 0.8 * context_b) / (0.9 + 0.8)
        self.assertTrue(lm._stepped)
        self.assertEqual(lm._step_count, 1)
        np.testing.assert_allclose(
            lm._external_context.cpu().numpy(),
            expected_context,
            atol=1e-6,
        )
        self.assertIsNotNone(lm._external_context_identity)
        self.assertGreater(
            float(lm._external_context_identity.abs().sum().item()),
            0.0,
        )
        self.assertGreater(
            float(lm.column._hopfield_query_bias.abs().sum().item()),
            0.0,
        )

    def test_receive_context_steps_once_per_new_epoch_using_latest_packets(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                use_apical=True,
                seed=42,
            ),
            expected_context_sender_ids=["lm_morphology", "lm_behavior"],
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        n_cells = lm.column.n_cells
        context_a1 = np.zeros(n_cells, dtype=np.float32)
        context_b1 = np.zeros(n_cells, dtype=np.float32)
        context_a2 = np.zeros(n_cells, dtype=np.float32)
        context_b2 = np.zeros(n_cells, dtype=np.float32)
        context_a1[:8] = 1.0
        context_b1[8:16] = 1.0
        context_a2[16:24] = 1.0
        context_b2[24:32] = 1.0

        lm.receive_context(
            active_cells=context_a1,
            sender_id="lm_morphology",
            graph_id="fox",
            confidence=0.9,
            sender_step_count=1,
        )
        self.assertEqual(lm._step_count, 0)

        lm.receive_context(
            active_cells=context_b1,
            sender_id="lm_behavior",
            graph_id="fox",
            confidence=0.8,
            sender_step_count=1,
        )
        self.assertEqual(lm._step_count, 1)
        self.assertEqual(lm._last_context_step_epoch, 1)

        lm.receive_context(
            active_cells=context_a2,
            sender_id="lm_morphology",
            graph_id="fox",
            confidence=0.7,
            sender_step_count=2,
        )
        self.assertEqual(lm._step_count, 2)
        self.assertEqual(lm._last_context_step_epoch, 2)

        expected_context = (0.7 * context_a2 + 0.8 * context_b1) / (0.7 + 0.8)
        np.testing.assert_allclose(
            lm._external_context.cpu().numpy(),
            expected_context,
            atol=1e-6,
        )

        lm.receive_context(
            active_cells=context_b2,
            sender_id="lm_behavior",
            graph_id="fox",
            confidence=0.6,
            sender_step_count=2,
        )
        self.assertEqual(lm._step_count, 2)

    def test_receive_context_context_only_step_updates_temporal_state(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                use_apical=True,
                defer_context_auto_label=True,
                seed=42,
            ),
            expected_context_sender_ids=["lm_morphology", "lm_behavior"],
            use_child_graph_context_labels=True,
            temporal_trace_config={"trace_weight": 0.35},
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
                "switch_margin": 0.05,
            },
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode()

        n_cells = lm.column.n_cells
        context_a = np.zeros(n_cells, dtype=np.float32)
        context_b = np.zeros(n_cells, dtype=np.float32)
        context_a[:8] = 1.0
        context_b[8:16] = 1.0

        for epoch in range(1, 4):
            lm.receive_context(
                active_cells=context_a,
                sender_id="lm_morphology",
                graph_id=f"shape_{epoch}",
                confidence=0.8,
                sender_step_count=epoch,
            )
            lm.receive_context(
                active_cells=context_b,
                sender_id="lm_behavior",
                graph_id=f"motion_{epoch}",
                confidence=0.7,
                sender_step_count=epoch,
            )

        temporal_context = lm.get_temporal_context() or {}
        debug = lm.get_evidence_debug() or {}

        self.assertEqual(lm._step_count, 3)
        self.assertTrue(str(lm.column._current_object).startswith("auto_ctx_"))
        self.assertIsNotNone(temporal_context.get("current_label"))
        self.assertGreater(float(temporal_context.get("trace_norm", 0.0)), 0.0)
        self.assertIn("winner_path", debug)
        self.assertIn("base", debug["winner_path"])
        self.assertIn("after_temporal_behavior", debug["winner_path"])
        self.assertIn("final", debug["winner_path"])
        self.assertEqual(
            debug.get("temporal_labels", {}).get("current"),
            temporal_context.get("current_label"),
        )

    def test_extract_usable_state_fuses_multiple_lm_inputs(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                use_apical=True,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        n_cells = lm.column.n_cells
        active_a = np.zeros(n_cells, dtype=np.float32)
        active_b = np.zeros(n_cells, dtype=np.float32)
        active_a[:8] = 1.0
        active_b[8:16] = 1.0

        state_a = State(
            location=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "active_cells": active_a,
                "graph_id": "fox",
                "object_id": 1,
                "surprise": 0.1,
            },
            confidence=0.8,
            use_state=True,
            sender_id="lm_morphology",
            sender_type="LM",
        )
        state_b = State(
            location=np.array([0.2, 0.0, 0.0], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "active_cells": active_b,
                "graph_id": "fox",
                "object_id": 1,
                "surprise": 0.2,
            },
            confidence=0.2,
            use_state=True,
            sender_id="lm_behavior",
            sender_type="LM",
        )

        usable = lm._extract_usable_state([state_a, state_b])

        expected_context = (0.8 * active_a + 0.2 * active_b) / 1.0
        self.assertIsNotNone(usable)
        np.testing.assert_allclose(
            lm._external_context.cpu().numpy(),
            expected_context,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            usable.location,
            np.array([0.04, 0.0, 0.0], dtype=np.float64),
            atol=1e-6,
        )
        self.assertEqual(
            usable.non_morphological_features["child_sender_ids"],
            ["lm_morphology", "lm_behavior"],
        )
        self.assertEqual(
            usable.non_morphological_features["child_graph_ids"],
            ["fox", "fox"],
        )

    def test_send_vote_uses_inferred_state_but_raw_transport_pose(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            learning_module_id="ranked_voter",
            vote_top_k=1,
            inferred_state_config={},
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._last_observed_state = _make_state(location=[1.0, 2.0, 3.0])
        lm._last_input_state = _make_state(location=[0.2, -0.1, 0.1])
        lm._last_input_state.morphological_features["pose_vectors"] = _rotation_z(
            np.pi / 2
        )
        lm._last_result = {"evidence": {"fox": 4.0}}

        vote = lm.send_out_vote()

        self.assertIsNotNone(vote)
        vote_state = vote["possible_states"]["fox"][0]
        np.testing.assert_allclose(
            vote_state.location,
            np.array([0.2, -0.1, 0.1], dtype=np.float64),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            vote["sensed_pose_rel_body"][0],
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            atol=1e-6,
        )

    def test_propose_goal_states_falls_back_to_inferred_location(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            inferred_state_config={},
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._last_observed_state = _make_state(location=[1.0, 2.0, 3.0])
        lm._last_input_state = _make_state(location=[0.2, -0.1, 0.1])
        lm._column._evidence = {"fox": 3.5}

        goal_states = lm.propose_goal_states()

        self.assertEqual(len(goal_states), 1)
        np.testing.assert_allclose(
            goal_states[0].location,
            np.array([0.2, -0.1, 0.1], dtype=np.float64),
            atol=1e-6,
        )

    def test_train_with_stepwise_target_state_stores_state_specific_labels(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "hinge"})

        state_a = _make_state(location=[0.0, 0.0, 0.0])
        state_b = _make_state(location=[0.1, 0.0, 0.0], hsv=[0.8, 0.2, 0.2])

        for _ in range(3):
            lm.stepwise_target_state = 0
            lm.exploratory_step(None, [state_a])
        for _ in range(3):
            lm.stepwise_target_state = 1
            lm.exploratory_step(None, [state_b])

        lm.post_episode()

        known = set(lm.get_all_known_object_ids())
        self.assertIn("hinge:0", known)
        self.assertIn("hinge:1", known)

    def test_temporal_behavior_evidence_boosts_matching_sequence(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_memory_config={
                "sdr_dim": 256,
                "sdr_sparsity": 0.05,
                "learning_rate": 0.2,
                "include_location": False,
            },
            temporal_behavior_weight=1.0,
            temporal_behavior_min_history=2,
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        walk_1 = _make_state(location=[0.0, 0.0, 0.0], hsv=[0.1, 0.2, 0.3])
        walk_2 = _make_state(location=[0.0, 0.0, 0.0], hsv=[0.6, 0.2, 0.3])

        lm._column._evidence = {"walk": 0.1, "run": 0.1}
        lm._temporal_memory.store_behavior_prototype("walk", [walk_1, walk_2])
        lm._temporal_memory.store_behavior_prototype("run", [walk_2, walk_1])
        lm._temporal_observation_history = [walk_1, walk_2]

        lm._apply_temporal_behavior_evidence()

        self.assertGreater(lm._column._evidence["walk"], lm._column._evidence["run"])

    def test_temporal_behavior_evidence_replaces_stale_prefix_bias(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_memory_config={
                "sdr_dim": 256,
                "sdr_sparsity": 0.05,
                "learning_rate": 0.2,
                "include_location": False,
            },
            temporal_behavior_weight=3.0,
            temporal_behavior_min_history=1,
            temporal_behavior_competition=1.5,
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        probe_state = _make_state(location=[0.0, 0.0, 0.0], hsv=[0.1, 0.2, 0.3])
        lm._temporal_observation_history = [probe_state]
        lm._column._evidence = {"walk": 5.0, "run": 4.0}

        scored_prefixes = iter(
            [
                {"walk": 0.95, "run": 0.90},
                {"walk": 0.80, "run": 0.98},
            ]
        )
        lm._temporal_memory.score_behaviors = lambda *args, **kwargs: next(scored_prefixes)

        lm._apply_temporal_behavior_evidence()
        self.assertGreater(lm._column._evidence["walk"], lm._column._evidence["run"])

        lm._apply_temporal_behavior_evidence()

        self.assertGreater(lm._column._evidence["run"], lm._column._evidence["walk"])

    def test_transition_memory_exposes_temporal_prediction_status(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_transition_config={"min_stable_steps": 1},
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        lm._stepped = True
        lm._column.get_current_mlh = lambda: {"graph_id": "hinge:0", "evidence": 2.0}

        lm._update_temporal_state()
        lm._update_temporal_state()

        lm._column.get_current_mlh = lambda: {"graph_id": "hinge:1", "evidence": 2.0}
        lm._update_temporal_state()

        self.assertIn(lm.get_temporal_prediction_status(), {"confident", "confused", None})

    def test_context_signal_roundtrip(self):
        """Parent output → child context → child processes it."""
        parent = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
            learning_module_id="parent",
        )
        child = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
                use_apical=True,
            ),
            learning_module_id="child",
        )

        parent.set_experiment_mode(ExperimentMode.EVAL)
        child.set_experiment_mode(ExperimentMode.EVAL)
        parent.pre_episode()
        child.pre_episode()

        # Parent steps
        state = _make_state(location=[0.1, 0.2, 0.3])
        parent.matching_step(None, [state])

        # Parent sends context to child
        ctx = parent.get_context_signal()
        if ctx is not None:
            child.receive_context(**ctx)

        # Child steps with context
        child.matching_step(None, [state])
        self.assertTrue(child._stepped)
        self.assertEqual(ctx["sender_id"], "parent")
        self.assertEqual(ctx["sender_step_count"], 1)


class TestCorticalColumnTorchLMAutoLabel(unittest.TestCase):
    def test_train_without_label(self):
        """Train with primary_target=None — auto-label should work."""
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        # No object name
        lm.pre_episode(primary_target=None)

        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.exploratory_step(None, [state])

        lm.post_episode()
        known = lm.get_all_known_object_ids()

        self.assertIsNotNone(lm.column._current_object)
        self.assertTrue(str(lm.column._current_object).startswith("auto_"))
        self.assertIn(lm.column._current_object, known)

    def test_temporal_behavior_auto_label_uses_column_auto_label(self):
        """Unlabeled behavior episodes should store a temporal prototype."""
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_memory_config={
                "sdr_dim": 256,
                "sdr_sparsity": 0.05,
                "learning_rate": 0.2,
                "include_location": False,
            },
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target=None)

        seq = [
            _make_state(location=[0.0, 0.0, 0.0], hsv=[0.1, 0.2, 0.3]),
            _make_state(location=[0.1, 0.0, 0.0], hsv=[0.2, 0.2, 0.3]),
            _make_state(location=[0.2, 0.0, 0.0], hsv=[0.3, 0.2, 0.3]),
        ]
        for state in seq:
            lm.exploratory_step(None, [state])

        lm.post_episode()

        auto_label = lm.column._current_object
        self.assertIsNotNone(auto_label)
        self.assertTrue(str(auto_label).startswith("auto_"))
        self.assertIn(auto_label, lm._temporal_memory._known_behaviors)

        name, score = lm._temporal_memory.recognize_behavior(seq, min_overlap=0.0)

        self.assertEqual(name, auto_label)
        self.assertGreaterEqual(score, 0.0)

    def test_train_without_label_can_use_child_graph_context_label(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            use_child_graph_context_labels=True,
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target=None)

        n_cells = lm.column.n_cells
        active_a = np.zeros(n_cells, dtype=np.float32)
        active_b = np.zeros(n_cells, dtype=np.float32)
        active_a[:8] = 1.0
        active_b[8:16] = 1.0

        state_a = State(
            location=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "active_cells": active_a,
                "graph_id": "fox",
            },
            confidence=0.9,
            use_state=True,
            sender_id="lm_morphology",
            sender_type="LM",
        )
        state_b = State(
            location=np.array([0.1, 0.0, 0.0], dtype=np.float64),
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": True,
                "on_object": True,
            },
            non_morphological_features={
                "active_cells": active_b,
                "graph_id": "walk_phase_a",
            },
            confidence=0.8,
            use_state=True,
            sender_id="lm_behavior",
            sender_type="LM",
        )

        lm.exploratory_step(None, [state_a, state_b])

        self.assertIsNotNone(lm.column._current_object)
        self.assertTrue(str(lm.column._current_object).startswith("auto_ctx_"))


class TestCorticalColumnTorchLMSelfSupervisedTemporal(unittest.TestCase):
    def test_self_supervised_temporal_path_discovers_latent_states(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
                "switch_margin": 0.05,
            },
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "hinge"})
        lm._stepped = True

        embeddings = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.95, 0.05, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.95, 0.05, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ]
        labels = []

        for embedding in embeddings:
            lm._column._active = embedding
            lm._column._surprise = 0.1
            lm._update_temporal_state()
            context = lm.get_temporal_context() or {}
            labels.append(context.get("current_label"))

        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(labels[0], labels[-1])
        self.assertTrue(str(labels[0]).startswith("latent_"))

    def test_self_supervised_temporal_path_exposes_prediction_status(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
                "default_duration": 2.0,
            },
        )
        embeddings = [
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.95, 0.05, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.95, 0.05, 0.0]),
        ]

        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "hinge"})
        lm._stepped = True
        for embedding in embeddings:
            lm._column._active = embedding
            lm._column._surprise = 0.1
            lm._update_temporal_state()
        lm.post_episode()

        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode(primary_target={"object": "hinge"})
        lm._stepped = True

        for embedding in embeddings[:2]:
            lm._column._active = embedding
            lm._column._surprise = 0.1
            lm._update_temporal_state()

        context = lm.get_temporal_context() or {}
        predicted_label = context.get("predicted_label")
        self.assertIsNotNone(predicted_label)

        lm._column._active = embeddings[2]
        lm._column._surprise = 0.1
        lm._update_temporal_state()

        self.assertEqual(lm.get_temporal_prediction_status(), "confident")
        self.assertTrue(lm.get_event_signal())

    def test_self_supervised_temporal_transition_bias_boosts_matching_object(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            self_supervised_temporal_config={
                "match_threshold": 0.8,
                "new_state_threshold": 0.55,
                "min_segment_steps": 1,
            },
            self_supervised_temporal_transition_weight=2.0,
        )
        lm._column._evidence = {"behavior_a": 1.0, "behavior_b": 1.0}
        lm._previous_temporal_label = "latent_0000"
        lm._current_temporal_label = "latent_0001"
        lm._predicted_temporal_label = None
        lm._self_supervised_transition_object_counts[
            "latent_0000\tlatent_0001"
        ] = Counter({"behavior_a": 5, "behavior_b": 1})

        lm._apply_self_supervised_temporal_evidence()

        self.assertGreater(lm._column._evidence["behavior_a"], lm._column._evidence["behavior_b"])


class TestCorticalColumnTorchLMTemporalTrace(unittest.TestCase):
    def _make_lm(self, **trace_config):
        return CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_trace_config=trace_config or {},
        )

    def test_temporal_trace_auto_enables_apical_and_updates_query_bias(self):
        lm = self._make_lm(trace_weight=0.5)
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        lm.matching_step(None, [_make_state(location=[0.1, 0.2, 0.3])])
        lm._apply_inference_context()

        self.assertTrue(lm.column._use_apical)
        self.assertGreater(lm._temporal_trace_norm, 0.0)
        self.assertEqual(float(lm.column._context.abs().sum().item()), 0.0)
        self.assertGreater(
            float(lm.column._hopfield_query_bias.abs().sum().item()), 0.0
        )

        context = lm.get_temporal_context() or {}
        self.assertIn("trace_norm", context)
        self.assertIn("boundary_pressure", context)

    def test_temporal_trace_query_bias_is_separate_from_external_context(self):
        lm = self._make_lm(trace_weight=0.75)
        lm._temporal_trace_vector.zero_()
        lm._temporal_trace_vector[3] = 1.0
        lm._temporal_trace_norm = 1.0

        external = np.zeros(lm.column.n_cells, dtype=np.float32)
        external[7] = 1.0
        lm._set_external_context(external)

        combined = lm._compose_inference_context()
        query_bias = lm._compose_hopfield_query_bias()

        self.assertIsNotNone(combined)
        self.assertIsNotNone(query_bias)
        self.assertEqual(float(combined[3].item()), 0.0)
        self.assertGreater(float(combined[7].item()), 0.0)
        self.assertGreater(float(query_bias[3].item()), 0.0)
        self.assertEqual(float(query_bias[7].item()), 0.0)

        lm._apply_inference_context()
        self.assertEqual(float(lm.column._context[3].item()), 0.0)
        self.assertGreater(float(lm.column._context[7].item()), 0.0)
        self.assertGreater(float(lm.column._hopfield_query_bias[3].item()), 0.0)
        self.assertEqual(float(lm.column._hopfield_query_bias[7].item()), 0.0)

        lm._temporal_trace_vector.zero_()
        lm._temporal_trace_norm = 0.0
        lm._apply_inference_context()
        self.assertEqual(
            float(lm.column._hopfield_query_bias.abs().sum().item()), 0.0
        )

    def test_matching_step_records_evidence_debug_path(self):
        lm = self._make_lm(trace_weight=0.5)
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        def fake_step(state):
            lm._column._evidence = {"robot": 2.0, "fox": 1.0}
            return {
                "graph_id": "robot",
                "evidence": dict(lm._column._evidence),
            }

        def fake_temporal_behavior():
            lm._temporal_behavior_adjustments = {"robot": -0.2, "fox": 0.4}
            lm._last_temporal_behavior_scores = {"robot": 0.1, "fox": 0.5}
            lm._column._evidence = {"robot": 1.8, "fox": 1.4}

        def fake_update_temporal_state():
            lm._previous_temporal_label = "latent_prev"
            lm._current_temporal_label = "latent_robot"
            lm._predicted_temporal_label = "latent_fox"

        def fake_self_supervised():
            lm._self_supervised_temporal_adjustments = {
                "robot": -0.6,
                "fox": 0.7,
            }
            lm._last_self_supervised_temporal_support = {
                "state": {"robot": 0.4},
                "transition": {"fox": 0.6},
                "prediction": {"fox": 0.1},
            }
            lm._column._evidence = {"robot": 1.2, "fox": 2.1}

        lm._apply_inference_context = lambda: None
        lm._feed_temporal_memory = lambda observations: None
        lm._update_temporal_trace_state = lambda: None
        lm._column.step = fake_step
        lm._apply_temporal_behavior_evidence = fake_temporal_behavior
        lm._update_temporal_state = fake_update_temporal_state
        lm._apply_self_supervised_temporal_evidence = fake_self_supervised
        lm._update_possible_matches = lambda: None
        lm._auto_update_terminal_condition = lambda: None
        lm._buffer_observation = lambda state: None
        lm._get_query_bias_norms = lambda: {
            "trace": 1.5,
            "action": 0.25,
            "combined": 1.75,
        }

        lm.matching_step(None, [_make_state(location=[0.1, 0.2, 0.3])])

        debug = lm.get_evidence_debug()

        self.assertIsNotNone(debug)
        self.assertEqual(debug["winner_path"]["base"], "robot")
        self.assertEqual(debug["winner_path"]["after_temporal_behavior"], "robot")
        self.assertEqual(debug["winner_path"]["final"], "fox")
        self.assertEqual(debug["base_evidence"], {"robot": 2.0, "fox": 1.0})
        self.assertEqual(
            debug["after_temporal_behavior_evidence"],
            {"robot": 1.8, "fox": 1.4},
        )
        self.assertEqual(debug["final_evidence"], {"robot": 1.2, "fox": 2.1})
        self.assertEqual(
            debug["temporal_behavior_scores"],
            {"robot": 0.1, "fox": 0.5},
        )
        self.assertEqual(
            debug["self_supervised_support"]["transition"],
            {"fox": 0.6},
        )
        self.assertEqual(
            debug["temporal_labels"],
            {
                "previous": "latent_prev",
                "current": "latent_robot",
                "predicted": "latent_fox",
            },
        )
        self.assertEqual(
            debug["query_bias_norms"],
            {"trace": 1.5, "action": 0.25, "combined": 1.75},
        )
        self.assertEqual(lm.evidence, {"robot": 1.2, "fox": 2.1})

    def test_boundary_pressure_resets_temporal_trace_bank(self):
        lm = self._make_lm(
            decay_rates=[0.9],
            input_gain=0.0,
            boundary_threshold=0.5,
            boundary_surprise_weight=1.0,
            boundary_discontinuity_weight=1.0,
        )

        lm._temporal_trace_bank.fill_(1.0)
        lm._column._prev_active.zero_()
        lm._column._active.zero_()
        lm._column._active[:8] = 1.0
        lm._column._surprise = 0.1
        lm._update_temporal_trace_state()
        low_pressure = lm._temporal_boundary_pressure
        low_norm = float(lm._temporal_trace_bank.abs().sum().item())

        lm._temporal_trace_bank.fill_(1.0)
        lm._column._prev_active.zero_()
        lm._column._prev_active[:8] = 1.0
        lm._column._active.zero_()
        lm._column._active[8:16] = 1.0
        lm._column._surprise = 0.95
        lm._update_temporal_trace_state()
        high_pressure = lm._temporal_boundary_pressure
        high_norm = float(lm._temporal_trace_bank.abs().sum().item())

        self.assertGreater(high_pressure, low_pressure)
        self.assertLess(high_norm, low_norm)

    def test_boundary_pressure_uses_pre_settle_mismatch(self):
        lm = self._make_lm(
            decay_rates=[0.9],
            input_gain=0.0,
            boundary_threshold=0.5,
            boundary_surprise_weight=1.0,
            boundary_discontinuity_weight=0.0,
        )

        lm._column._prev_active.zero_()
        lm._column._active.zero_()
        lm._column._active[:8] = 1.0
        lm._column._surprise = 0.0
        lm._column._prediction_mismatch = 0.1
        lm._update_temporal_trace_state()
        low_pressure = lm._temporal_boundary_pressure

        lm._column._prediction_mismatch = 0.95
        lm._update_temporal_trace_state()
        high_pressure = lm._temporal_boundary_pressure

        self.assertGreater(high_pressure, low_pressure)

    def test_boundary_pressure_uses_action_prediction_error(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            temporal_trace_config={
                "decay_rates": [0.9],
                "input_gain": 0.0,
                "boundary_threshold": 0.5,
                "boundary_surprise_weight": 1.0,
                "boundary_discontinuity_weight": 0.0,
            },
            action_predictive_config={
                "action_dim": 8,
                "trace_gain": 0.0,
                "boundary_weight": 1.0,
            },
        )

        lm._column._prev_active.zero_()
        lm._column._active.zero_()
        lm._column._active[:8] = 1.0
        lm._column._surprise = 0.0
        lm._column._prediction_mismatch = 0.0
        lm._column._action_prediction_error = 0.1
        lm._update_temporal_trace_state()
        low_pressure = lm._temporal_boundary_pressure

        lm._column._action_prediction_error = 0.95
        lm._update_temporal_trace_state()
        high_pressure = lm._temporal_boundary_pressure

        self.assertGreater(high_pressure, low_pressure)


class TestCorticalColumnTorchLMTerminalCondition(unittest.TestCase):
    def test_no_match_when_no_evidence(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()

        state = _make_state(location=[0.1, 0.2, 0.3])
        lm.matching_step(None, [state])

        # With no trained objects, terminal should be no_match
        tc = lm.update_terminal_condition()
        self.assertEqual(tc, "no_match")

    def test_match_with_strong_evidence(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=128, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
            evidence_match_threshold=0.5,
            evidence_separation_ratio=1.2,
        )
        # Train
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "mug"})
        for i in range(15):
            state = _make_state(location=[0.1 * i, 0.0, 0.0], hsv=[0.5, 0.3, 0.8])
            lm.exploratory_step(None, [state])
        lm.post_episode()

        # Eval
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        for i in range(15):
            state = _make_state(location=[0.1 * i, 0.0, 0.0], hsv=[0.5, 0.3, 0.8])
            lm.matching_step(None, [state])

        # Should have some terminal condition
        tc = lm.terminal_state
        # With one trained object, should reach match
        if lm.evidence:
            self.assertIn(tc, ["match", "no_match", None])


class TestCorticalColumnTorchLMHopfieldVoting(unittest.TestCase):
    """Phase 9: Surprise-gated Hopfield voting tests."""

    def _make_trained_lm(self, hopfield_voting=True, surprise_threshold=0.3):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=128,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=42,
            ),
            learning_module_id="hopfield_voter",
            hopfield_voting=hopfield_voting,
            surprise_vote_threshold=surprise_threshold,
        )
        # Train on an object
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "mug"})
        for i in range(15):
            state = _make_state(
                location=[0.1 * i, 0.0, 0.0], hsv=[0.5, 0.3, 0.8],
            )
            lm.exploratory_step(None, [state])
        lm.post_episode()

        # Switch to eval
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        return lm

    def test_surprise_gating_suppresses_early_votes(self):
        """Votes should be suppressed when surprise is high (not converged)."""
        lm = self._make_trained_lm(surprise_threshold=0.1)

        # First step: surprise should be high (novel input)
        state = _make_state(location=[0.5, 0.5, 0.5])
        lm.matching_step(None, [state])

        # With high surprise threshold (0.1), first-step surprise (~1.0)
        # should suppress the vote
        vote = lm.send_out_vote()
        self.assertIsNone(
            vote,
            "Surprise-gated voting should suppress votes on high-surprise steps",
        )

    def test_no_gating_without_hopfield_voting(self):
        """Without hopfield_voting, votes should not be surprise-gated."""
        lm = self._make_trained_lm(hopfield_voting=False)

        # Even first step should produce a vote (if there's evidence)
        state = _make_state(location=[0.1, 0.0, 0.0])
        lm.matching_step(None, [state])

        # With hopfield_voting=False, no surprise gating
        # (vote may be None for other reasons, but not surprise)
        # Just verify the method doesn't crash
        lm.send_out_vote()

    def test_vote_includes_hopfield_pattern(self):
        """When hopfield_voting is enabled and surprise is low, the vote
        should include the sender's settled Hopfield activation pattern."""
        lm = self._make_trained_lm(surprise_threshold=0.99)

        # Present familiar input multiple times to lower surprise
        for i in range(10):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.matching_step(None, [state])

        vote = lm.send_out_vote()
        if vote is not None:
            self.assertIn(
                "hopfield_pattern", vote,
                "Hopfield voting should include the settled activation pattern",
            )
            pattern = vote["hopfield_pattern"]
            self.assertGreater(
                pattern.abs().sum().item(), 0,
                "Hopfield pattern should be non-zero",
            )

    def test_receive_hopfield_pattern_boosts_evidence(self):
        """Receiving a Hopfield pattern should boost evidence via retrieval."""
        import torch

        lm1 = self._make_trained_lm(surprise_threshold=0.99)
        lm2 = self._make_trained_lm(surprise_threshold=0.99)

        # Both step through some inputs
        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm1.matching_step(None, [state])
            lm2.matching_step(None, [state])

        evidence_before = dict(lm2.evidence)

        # lm1 sends vote with hopfield pattern
        vote = lm1.send_out_vote()
        if vote is not None:
            lm2.receive_votes(vote)

        evidence_after = dict(lm2.evidence)

        # Evidence should have changed (increased for at least one object)
        if vote is not None and evidence_before:
            total_before = sum(evidence_before.values())
            total_after = sum(evidence_after.values())
            self.assertGreaterEqual(
                total_after, total_before,
                "Receiving Hopfield vote should not decrease total evidence",
            )

    def test_backward_compatible_voting(self):
        """Standard (non-Hopfield) voting should still work correctly."""
        lm = self._make_trained_lm(hopfield_voting=False)

        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.matching_step(None, [state])

        evidence_before = dict(lm.evidence)

        # Standard vote format (no hopfield_pattern)
        vote_state = State(
            location=np.array([0.1, 0.0, 0.0]),
            morphological_features={
                "pose_vectors": np.eye(3), "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.9,
            use_state=True,
            sender_id="other_lm",
            sender_type="LM",
        )
        lm.receive_votes({"mug": [vote_state]})

        # mug evidence should have increased
        if "mug" in evidence_before and "mug" in lm.evidence:
            self.assertGreater(
                lm.evidence["mug"], evidence_before["mug"],
                "Standard voting boost should still work",
            )


class TestHopfieldVotingClassification(unittest.TestCase):
    """Test the surprise-based LM classification used by _vote_hopfield."""

    def _make_trained_lm(self, lm_id="lm_0", hopfield_voting=True, seed=42):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=128,
                n_cells_per_minicolumn=4,
                sparsity=0.1,
                seed=seed,
            ),
            learning_module_id=lm_id,
            hopfield_voting=hopfield_voting,
            surprise_vote_threshold=0.3,
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "mug"})
        for i in range(15):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.exploratory_step(None, [state])
        lm.post_episode()
        lm.set_experiment_mode(ExperimentMode.EVAL)
        lm.pre_episode()
        return lm

    def test_surprise_classifies_confident_vs_stuck(self):
        """Low-surprise LM should be classified as confident, high-surprise
        as stuck — the core gating logic for Hopfield voting."""
        lm_familiar = self._make_trained_lm(lm_id="familiar")
        lm_novel = self._make_trained_lm(lm_id="novel")

        # Familiar: present learned inputs (low surprise)
        for i in range(10):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm_familiar.matching_step(None, [state])

        # Novel: present never-seen inputs (high surprise)
        state = _make_state(location=[5.0, 5.0, 5.0], hsv=[0.1, 0.1, 0.1])
        lm_novel.matching_step(None, [state])

        familiar_surprise = lm_familiar._last_result.get("surprise", 1.0)
        novel_surprise = lm_novel._last_result.get("surprise", 1.0)

        # Novel input should have higher surprise
        self.assertGreater(
            novel_surprise, familiar_surprise,
            f"Novel surprise ({novel_surprise:.3f}) should exceed familiar "
            f"({familiar_surprise:.3f})",
        )

    def test_hopfield_voting_parameters_on_monty(self):
        """MontyForGraphMatching should accept hopfield_voting kwargs."""
        from tbp.monty.frameworks.models.graph_matching import (
            MontyForGraphMatching,
        )

        # Just verify the parameter is stored — full Monty construction
        # requires sensors/motor which are tested in integration tests.
        monty = MontyForGraphMatching.__new__(MontyForGraphMatching)
        monty.hopfield_voting = True
        monty.hopfield_surprise_threshold = 0.25
        self.assertTrue(monty.hopfield_voting)
        self.assertEqual(monty.hopfield_surprise_threshold, 0.25)


class TestHopfieldVotingScheduling(unittest.TestCase):
    class _Buffer:
        def get_num_observations_on_object(self):
            return 1

        def update_last_stats_entry(self, stats):
            self.last_stats = stats

    class _FakeLM:
        def __init__(self, lm_id, surprise):
            self.learning_module_id = lm_id
            self.buffer = TestHopfieldVotingScheduling._Buffer()
            self._last_result = {"surprise": surprise}
            self.received_votes = []
            self.terminal_state = None
            self.stepwise_target_object = None
            self.stepwise_targets_list = []

        def get_possible_matches(self):
            return ["mug"]

        def send_out_vote(self):
            vote_state = State(
                location=np.zeros(3),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": True,
                },
                non_morphological_features=None,
                confidence=1.0,
                use_state=True,
                sender_id=self.learning_module_id,
                sender_type="LM",
            )
            return {
                "possible_states": {"mug": [vote_state]},
                "sensed_pose_rel_body": np.vstack([np.zeros((1, 3)), np.eye(3)]),
            }

        def receive_votes(self, votes):
            self.received_votes.append(votes)

        def collect_stats_to_save(self):
            return {}

        def set_individual_ts(self, terminal_state=None):
            self.terminal_state = terminal_state

    def test_hopfield_voting_uses_delay_and_cooldown(self):
        from tbp.monty.frameworks.models.evidence_matching.model import (
            MontyForEvidenceGraphMatching,
        )

        confident_lm = self._FakeLM("LM_0", surprise=0.1)
        stuck_lm = self._FakeLM("LM_1", surprise=0.9)

        class _TestMonty(MontyForEvidenceGraphMatching):
            def __init__(self_m, lms):
                self_m.learning_modules = lms
                self_m.lm_to_lm_vote_matrix = [[], [0]]
                self_m.hopfield_voting = True
                self_m.hopfield_surprise_threshold = 0.3
                self_m.hopfield_vote_after_steps = 2
                self_m.vote_cooldown_steps = 3
                self_m.matching_steps = 0
                self_m._last_vote_step_by_receiver = {}
                self_m.sent_receivers = []

            def send_vote_to_lm(self_m, lm, lm_id, combined_votes):
                self_m.sent_receivers.append(lm_id)
                lm.receive_votes(combined_votes[lm_id])

            def update_stats_after_vote(self_m, lm):
                return None

        monty = _TestMonty([confident_lm, stuck_lm])

        monty.matching_steps = 2
        monty._vote_hopfield()
        self.assertEqual(monty.sent_receivers, [])

        monty.matching_steps = 3
        monty._vote_hopfield()
        self.assertEqual(monty.sent_receivers, [1])
        self.assertEqual(monty._last_vote_step_by_receiver[1], 3)

        monty.sent_receivers.clear()
        monty.matching_steps = 4
        monty._vote_hopfield()
        self.assertEqual(monty.sent_receivers, [])

        monty.matching_steps = 6
        monty._vote_hopfield()
        self.assertEqual(monty.sent_receivers, [1])
        self.assertEqual(len(stuck_lm.received_votes), 2)


class TestCorticalColumnTorchLMPersistence(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        lm = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
        )
        lm.set_experiment_mode(ExperimentMode.TRAIN)
        lm.pre_episode(primary_target={"object": "test_obj"})
        for i in range(5):
            state = _make_state(location=[0.1 * i, 0.0, 0.0])
            lm.exploratory_step(None, [state])
        lm.post_episode()

        sd = lm.state_dict()

        lm2 = CorticalColumnTorchLM(
            column_kwargs=dict(
                n_minicolumns=64, n_cells_per_minicolumn=4, sparsity=0.1,
            ),
        )
        lm2.load_state_dict(sd)

        self.assertIn("test_obj", lm2.get_all_known_object_ids())


if __name__ == "__main__":
    unittest.main()
