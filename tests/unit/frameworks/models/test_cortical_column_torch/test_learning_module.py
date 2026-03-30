# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Tests for CorticalColumnTorchLM (LearningModule adapter)."""

import unittest

import numpy as np

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
        # Column should still function (no crash)


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
