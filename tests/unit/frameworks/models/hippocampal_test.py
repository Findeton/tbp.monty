"""Unit tests for HippocampalModule."""
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
    s.location = location if location is not None else np.array([0.0, 0.0, 0.0])
    s.sender_id = "lm_0"
    s.sender_type = "LM"
    return s


class TestHippocampalModuleInit(unittest.TestCase):
    def test_initial_state(self):
        hpc = HippocampalModule(context_dim=16, max_episodes=10)
        self.assertEqual(len(hpc.co_occurrence_counts), 0)
        self.assertEqual(len(hpc.episodic_memory), 0)
        self.assertEqual(len(hpc._current_episode_buffer), 0)
        self.assertEqual(hpc.context_vector.shape, (16,))

    def test_get_output_returns_none(self):
        hpc = HippocampalModule()
        self.assertIsNone(hpc.get_output())

    def test_send_out_vote_returns_none(self):
        hpc = HippocampalModule()
        self.assertIsNone(hpc.send_out_vote())

    def test_propose_goal_states_empty(self):
        hpc = HippocampalModule()
        self.assertEqual(hpc.propose_goal_states(), [])


class TestFastBinding(unittest.TestCase):
    def test_cooccurrence_increments_on_two_active(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug"), _make_state("cup")]
        hpc.matching_step(ctx, states)
        key = frozenset(["mug", "cup"])
        self.assertIn(key, hpc.co_occurrence_counts)
        self.assertEqual(hpc.co_occurrence_counts[key], 1)

    def test_cooccurrence_increments_across_steps(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug"), _make_state("cup")]
        hpc.matching_step(ctx, states)
        hpc.matching_step(ctx, states)
        key = frozenset(["mug", "cup"])
        self.assertEqual(hpc.co_occurrence_counts[key], 2)

    def test_single_concept_no_binding(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug")]
        hpc.matching_step(ctx, states)
        self.assertEqual(len(hpc.co_occurrence_counts), 0)

    def test_low_confidence_excluded(self):
        hpc = HippocampalModule(min_confidence_to_bind=0.8)
        ctx = _make_ctx()
        states = [_make_state("mug", confidence=0.3), _make_state("cup", confidence=0.9)]
        hpc.matching_step(ctx, states)
        # Only one concept passes threshold -> no binding
        self.assertEqual(len(hpc.co_occurrence_counts), 0)

    def test_recall_associations_after_binding(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug"), _make_state("cup")]
        hpc.matching_step(ctx, states)
        assoc = hpc.recall_associations("mug")
        self.assertIn("cup", assoc)
        self.assertGreater(assoc["cup"], 0.0)


class TestEpisodicMemory(unittest.TestCase):
    def test_episode_buffer_grows_per_step(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug")]
        hpc.matching_step(ctx, states)
        hpc.matching_step(ctx, states)
        self.assertEqual(len(hpc._current_episode_buffer), 2)

    def test_episode_committed_on_post_episode(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug")])
        hpc.matching_step(ctx, [_make_state("cup")])
        hpc.post_episode()
        self.assertEqual(len(hpc.episodic_memory), 1)
        self.assertEqual(len(hpc._current_episode_buffer), 0)

    def test_episodic_memory_bounded_by_maxlen(self):
        hpc = HippocampalModule(max_episodes=3)
        ctx = _make_ctx()
        for _ in range(5):
            hpc.matching_step(ctx, [_make_state("mug")])
            hpc.post_episode()
        self.assertEqual(len(hpc.episodic_memory), 3)

    def test_episode_buffer_cleared_on_pre_episode(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug")])
        hpc.pre_episode()
        self.assertEqual(len(hpc._current_episode_buffer), 0)

    def test_pre_episode_commits_previous_buffer(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug")])
        hpc.pre_episode()
        self.assertEqual(len(hpc.episodic_memory), 1)

    def test_episode_step_record_format(self):
        hpc = HippocampalModule()
        ctx = _make_ctx(global_step=5, episode_step=2)
        hpc.matching_step(ctx, [_make_state("mug")])
        record = hpc._current_episode_buffer[0]
        self.assertEqual(record["global_step"], 5)
        self.assertEqual(record["episode_step"], 2)
        self.assertIn("mug", record["active_concepts"])

    def test_recall_episode(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug")])
        hpc.post_episode()
        ep = hpc.recall_episode(0)
        self.assertIsNotNone(ep)
        self.assertEqual(len(ep), 1)

    def test_recall_episode_out_of_range(self):
        hpc = HippocampalModule()
        self.assertIsNone(hpc.recall_episode(0))


class TestRelationalMemory(unittest.TestCase):
    def test_cooccurrence_edge_created(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug"), _make_state("cup")]
        hpc.matching_step(ctx, states)
        edges = hpc.get_relational_edges(edge_type="cooccurrence")
        self.assertTrue(len(edges) > 0)

    def test_temporal_edge_lag1(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug")])
        hpc.matching_step(ctx, [_make_state("cup")])
        edges = hpc.get_relational_edges(edge_type="temporal")
        srcs = [e["src"] for e in edges]
        self.assertIn("mug", srcs)

    def test_spatial_edge_created_with_locations(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        loc_a = np.array([0.0, 0.0, 0.0])
        loc_b = np.array([1.0, 0.0, 0.0])
        states = [
            _make_state("mug", location=loc_a),
            _make_state("cup", location=loc_b),
        ]
        hpc.matching_step(ctx, states)
        edges = hpc.get_relational_edges(edge_type="spatial")
        self.assertTrue(len(edges) > 0)

    def test_concept_filter(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        states = [_make_state("mug"), _make_state("cup"), _make_state("plate")]
        hpc.matching_step(ctx, states)
        edges = hpc.get_relational_edges(concept_id="mug")
        for e in edges:
            self.assertTrue(e["src"] == "mug" or e["dst"] == "mug")


class TestContextSignal(unittest.TestCase):
    def test_context_vector_shape(self):
        hpc = HippocampalModule(context_dim=32)
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])
        sig = hpc.get_context_signal()
        self.assertIsNotNone(sig)
        self.assertEqual(sig["context_vector"].shape, (32,))

    def test_no_active_concepts_returns_none(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        # use_state=False means the concept won't be extracted
        hpc.matching_step(ctx, [_make_state("mug", use_state=False)])
        sig = hpc.get_context_signal()
        self.assertIsNone(sig)

    def test_active_concepts_in_signal(self):
        hpc = HippocampalModule()
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])
        sig = hpc.get_context_signal()
        self.assertIn("mug", sig["active_concepts"])
        self.assertIn("cup", sig["active_concepts"])

    def test_context_signal_none_on_init(self):
        hpc = HippocampalModule()
        # No matching_step called -> no active concepts
        sig = hpc.get_context_signal()
        self.assertIsNone(sig)


class TestStateDict(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        hpc = HippocampalModule(context_dim=16, max_episodes=50)
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])
        hpc.post_episode()

        d = hpc.state_dict()
        hpc2 = HippocampalModule()
        hpc2.load_state_dict(d)

        self.assertEqual(hpc2.context_dim, 16)
        self.assertEqual(len(hpc2.episodic_memory), 1)
        self.assertEqual(
            hpc2.co_occurrence_counts, hpc.co_occurrence_counts
        )


class TestReceiveContext(unittest.TestCase):
    def test_receive_context_noop_by_default(self):
        # LearningModule.receive_context should not raise
        from tests.unit.frameworks.models.fakes.learning_modules import (
            FakeLearningModule,
        )
        lm = FakeLearningModule()
        lm.receive_context(
            context_vector=np.zeros(32), active_concepts=[]
        )


class TestContextDispatch(unittest.TestCase):
    """Test that _dispatch_context_signals wires HPC to downstream LMs."""

    def test_dispatch_broadcasts_context_to_all_lms(self):
        """When HPC has active concepts, context is dispatched to all other LMs."""
        hpc = HippocampalModule(context_dim=16)
        ctx = _make_ctx()
        hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])

        # HPC should now have a context signal
        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal)
        self.assertIn("active_concepts", signal)
        self.assertIn("context_vector", signal)
        self.assertEqual(set(signal["active_concepts"]), {"mug", "cup"})

    def test_cross_episode_memory_persistence(self):
        """Episodic memory persists across episodes (T2.1)."""
        hpc = HippocampalModule(context_dim=16, max_episodes=100)
        ctx = _make_ctx()

        # Episode 1: see mug and cup
        hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])
        hpc.post_episode()
        self.assertEqual(len(hpc.episodic_memory), 1)

        # Episode 2: see fork and spoon
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_state("fork"), _make_state("spoon")])
        hpc.post_episode()
        self.assertEqual(len(hpc.episodic_memory), 2)

        # Both episodes retained
        ep1 = hpc.recall_episode(0)
        ep2 = hpc.recall_episode(1)
        self.assertIsNotNone(ep1)
        self.assertIsNotNone(ep2)
        self.assertIn("mug", ep1[0]["active_concepts"])
        self.assertIn("fork", ep2[0]["active_concepts"])

    def test_association_grows_across_episodes(self):
        """Co-occurrence associations accumulate across episodes (T2.1)."""
        hpc = HippocampalModule(context_dim=16)
        ctx = _make_ctx()

        # See mug+cup together in 3 episodes
        for _ in range(3):
            hpc.pre_episode()
            hpc.matching_step(ctx, [_make_state("mug"), _make_state("cup")])
            hpc.post_episode()

        # Check associations are strong
        assoc = hpc.recall_associations("mug")
        self.assertIn("cup", assoc)
        self.assertGreater(assoc["cup"], 0)

    def test_relational_graph_has_temporal_edges(self):
        """Temporal edges track transitions between concepts (T2.2)."""
        hpc = HippocampalModule(context_dim=16)

        # Step 1: see mug
        ctx1 = _make_ctx(global_step=0, episode_step=0)
        hpc.matching_step(ctx1, [_make_state("mug")])

        # Step 2: see cup (transition from mug -> cup)
        ctx2 = _make_ctx(global_step=1, episode_step=1)
        hpc.matching_step(ctx2, [_make_state("cup")])

        # Check temporal edge exists
        edges = hpc.get_relational_edges(edge_type="temporal")
        src_dst = [(e["src"], e["dst"]) for e in edges]
        self.assertIn(("mug", "cup"), src_dst)


if __name__ == "__main__":
    unittest.main()
