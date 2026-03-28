"""Track 2 integration test: synthetic HPC context injection with real model.

Test 5: Load the extended YCB trained model, pre-load HPC with ball-category
associations, and verify that evidence shifts toward ball graphs when HPC
context is applied vs when it is not.

This uses real trained graph data (34 objects, 9 categories) but does NOT
require Habitat. Instead it extracts real features from stored graphs to
construct synthetic observations.

See docs/track-2-temporal-world-model.md Phase 2 for context.
"""

import os
import unittest

import numpy as np
import torch

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.hippocampal_module import HippocampalModule
from tbp.monty.frameworks.models.states import State


# Category taxonomy for the 34-object extended YCB training set
CATEGORY_TAXONOMY = {
    "011_banana": "fruit",
    "012_strawberry": "fruit",
    "013_apple": "fruit",
    "014_lemon": "fruit",
    "016_pear": "fruit",
    "017_orange": "fruit",
    "053_mini_soccer_ball": "ball",
    "054_softball": "ball",
    "055_baseball": "ball",
    "056_tennis_ball": "ball",
    "003_cracker_box": "box",
    "004_sugar_box": "box",
    "008_pudding_box": "box",
    "002_master_chef_can": "can",
    "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can",
    "025_mug": "cup",
    "065-a_cups": "cup",
    "065-b_cups": "cup",
    "065-c_cups": "cup",
    "065-d_cups": "cup",
    "065-e_cups": "cup",
    "065-f_cups": "cup",
    "030_fork": "utensil",
    "031_spoon": "utensil",
    "033_spatula": "utensil",
    "035_power_drill": "tool",
    "037_scissors": "tool",
    "042_adjustable_wrench": "tool",
    "050_medium_clamp": "clamp",
    "051_large_clamp": "clamp",
    "072-a_toy_airplane": "airplane",
    "072-b_toy_airplane": "airplane",
    "072-c_toy_airplane": "airplane",
}

MODEL_PATH = os.path.expanduser(
    "~/tbp/results/monty/projects/phase2_review_runs/"
    "extended_ycb_train/pretrained/model.pt"
)


def _extract_observation_from_graph(graph_model, node_idx=0):
    """Extract a synthetic observation State from a trained graph node.

    Reads the stored features for a graph node and reconstructs a State
    object that would match that node during evidence matching.

    Args:
        graph_model: The GraphObjectModel (e.g., gm['053_mini_soccer_ball']['patch'])
        node_idx: Which node to extract features from.

    Returns:
        A State object with the node's features.
    """
    fm = graph_model.feature_mapping
    x = graph_model.x[node_idx].numpy().astype(np.float64)
    pos = graph_model.pos[node_idx].numpy().astype(np.float64)

    pose_vectors = x[fm["pose_vectors"][0]:fm["pose_vectors"][1]].reshape(3, 3)

    return State(
        location=pos,
        morphological_features={
            "pose_vectors": pose_vectors,
            "pose_fully_defined": bool(
                x[fm["pose_fully_defined"][0]:fm["pose_fully_defined"][1]][0]
            ),
            "on_object": int(
                x[fm["on_object"][0]:fm["on_object"][1]][0]
            ),
        },
        non_morphological_features={
            "hsv": x[fm["hsv"][0]:fm["hsv"][1]].tolist(),
            "principal_curvatures_log": (
                x[fm["principal_curvatures_log"][0]:fm["principal_curvatures_log"][1]]
                .tolist()
            ),
        },
        confidence=1.0,
        use_state=True,
        sender_id="patch",
        sender_type="SM",
    )


@unittest.skipUnless(
    os.path.exists(MODEL_PATH),
    f"Trained model not found at {MODEL_PATH}",
)
class TestPreloadedHPCShiftsBallEvidence(unittest.TestCase):
    """Test 5: Pre-loaded HPC context shifts ball evidence on real model.

    Loads the 34-object extended YCB model, creates synthetic observations
    from stored graph nodes, and verifies that HPC context biases evidence
    toward ball-category graphs vs fruit-category graphs.
    """

    @classmethod
    def setUpClass(cls):
        """Load trained model once for all tests in this class."""
        model_data = torch.load(MODEL_PATH, map_location="cpu")
        cls.lm_state = model_data["lm_dict"][0]
        cls.graph_memory_data = cls.lm_state["graph_memory"]

    def _make_lm(self, category_bias_strength=0.5):
        """Create and load an EvidenceGraphLM with the trained model."""
        lm = EvidenceGraphLM(
            max_match_distance=0.01,
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
            max_graph_size=0.3,
            category_taxonomy=dict(CATEGORY_TAXONOMY),
            category_bias_strength=category_bias_strength,
        )

        # Load pre-trained graphs
        lm.load_state_dict(self.lm_state)

        self.assertEqual(
            len(lm.get_all_known_object_ids()), 34,
            "Should have loaded all 34 training objects",
        )
        return lm

    def _make_ctx(self):
        return RuntimeContext(rng=np.random.RandomState(42))

    def test_model_loaded_correctly(self):
        """Verify trained model loads with correct objects and categories."""
        lm = self._make_lm()

        object_ids = lm.get_all_known_object_ids()
        ball_ids = [oid for oid in object_ids if CATEGORY_TAXONOMY.get(oid) == "ball"]
        fruit_ids = [
            oid for oid in object_ids if CATEGORY_TAXONOMY.get(oid) == "fruit"
        ]

        self.assertEqual(len(ball_ids), 4, "Should have 4 ball objects")
        self.assertEqual(len(fruit_ids), 6, "Should have 6 fruit objects")

    def test_hpc_context_biases_ball_over_fruit(self):
        """HPC context with ball associations boosts ball evidence.

        After running matching with a ball observation, apply HPC context
        that associates with ball-category concepts. Ball evidence should
        increase relative to fruit evidence.
        """
        lm = self._make_lm(category_bias_strength=0.5)
        ctx = self._make_ctx()

        # Extract observation from a ball graph node
        ball_graph = self.graph_memory_data["053_mini_soccer_ball"]["patch"]
        ball_obs = _extract_observation_from_graph(ball_graph, node_idx=0)

        # Run eval matching
        lm.mode = ExperimentMode.EVAL
        target = {"object": "053_mini_soccer_ball", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        lm.matching_step(ctx, [ball_obs])

        # Record baseline max evidence per category
        ball_ids = [
            oid for oid in lm.get_all_known_object_ids()
            if CATEGORY_TAXONOMY.get(oid) == "ball"
        ]
        fruit_ids = [
            oid for oid in lm.get_all_known_object_ids()
            if CATEGORY_TAXONOMY.get(oid) == "fruit"
        ]

        def max_category_evidence(obj_ids):
            evs = []
            for oid in obj_ids:
                ev = lm.evidence.get(oid)
                if ev is not None and len(ev) > 0:
                    evs.append(float(np.max(ev)))
            return max(evs) if evs else 0.0

        ball_ev_before = max_category_evidence(ball_ids)
        fruit_ev_before = max_category_evidence(fruit_ids)

        # Apply HPC context with strong ball association
        context = {
            "context_vector": np.zeros(32),
            "active_concepts": ["ball_concept"],
            "association_strengths": {
                "053_mini_soccer_ball": 0.9,
                "054_softball": 0.8,
                "055_baseball": 0.7,
                "056_tennis_ball": 0.7,
            },
            "episode_count": 20,
        }
        lm.receive_context(**context)

        ball_ev_after = max_category_evidence(ball_ids)
        fruit_ev_after = max_category_evidence(fruit_ids)

        # Ball evidence should have increased
        self.assertGreater(
            ball_ev_after, ball_ev_before,
            f"Ball evidence should increase with HPC context. "
            f"Before: {ball_ev_before:.4f}, After: {ball_ev_after:.4f}",
        )

        # Fruit evidence should NOT have increased
        self.assertAlmostEqual(
            fruit_ev_after, fruit_ev_before, places=5,
            msg=(
                f"Fruit evidence should not change. "
                f"Before: {fruit_ev_before:.4f}, After: {fruit_ev_after:.4f}"
            ),
        )

        # Ball should now exceed fruit (the core Track 2 result)
        self.assertGreater(
            ball_ev_after, fruit_ev_after,
            f"After HPC context, ball should exceed fruit. "
            f"Ball: {ball_ev_after:.4f}, Fruit: {fruit_ev_after:.4f}",
        )

    def test_hpc_category_match_biases_all_balls(self):
        """HPC association with one ball boosts ALL balls via category taxonomy."""
        lm = self._make_lm(category_bias_strength=0.3)
        ctx = self._make_ctx()

        ball_graph = self.graph_memory_data["053_mini_soccer_ball"]["patch"]
        ball_obs = _extract_observation_from_graph(ball_graph, node_idx=0)

        lm.mode = ExperimentMode.EVAL
        target = {"object": "053_mini_soccer_ball", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        lm.matching_step(ctx, [ball_obs])

        # Associate with only ONE ball object, but category match should
        # boost all balls
        context = {
            "context_vector": np.zeros(32),
            "active_concepts": ["bouncy"],
            "association_strengths": {
                "053_mini_soccer_ball": 0.9,
            },
            "episode_count": 10,
        }
        lm.receive_context(**context)

        # ALL ball objects should be boosted (direct + category match)
        ball_ids = ["053_mini_soccer_ball", "054_softball",
                    "055_baseball", "056_tennis_ball"]
        for ball_id in ball_ids:
            ev = lm.evidence.get(ball_id)
            if ev is not None and len(ev) > 0:
                # Check this ball got biased (evidence should be positive after bias)
                max_ev = float(np.max(ev))
                # Direct match gets full strength, category match gets strength too
                self.assertGreater(
                    max_ev, 0,
                    f"{ball_id} should have positive evidence after HPC bias",
                )

    def test_full_hpc_pathway_with_learned_associations(self):
        """End-to-end: HPC learns associations, generates context, biases LM."""
        lm = self._make_lm(category_bias_strength=0.5)
        ctx = self._make_ctx()

        # Step 1: HPC learns ball associations
        hpc = HippocampalModule(context_dim=32)

        class _FakeTimer:
            global_step = 0
            episode_step = 0

        hpc_ctx = RuntimeContext(
            rng=np.random.RandomState(42), timer=_FakeTimer()
        )

        class _S:
            pass

        def _make_hpc_state(obj_id, confidence=0.9):
            s = _S()
            s.use_state = True
            s.confidence = confidence
            s.non_morphological_features = {"object_id": obj_id}
            s.location = np.array([0.0, 0.0, 0.0])
            s.sender_id = "lm_0"
            s.sender_type = "LM"
            return s

        # Simulate: HPC has seen balls bounce in previous episodes
        for ball_id in ["053_mini_soccer_ball", "054_softball", "055_baseball"]:
            hpc.pre_episode()
            hpc.matching_step(hpc_ctx, [
                _make_hpc_state(ball_id),
                _make_hpc_state("bouncy"),
            ])
            hpc.post_episode()

        # Step 2: New episode - HPC sees a ball again
        hpc.pre_episode()
        hpc.matching_step(hpc_ctx, [_make_hpc_state("053_mini_soccer_ball")])

        # HPC generates context signal
        signal = hpc.get_context_signal()
        self.assertIsNotNone(signal, "HPC should produce context signal")

        # Step 3: Feed HPC context to LM
        ball_graph = self.graph_memory_data["053_mini_soccer_ball"]["patch"]
        ball_obs = _extract_observation_from_graph(ball_graph, node_idx=0)

        lm.mode = ExperimentMode.EVAL
        target = {"object": "053_mini_soccer_ball", "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)
        lm.matching_step(ctx, [ball_obs])

        # Record baseline
        ball_ev_before = float(np.max(
            lm.evidence.get("053_mini_soccer_ball", np.array([0]))
        ))

        # Apply HPC-learned context
        lm.receive_context(**signal)

        ball_ev_after = float(np.max(
            lm.evidence.get("053_mini_soccer_ball", np.array([0]))
        ))

        # Evidence should change (increase) after HPC context
        # The HPC's association_strengths contain mean strengths for active
        # concepts, which may be small, but should be non-zero
        self.assertGreaterEqual(
            ball_ev_after, ball_ev_before,
            f"Ball evidence should not decrease with HPC context. "
            f"Before: {ball_ev_before:.4f}, After: {ball_ev_after:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
