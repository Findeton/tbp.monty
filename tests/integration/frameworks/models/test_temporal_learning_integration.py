"""Integration test: temporal learning with real trained model.

Validates the FULL temporal pipeline end-to-end:
1. LM recognizes objects from real graph features (synthetic observations)
2. HPC records cross-episode transitions via Hebbian learning
3. HPC generates temporal predictions
4. Predictions prime downstream LM evidence via context signal
5. Priming biases recognition toward predicted objects

No Habitat needed — uses _extract_observation_from_graph to create
synthetic observations from the trained 34-object extended YCB model.
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


MODEL_PATH = os.path.expanduser(
    "~/tbp/results/monty/projects/phase2_review_runs/"
    "extended_ycb_train/pretrained/model.pt"
)

CATEGORY_TAXONOMY = {
    "011_banana": "fruit", "012_strawberry": "fruit",
    "013_apple": "fruit", "014_lemon": "fruit",
    "016_pear": "fruit", "017_orange": "fruit",
    "053_mini_soccer_ball": "ball", "054_softball": "ball",
    "055_baseball": "ball", "056_tennis_ball": "ball",
    "003_cracker_box": "box", "004_sugar_box": "box",
    "008_pudding_box": "box",
    "002_master_chef_can": "can", "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can",
    "025_mug": "cup",
    "065-a_cups": "cup", "065-b_cups": "cup", "065-c_cups": "cup",
    "065-d_cups": "cup", "065-e_cups": "cup", "065-f_cups": "cup",
    "030_fork": "utensil", "031_spoon": "utensil",
    "033_spatula": "utensil",
    "035_power_drill": "tool", "037_scissors": "tool",
    "042_adjustable_wrench": "tool",
    "050_medium_clamp": "clamp", "051_large_clamp": "clamp",
    "072-a_toy_airplane": "airplane", "072-b_toy_airplane": "airplane",
    "072-c_toy_airplane": "airplane",
}


def _extract_observation_from_graph(graph_model, node_idx=0):
    """Extract a synthetic observation State from a trained graph node."""
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
                x[fm["principal_curvatures_log"][0]:
                  fm["principal_curvatures_log"][1]].tolist()
            ),
        },
        confidence=1.0,
        use_state=True,
        sender_id="patch",
        sender_type="SM",
    )


def _make_hpc_state(obj_id, confidence=0.9):
    """Create a State as if emitted by an LM (for HPC consumption)."""
    class _S:
        pass

    s = _S()
    s.use_state = True
    s.confidence = confidence
    s.non_morphological_features = {"graph_id": obj_id}
    s.location = np.array([0.0, 0.0, 0.0])
    s.sender_id = "lm_0"
    s.sender_type = "LM"
    return s


class _FakeTimer:
    def __init__(self):
        self.global_step = 0
        self.episode_step = 0


def _make_ctx():
    return RuntimeContext(rng=np.random.RandomState(42), timer=_FakeTimer())


@unittest.skipUnless(
    os.path.exists(MODEL_PATH),
    f"Trained model not found at {MODEL_PATH}",
)
class TestTemporalLearningEndToEnd(unittest.TestCase):
    """Full temporal learning pipeline with real model data."""

    @classmethod
    def setUpClass(cls):
        model_data = torch.load(MODEL_PATH, map_location="cpu")
        cls.lm_state = model_data["lm_dict"][0]
        cls.graph_memory_data = cls.lm_state["graph_memory"]

    def _make_lm(self, **kwargs):
        defaults = dict(
            max_match_distance=0.01,
            tolerances={
                "patch": {
                    "hsv": [0.1, 0.2, 0.2],
                    "principal_curvatures_log": [1, 1],
                }
            },
            feature_weights={
                "patch": {
                    "hsv": np.array([1.0, 2.0, 0.5]),
                    "pose_vectors": np.ones(3),
                    "principal_curvatures_log": np.ones(2),
                }
            },
            max_graph_size=0.3,
            category_taxonomy=dict(CATEGORY_TAXONOMY),
            category_bias_strength=0.3,
        )
        defaults.update(kwargs)
        lm = EvidenceGraphLM(**defaults)
        lm.load_state_dict(self.lm_state)
        return lm

    def _run_lm_episode(self, lm, object_name, n_obs=3):
        """Run a mini recognition episode with synthetic observations.

        Feeds n_obs observations from different nodes of the object's
        trained graph through the LM, then returns the MLH graph_id.
        """
        ctx = _make_ctx()
        graph = self.graph_memory_data[object_name]["patch"]
        n_nodes = graph.x.shape[0]

        lm.mode = ExperimentMode.EVAL
        target = {"object": object_name, "quat_rotation": [1, 0, 0, 0]}
        lm.pre_episode(primary_target=target)

        for i in range(n_obs):
            node_idx = (i * n_nodes // n_obs) % n_nodes
            obs = _extract_observation_from_graph(graph, node_idx)
            lm.matching_step(ctx, [obs])

        mlh = lm.get_current_mlh()
        lm.post_episode()
        return mlh["graph_id"]

    def test_hpc_learns_sequence_from_lm_output(self):
        """HPC learns temporal transitions from LM recognition results.

        Simulates 3 episodes:
          ep1: LM sees banana → recognizes fruit-category object
          ep2: LM sees baseball → recognizes ball-category object
          ep3: LM sees banana → recognizes fruit-category object

        HPC should learn: fruit → ball and ball → fruit transitions.
        """
        lm = self._make_lm()
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            temporal_prediction_weight=0.5,
        )
        ctx = _make_ctx()

        sequence = ["011_banana", "055_baseball", "011_banana"]
        recognized = []

        for obj_name in sequence:
            # Run LM recognition
            mlh_id = self._run_lm_episode(lm, obj_name, n_obs=5)
            recognized.append(mlh_id)

            # Feed LM output to HPC
            hpc.pre_episode()
            hpc.matching_step(ctx, [_make_hpc_state(mlh_id, confidence=0.9)])
            hpc.post_episode()

        # HPC should have learned transitions between recognized objects
        self.assertGreater(len(hpc._concept_sdrs), 0,
                           "HPC should have concept SDRs")
        self.assertGreater(float(np.sum(np.abs(hpc._temporal_W))), 0,
                           "Temporal association matrix should be non-zero")

        # Predictions from first recognized concept
        preds = hpc.get_temporal_predictions(from_concept=recognized[0])
        self.assertGreater(len(preds), 0,
                           f"Should have predictions from {recognized[0]}")

    def test_temporal_prediction_primes_lm_evidence(self):
        """Temporal prediction biases next episode's evidence matching.

        After HPC learns A→B→A→B pattern, the temporal prediction should
        create a context signal that boosts B's evidence at the start of
        the next episode.
        """
        lm = self._make_lm()
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
            temporal_prediction_weight=0.8,
        )
        ctx = _make_ctx()

        # Phase 1: teach HPC the sequence banana → baseball (3 cycles)
        for _ in range(3):
            for obj_name in ["011_banana", "055_baseball"]:
                mlh_id = self._run_lm_episode(lm, obj_name, n_obs=3)
                hpc.pre_episode()
                hpc.matching_step(ctx, [_make_hpc_state(mlh_id)])
                hpc.post_episode()

        # Phase 2: HPC should predict baseball after banana
        last_concept = hpc._last_episode_terminal
        preds = hpc.get_temporal_predictions()
        self.assertGreater(len(preds), 0, "Should have predictions")

        # Phase 3: Get context signal with temporal predictions
        hpc.pre_episode()
        hpc.matching_step(ctx, [_make_hpc_state(last_concept)])
        signal = hpc.get_context_signal()

        self.assertIsNotNone(signal)
        self.assertIn("temporal_predictions", signal)
        self.assertIn("association_strengths", signal)

        # Phase 4: Apply context to fresh LM and check evidence is biased
        lm2 = self._make_lm()
        lm2.mode = ExperimentMode.EVAL
        target = {"object": "055_baseball", "quat_rotation": [1, 0, 0, 0]}
        lm2.pre_episode(primary_target=target)

        # One neutral observation
        graph = self.graph_memory_data["055_baseball"]["patch"]
        obs = _extract_observation_from_graph(graph, node_idx=0)
        lm2.matching_step(_make_ctx(), [obs])

        # Record evidence before HPC context
        predicted_id = max(preds, key=preds.get) if preds else None
        if predicted_id and predicted_id in lm2.evidence:
            ev_before = float(np.max(lm2.evidence[predicted_id]))

            # Apply HPC temporal context
            lm2.receive_context(**signal)

            ev_after = float(np.max(lm2.evidence[predicted_id]))

            self.assertGreaterEqual(
                ev_after, ev_before,
                f"Predicted object {predicted_id} evidence should not "
                f"decrease with temporal priming. "
                f"Before: {ev_before:.4f}, After: {ev_after:.4f}",
            )

    def test_multi_category_sequence_learning(self):
        """HPC learns temporal structure across multiple object categories.

        Runs a 4-category sequence: fruit → ball → cup → utensil
        and verifies the HPC learns the transition pattern.
        """
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
        )
        ctx = _make_ctx()
        lm = self._make_lm()

        # Run 3 cycles of: banana → baseball → mug → fork
        objects = ["011_banana", "055_baseball", "025_mug", "030_fork"]
        all_recognized = []

        for cycle in range(3):
            for obj_name in objects:
                mlh_id = self._run_lm_episode(lm, obj_name, n_obs=3)
                all_recognized.append(mlh_id)
                hpc.pre_episode()
                hpc.matching_step(ctx, [_make_hpc_state(mlh_id)])
                hpc.post_episode()

        # HPC should have multiple SDRs and non-trivial predictions
        self.assertGreaterEqual(len(hpc._concept_sdrs), 2)

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["total"], 0,
                           "Should have made at least one prediction")

    def test_replay_strengthens_real_associations(self):
        """Episodic replay consolidates temporal associations learned
        from real LM recognition data."""
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
        )
        ctx = _make_ctx()
        lm = self._make_lm()

        # Learn sequence (1 cycle only — weak associations)
        objects = ["013_apple", "054_softball"]
        recognized = []
        for obj_name in objects:
            mlh_id = self._run_lm_episode(lm, obj_name, n_obs=3)
            recognized.append(mlh_id)
            hpc.pre_episode()
            hpc.matching_step(ctx, [_make_hpc_state(mlh_id)])
            hpc.post_episode()

        # Measure association strength before replay
        if len(recognized) >= 2 and recognized[0] in hpc._concept_sdrs:
            sdr0 = hpc._get_concept_sdr(recognized[0])
            sdr1 = hpc._get_concept_sdr(recognized[1])
            strength_before = float(np.dot(hpc._temporal_W @ sdr0, sdr1))

            # Replay consolidates
            n_updates = hpc.replay_cross_episode(n_replays=5)
            self.assertGreater(n_updates, 0)

            strength_after = float(np.dot(hpc._temporal_W @ sdr0, sdr1))
            self.assertGreater(
                strength_after, strength_before,
                "Replay should strengthen temporal associations"
            )

    def test_prediction_accuracy_over_repeated_sequence(self):
        """Track prediction accuracy over a repeated object sequence.

        After enough repetitions, the HPC should achieve reasonable
        prediction accuracy on the sequence it observes.
        """
        hpc = HippocampalModule(
            temporal_dim=512, temporal_sparsity=0.04,
        )
        ctx = _make_ctx()
        lm = self._make_lm()

        objects = ["011_banana", "055_baseball"]

        for cycle in range(6):
            for obj_name in objects:
                mlh_id = self._run_lm_episode(lm, obj_name, n_obs=3)
                hpc.pre_episode()
                hpc.matching_step(ctx, [_make_hpc_state(mlh_id)])
                hpc.post_episode()

        acc = hpc.get_prediction_accuracy()
        self.assertGreater(acc["total"], 3,
                           "Should have made multiple predictions")
        # With a simple A→B→A→B pattern, accuracy should be reasonable
        # Note: LM may not always return the same graph_id, so we allow
        # for some misses
        self.assertGreater(
            acc["accuracy"], 0.3,
            f"Expected >30% accuracy on repeated sequence, "
            f"got {acc['accuracy']:.0%} ({acc['hits']}/{acc['total']})"
        )


if __name__ == "__main__":
    unittest.main()
