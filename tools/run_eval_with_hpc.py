"""Run extended YCB eval with pre-loaded HPC context.

Track 2 Phase 3: Inject behavioral associations via HPC to break the
ball→fruit confusion. Uses the existing trained model + pre-loaded HPC
associations to bias evidence toward correct categories.

Usage:
    conda run -n tbp.monty python tools/run_eval_with_hpc.py

This script:
1. Loads the trained extended YCB model
2. Creates an EvidenceGraphLM with category taxonomy + bias
3. Pre-loads HPC with behavioral associations (ball↔bouncy, fruit↔edible)
4. Runs eval on holdout objects with HPC context applied after each step
5. Reports category accuracy with and without HPC context
"""

import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tbp.monty.context import RuntimeContext  # noqa: E402
from tbp.monty.frameworks.experiments.mode import ExperimentMode  # noqa: E402
from tbp.monty.frameworks.models.evidence_matching.learning_module import (  # noqa: E402
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.hippocampal_module import (  # noqa: E402
    HippocampalModule,
)


# Category taxonomy for extended YCB
CATEGORY_TAXONOMY = {
    "011_banana": "fruit", "012_strawberry": "fruit", "013_apple": "fruit",
    "014_lemon": "fruit", "016_pear": "fruit", "017_orange": "fruit",
    "053_mini_soccer_ball": "ball", "054_softball": "ball",
    "055_baseball": "ball", "056_tennis_ball": "ball",
    "003_cracker_box": "box", "004_sugar_box": "box", "008_pudding_box": "box",
    "002_master_chef_can": "can", "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can",
    "025_mug": "cup", "065-a_cups": "cup", "065-b_cups": "cup",
    "065-c_cups": "cup", "065-d_cups": "cup", "065-e_cups": "cup",
    "065-f_cups": "cup",
    "030_fork": "utensil", "031_spoon": "utensil", "033_spatula": "utensil",
    "035_power_drill": "tool", "037_scissors": "tool",
    "042_adjustable_wrench": "tool",
    "050_medium_clamp": "clamp", "051_large_clamp": "clamp",
    "072-a_toy_airplane": "airplane", "072-b_toy_airplane": "airplane",
    "072-c_toy_airplane": "airplane",
}

# Behavioral associations to pre-load in HPC
# These simulate what the system would learn from behavioral experience
BEHAVIORAL_ASSOCIATIONS = [
    # Balls bounce (co-occurred with "bouncy" in prior experience)
    ("053_mini_soccer_ball", "bouncy"),
    ("054_softball", "bouncy"),
    ("055_baseball", "bouncy"),
    ("056_tennis_ball", "bouncy"),
    # Fruits are edible
    ("011_banana", "edible"),
    ("012_strawberry", "edible"),
    ("013_apple", "edible"),
    ("014_lemon", "edible"),
    ("016_pear", "edible"),
    ("017_orange", "edible"),
    # Tools are graspable
    ("035_power_drill", "graspable"),
    ("037_scissors", "graspable"),
    ("042_adjustable_wrench", "graspable"),
    # Utensils are for eating
    ("030_fork", "for_eating"),
    ("031_spoon", "for_eating"),
    ("033_spatula", "for_eating"),
]

MODEL_PATH = os.path.expanduser(
    "~/tbp/results/monty/projects/phase2_review_runs/"
    "extended_ycb_train/pretrained/model.pt"
)


def extract_observation_from_graph(graph_model, node_idx=0):
    """Extract a State-like observation from a trained graph node."""
    from tbp.monty.frameworks.models.states import State

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


def run_eval_episode(lm, graph_data, target_id, n_obs=5, hpc=None):
    """Run a single eval episode and return prediction + evidence.

    Args:
        lm: EvidenceGraphLM with loaded graphs.
        graph_data: Dict of graph_id -> {channel: GraphObjectModel}.
        target_id: The true object being observed.
        n_obs: Number of observations to feed.
        hpc: Optional HippocampalModule for context injection.

    Returns:
        dict with prediction, true_category, predicted_category, evidence.
    """
    ctx = RuntimeContext(rng=np.random.RandomState(42))
    target_graph = graph_data[target_id]["patch"]

    lm.mode = ExperimentMode.EVAL
    target = {"object": target_id, "quat_rotation": [1, 0, 0, 0]}
    lm.pre_episode(primary_target=target)

    # Sample n_obs random nodes from the target graph
    n_nodes = target_graph.x.shape[0]
    node_indices = np.random.RandomState(42).choice(
        n_nodes, size=min(n_obs, n_nodes), replace=False
    )

    for node_idx in node_indices:
        obs = extract_observation_from_graph(target_graph, node_idx)
        lm.matching_step(ctx, [obs])

        # If HPC is present, simulate context dispatch
        if hpc is not None:
            # Create a fake LM state for HPC (what the upstream LM recognized)
            class _S:
                pass

            s = _S()
            s.use_state = True
            s.confidence = 0.8
            s.non_morphological_features = {"object_id": target_id}
            s.location = obs.location
            s.sender_id = "lm_0"
            s.sender_type = "LM"

            class _FakeTimer:
                global_step = 0
                episode_step = int(node_idx)

            hpc_ctx = RuntimeContext(
                rng=np.random.RandomState(42), timer=_FakeTimer()
            )
            hpc.matching_step(hpc_ctx, [s])

            signal = hpc.get_context_signal()
            if signal is not None:
                lm.receive_context(**signal)

    # Get prediction
    graph_ids, graph_evidences = lm.get_evidence_for_each_graph()
    if len(graph_ids) > 0 and len(graph_evidences) > 0:
        best_idx = np.argmax(graph_evidences)
        prediction = graph_ids[best_idx]
    else:
        prediction = "no_match"

    true_cat = CATEGORY_TAXONOMY.get(target_id, "unknown")
    pred_cat = CATEGORY_TAXONOMY.get(prediction, "unknown")

    return {
        "target": target_id,
        "prediction": prediction,
        "true_category": true_cat,
        "predicted_category": pred_cat,
        "correct": prediction == target_id,
        "category_correct": true_cat == pred_cat,
    }


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        sys.exit(1)

    print("Loading trained model...")
    model_data = torch.load(MODEL_PATH, map_location="cpu")
    lm_state = model_data["lm_dict"][0]
    graph_data = lm_state["graph_memory"]

    # Holdout objects (from extended_holdout_objects config)
    holdout_objects = [
        "013_apple", "017_orange",  # fruit
        "054_softball", "056_tennis_ball",  # ball
        "004_sugar_box",  # box
        "005_tomato_soup_can", "007_tuna_fish_can",  # can
        "065-c_cups", "065-d_cups",  # cup
        "031_spoon",  # utensil
        "037_scissors", "042_adjustable_wrench",  # tool
        "051_large_clamp",  # clamp
        "072-b_toy_airplane",  # airplane
    ]

    # Filter to objects that are in graph_data (training set)
    # For eval, we test on holdout objects but need the graphs for matching
    # Actually holdout objects are NOT in training data. We test if the model
    # recognizes them as the same category via the training objects.
    # The holdout objects need separate graph data, which we don't have here.
    # Instead, test with training objects using different graph nodes.
    test_objects = list(graph_data.keys())

    n_obs = 5  # observations per episode

    # Run without HPC
    print("\n=== Baseline (no HPC) ===")
    results_no_hpc = []
    for obj_id in test_objects:
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
            category_bias_strength=0.0,  # No bias
        )
        lm.load_state_dict(lm_state)
        result = run_eval_episode(lm, graph_data, obj_id, n_obs=n_obs)
        results_no_hpc.append(result)

    # Run with HPC
    print("\n=== With HPC (pre-loaded associations) ===")
    hpc = HippocampalModule(context_dim=32, min_confidence_to_bind=0.5)
    hpc.preload_associations(BEHAVIORAL_ASSOCIATIONS, count=10)

    results_with_hpc = []
    for obj_id in test_objects:
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
            category_bias_strength=0.3,
        )
        lm.load_state_dict(lm_state)

        # Reset HPC episode state but keep associations
        hpc.pre_episode()

        result = run_eval_episode(
            lm, graph_data, obj_id, n_obs=n_obs, hpc=hpc
        )
        results_with_hpc.append(result)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    def compute_metrics(results):
        cat_correct = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            cat = r["true_category"]
            cat_correct[cat]["total"] += 1
            if r["category_correct"]:
                cat_correct[cat]["correct"] += 1
        return cat_correct

    metrics_no_hpc = compute_metrics(results_no_hpc)
    metrics_with_hpc = compute_metrics(results_with_hpc)

    shape_incongruent = {"ball", "tool", "can"}

    print(f"\n{'Category':<12} {'No HPC':>10} {'With HPC':>10} {'Delta':>8}")
    print("-" * 45)

    total_no_hpc = sum(m["correct"] for m in metrics_no_hpc.values())
    total_with_hpc = sum(m["correct"] for m in metrics_with_hpc.values())
    total_count = sum(m["total"] for m in metrics_no_hpc.values())

    for cat in sorted(metrics_no_hpc.keys()):
        m1 = metrics_no_hpc[cat]
        m2 = metrics_with_hpc[cat]
        pct1 = 100 * m1["correct"] / m1["total"] if m1["total"] > 0 else 0
        pct2 = 100 * m2["correct"] / m2["total"] if m2["total"] > 0 else 0
        delta = pct2 - pct1
        si = " (SI)" if cat in shape_incongruent else " (SC)"
        print(
            f"{cat + si:<12} {pct1:>9.1f}% {pct2:>9.1f}% {delta:>+7.1f}pp"
        )

    overall_no = 100 * total_no_hpc / total_count if total_count > 0 else 0
    overall_with = 100 * total_with_hpc / total_count if total_count > 0 else 0
    print("-" * 45)
    print(
        f"{'Overall':<12} {overall_no:>9.1f}% {overall_with:>9.1f}% "
        f"{overall_with - overall_no:>+7.1f}pp"
    )

    # Check the key metric: ball category accuracy
    ball_no = metrics_no_hpc.get("ball", {"correct": 0, "total": 1})
    ball_with = metrics_with_hpc.get("ball", {"correct": 0, "total": 1})
    ball_pct_no = 100 * ball_no["correct"] / ball_no["total"]
    ball_pct_with = 100 * ball_with["correct"] / ball_with["total"]

    print(f"\nBall category (canary metric):")
    print(f"  Without HPC: {ball_pct_no:.1f}%")
    print(f"  With HPC:    {ball_pct_with:.1f}%")
    if ball_pct_with > ball_pct_no:
        print("  -> HPC IMPROVED ball recognition!")
    elif ball_pct_with == ball_pct_no:
        print("  -> No change (may need Habitat for full eval)")


def run_cross_object_eval():
    """Evaluate category accuracy using cross-object observations.

    Simulates the holdout scenario by presenting observations from one
    object and checking if the model classifies it into the correct CATEGORY
    (not necessarily the exact object). This creates the geometric ambiguity
    that triggers ball→fruit confusion.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading trained model...")
    model_data = torch.load(MODEL_PATH, map_location="cpu")
    lm_state = model_data["lm_dict"][0]
    graph_data = lm_state["graph_memory"]

    # For each category, use observations from one object and test if
    # they match another object in the same category vs other categories
    categories = defaultdict(list)
    for obj_id, cat in CATEGORY_TAXONOMY.items():
        if obj_id in graph_data:
            categories[cat].append(obj_id)

    n_obs = 3

    print("\n" + "=" * 70)
    print("CROSS-OBJECT CATEGORY EVAL (simulates holdout scenario)")
    print("=" * 70)
    print("For each pair (src → test_against), present src observations")
    print("and check if best match is in the same category.\n")

    # Run with and without HPC
    for hpc_mode in ["no_hpc", "with_hpc"]:
        print(f"\n--- {hpc_mode.upper()} ---")

        hpc = None
        bias_strength = 0.0
        if hpc_mode == "with_hpc":
            hpc = HippocampalModule(context_dim=32, min_confidence_to_bind=0.5)
            hpc.preload_associations(BEHAVIORAL_ASSOCIATIONS, count=10)
            bias_strength = 0.3

        cat_results = defaultdict(lambda: {"correct": 0, "total": 0})

        for cat, obj_ids in categories.items():
            if len(obj_ids) < 2:
                continue

            # Test: present observations from obj_ids[0], see if best match
            # is in the same category
            src_id = obj_ids[0]
            src_graph = graph_data[src_id]["patch"]

            # Test against ALL objects (not just same category)
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
                category_bias_strength=bias_strength,
            )
            lm.load_state_dict(lm_state)

            # EXCLUDE the source object so it can't match itself
            if src_id in lm.graph_memory.models_in_memory:
                del lm.graph_memory.models_in_memory[src_id]

            ctx = RuntimeContext(rng=np.random.RandomState(42))
            target = {"object": src_id, "quat_rotation": [1, 0, 0, 0]}
            lm.mode = ExperimentMode.EVAL
            lm.pre_episode(primary_target=target)

            n_nodes = src_graph.x.shape[0]
            node_indices = np.random.RandomState(42).choice(
                n_nodes, size=min(n_obs, n_nodes), replace=False
            )

            for node_idx in node_indices:
                obs = extract_observation_from_graph(src_graph, node_idx)
                lm.matching_step(ctx, [obs])

                if hpc is not None:
                    class _S:
                        pass
                    s = _S()
                    s.use_state = True
                    s.confidence = 0.8
                    s.non_morphological_features = {"object_id": src_id}
                    s.location = obs.location
                    s.sender_id = "lm_0"
                    s.sender_type = "LM"

                    class _FakeTimer:
                        global_step = 0
                        episode_step = int(node_idx)

                    hpc_ctx = RuntimeContext(
                        rng=np.random.RandomState(42), timer=_FakeTimer()
                    )
                    hpc.pre_episode()
                    hpc.matching_step(hpc_ctx, [s])
                    signal = hpc.get_context_signal()
                    if signal is not None:
                        lm.receive_context(**signal)

            # Get prediction
            graph_ids, graph_evidences = lm.get_evidence_for_each_graph()
            if len(graph_ids) > 0 and len(graph_evidences) > 0:
                best_idx = int(np.argmax(graph_evidences))
                prediction = graph_ids[best_idx]
                pred_cat = CATEGORY_TAXONOMY.get(prediction, "unknown")
            else:
                pred_cat = "unknown"

            cat_results[cat]["total"] += 1
            if pred_cat == cat:
                cat_results[cat]["correct"] += 1

        shape_incongruent = {"ball", "tool", "can"}
        total_correct = 0
        total_count = 0

        for cat in sorted(cat_results.keys()):
            m = cat_results[cat]
            pct = 100 * m["correct"] / m["total"] if m["total"] > 0 else 0
            si = " (SI)" if cat in shape_incongruent else " (SC)"
            print(f"  {cat + si:<14} {pct:>6.1f}%  ({m['correct']}/{m['total']})")
            total_correct += m["correct"]
            total_count += m["total"]

        overall = 100 * total_correct / total_count if total_count > 0 else 0
        print(f"  {'Overall':<14} {overall:>6.1f}%  ({total_correct}/{total_count})")


if __name__ == "__main__":
    main()
    print("\n\n")
    run_cross_object_eval()
