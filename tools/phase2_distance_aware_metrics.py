"""Phase 2: Distance-aware evaluation metrics.

Instead of binary correct/incorrect, assigns partial credit based on
representation distance. A system that maps disk->other_disk gets partial
credit; disk->cube gets none.

For holdout objects not in the checkpoint, uses the training-object pairwise
spatial overlap as a proxy: within-category predictions get credit proportional
to the average intra-category spatial overlap; cross-category predictions get
credit proportional to the average inter-category overlap.

Usage:
    python tools/phase2_distance_aware_metrics.py \
        --checkpoint <path_to_model.pt> \
        --eval-stats <path_to_eval_stats.csv> \
        --taxonomy <path_to_taxonomy.yaml> \
        --output <output.json> \
        --lm-index 0 \
        --mapping-key object_to_category
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import KDTree


def load_graphs(checkpoint_path, lm_index=0):
    """Load graph positions from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gm = ckpt["lm_dict"][lm_index]["graph_memory"]
    graphs = {}
    for gid, g_dict in gm.items():
        for key in ("patch", "patch_0", "patch_1"):
            if key in g_dict:
                model = g_dict[key]
                break
        else:
            model = next(iter(g_dict.values()))
        graphs[gid] = np.asarray(model.pos, dtype=np.float64)
    return graphs


def center_and_scale(pos):
    centered = pos - pos.mean(axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered


def compute_spatial_overlap(pos_a, pos_b, threshold=0.05):
    """Compute bidirectional spatial overlap between two graphs."""
    ca = center_and_scale(pos_a)
    cb = center_and_scale(pos_b)

    tree_b = KDTree(cb)
    dists_ab, _ = tree_b.query(ca)
    frac_a_in_b = float(np.mean(dists_ab < threshold))

    tree_a = KDTree(ca)
    dists_ba, _ = tree_a.query(cb)
    frac_b_in_a = float(np.mean(dists_ba < threshold))

    return (frac_a_in_b + frac_b_in_a) / 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-stats", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lm-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--mapping-key", default="object_to_category",
        help="Key in taxonomy YAML (default: object_to_category)",
    )
    args = parser.parse_args()

    # Load taxonomy
    with open(args.taxonomy) as f:
        tax = yaml.safe_load(f)
    obj_to_cat = dict(tax[args.mapping_key])

    # Auto-add category/baseshape subgraph IDs to taxonomy
    # e.g., "category_cup" -> "cup", "baseshape_cube" -> "cube"
    for prefix in ("category_", "baseshape_"):
        for obj_id in list(obj_to_cat.keys()):
            pass  # existing entries fine
        # Infer from taxonomy values
        known_cats = set(obj_to_cat.values())
        for cat in known_cats:
            synth_id = f"{prefix}{cat}"
            if synth_id not in obj_to_cat:
                obj_to_cat[synth_id] = cat

    # Load graphs and compute pairwise overlaps
    print("Loading graphs...")
    graphs = load_graphs(args.checkpoint, lm_index=args.lm_index)
    graph_ids = sorted(graphs.keys())
    print(f"  {len(graphs)} graphs: {graph_ids}")

    print("Computing pairwise overlaps...")
    pairwise = {}
    for ga, gb in combinations(graph_ids, 2):
        overlap = compute_spatial_overlap(graphs[ga], graphs[gb], args.threshold)
        pairwise[(ga, gb)] = overlap
        pairwise[(gb, ga)] = overlap
    for g in graph_ids:
        pairwise[(g, g)] = 1.0

    # Compute per-category stats
    cat_to_objs = defaultdict(list)
    for obj, cat in obj_to_cat.items():
        if obj in graphs:
            cat_to_objs[cat].append(obj)

    intra_overlaps = []
    inter_overlaps = []
    for ga, gb in combinations(graph_ids, 2):
        cat_a = obj_to_cat.get(ga)
        cat_b = obj_to_cat.get(gb)
        if cat_a and cat_b:
            if cat_a == cat_b:
                intra_overlaps.append(pairwise[(ga, gb)])
            else:
                inter_overlaps.append(pairwise[(ga, gb)])

    avg_intra = float(np.mean(intra_overlaps)) if intra_overlaps else 0.0
    avg_inter = float(np.mean(inter_overlaps)) if inter_overlaps else 0.0

    print(f"  Avg intra-category overlap: {avg_intra:.4f}")
    print(f"  Avg inter-category overlap: {avg_inter:.4f}")

    # Load eval stats
    print("Loading eval stats...")
    df = pd.read_csv(args.eval_stats)
    print(f"  {len(df)} rows")

    # Compute distance-aware metrics
    scores = []
    details = []

    for _, row in df.iterrows():
        target = str(row["primary_target_object"])
        predicted = row.get("most_likely_object")

        if pd.isna(predicted):
            scores.append(0.0)
            continue

        predicted = str(predicted)

        if target == predicted:
            scores.append(1.0)
            continue

        target_cat = obj_to_cat.get(target)
        predicted_cat = obj_to_cat.get(predicted)

        if target_cat and predicted_cat and target_cat == predicted_cat:
            # Within-category confusion: partial credit
            # Use overlap between predicted and same-category training objects
            if predicted in graphs:
                same_cat_overlaps = [
                    pairwise.get((predicted, other), avg_intra)
                    for other in cat_to_objs.get(target_cat, [])
                    if other != predicted
                ]
                score = float(np.mean(same_cat_overlaps)) if same_cat_overlaps else avg_intra
            else:
                score = avg_intra
            scores.append(score)
            details.append({
                "target": target,
                "predicted": predicted,
                "category": target_cat,
                "type": "within_category",
                "score": round(score, 4),
            })
        else:
            # Cross-category confusion: minimal credit
            if predicted in graphs and target in graphs:
                score = pairwise.get((predicted, target), avg_inter)
            elif predicted in graphs:
                score = avg_inter
            else:
                score = 0.0
            scores.append(score)
            details.append({
                "target": target,
                "predicted": predicted,
                "target_cat": target_cat,
                "predicted_cat": predicted_cat,
                "type": "cross_category",
                "score": round(score, 4),
            })

    binary_accuracy = float(
        (df["most_likely_object"] == df["primary_target_object"]).mean() * 100
    )
    distance_weighted_accuracy = float(np.mean(scores) * 100)

    # Count confusion types
    n_exact = sum(1 for s in scores if s == 1.0)
    n_within = sum(1 for d in details if d.get("type") == "within_category")
    n_cross = sum(1 for d in details if d.get("type") == "cross_category")
    n_no_pred = len(df) - n_exact - n_within - n_cross

    # Top similar training pairs
    top_pairs = sorted(
        [(ga, gb, pairwise[(ga, gb)]) for ga, gb in combinations(graph_ids, 2)],
        key=lambda x: x[2], reverse=True,
    )

    summary = {
        "binary_accuracy": round(binary_accuracy, 2),
        "distance_weighted_accuracy": round(distance_weighted_accuracy, 2),
        "avg_intra_category_overlap": round(avg_intra, 4),
        "avg_inter_category_overlap": round(avg_inter, 4),
        "n_episodes": len(df),
        "n_exact_match": n_exact,
        "n_within_category_confusion": n_within,
        "n_cross_category_confusion": n_cross,
        "n_no_prediction": n_no_pred,
        "top_similar_pairs": [
            {"a": a, "b": b, "overlap": round(o, 4)} for a, b, o in top_pairs[:10]
        ],
        "confusion_details": details[:30],
    }

    print(f"\n=== Distance-Aware Metrics ===")
    print(f"  Binary accuracy:              {binary_accuracy:.1f}%")
    print(f"  Distance-weighted accuracy:   {distance_weighted_accuracy:.1f}%")
    print(f"  Exact matches:                {n_exact}/{len(df)}")
    print(f"  Within-category confusions:   {n_within}/{len(df)}")
    print(f"  Cross-category confusions:    {n_cross}/{len(df)}")
    print(f"  No predictions:               {n_no_pred}/{len(df)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
