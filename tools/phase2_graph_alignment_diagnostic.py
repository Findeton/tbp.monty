"""Phase 2 Assumption C1: Graph Structural Alignment Diagnostic.

Computes pairwise structural similarity between all stored graph memories
in a checkpoint. Tests whether intra-category graph pairs are more similar
than inter-category pairs, which would confirm that graph structure carries
category-level information.

Three complementary similarity measures are used:
  1. Spatial overlap: fraction of nodes in graph A that have a nearest neighbor
     in graph B within a distance threshold (and vice versa).
  2. Feature similarity at corresponding nodes: mean cosine similarity of
     feature vectors at spatially matched node pairs.
  3. Surface normal alignment: mean dot product of surface normals at
     spatially matched node pairs.

Usage:
    python tools/phase2_graph_alignment_diagnostic.py \
        --checkpoint <path_to_model.pt> \
        --taxonomy <path_to_taxonomy.yaml> \
        --output <output_directory>
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import KDTree


def load_graphs(checkpoint_path):
    """Load graph memories from a Monty checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gm = ckpt["lm_dict"][0]["graph_memory"]
    graphs = {}
    for gid, g_dict in gm.items():
        g = g_dict["patch"]
        graphs[gid] = {
            "pos": g.pos.numpy(),
            "x": g.x.numpy(),
            "norm": g.norm.numpy(),
            "n_nodes": g.pos.shape[0],
        }
    return graphs


def load_taxonomy(taxonomy_path):
    """Load category taxonomy and return object-to-category mapping."""
    with open(taxonomy_path) as f:
        tax = yaml.safe_load(f)
    return dict(tax["object_to_category"])


def center_and_scale(pos):
    """Center point cloud at origin and normalize to unit scale."""
    centered = pos - pos.mean(axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered


def compute_spatial_overlap(pos_a, pos_b, threshold=0.05):
    """Fraction of nodes in A with a neighbor in B within threshold (and reverse).

    Both point clouds are centered and scaled to unit before comparison.
    """
    a = center_and_scale(pos_a)
    b = center_and_scale(pos_b)

    tree_b = KDTree(b)
    dists_a_to_b, _ = tree_b.query(a)
    frac_a_covered = np.mean(dists_a_to_b < threshold)

    tree_a = KDTree(a)
    dists_b_to_a, _ = tree_a.query(b)
    frac_b_covered = np.mean(dists_b_to_a < threshold)

    return (frac_a_covered + frac_b_covered) / 2.0


def compute_feature_similarity(pos_a, x_a, pos_b, x_b):
    """Mean cosine similarity of features at spatially matched node pairs.

    For each node in the smaller graph, find its nearest spatial neighbor in the
    larger graph (after centering/scaling), then compute cosine similarity of
    their feature vectors.
    """
    a = center_and_scale(pos_a)
    b = center_and_scale(pos_b)

    tree_b = KDTree(b)
    _, indices = tree_b.query(a)

    matched_b = x_b[indices]

    # Cosine similarity per pair
    dot = np.sum(x_a * matched_b, axis=1)
    norm_a = np.linalg.norm(x_a, axis=1)
    norm_b = np.linalg.norm(matched_b, axis=1)
    denom = norm_a * norm_b
    denom = np.where(denom > 0, denom, 1.0)
    cos_sim = dot / denom

    return float(np.mean(cos_sim))


def compute_normal_alignment(pos_a, norm_a, pos_b, norm_b):
    """Mean absolute dot product of normals at spatially matched node pairs."""
    a = center_and_scale(pos_a)
    b = center_and_scale(pos_b)

    tree_b = KDTree(b)
    _, indices = tree_b.query(a)

    matched_norm_b = norm_b[indices]
    dot = np.sum(norm_a * matched_norm_b, axis=1)

    return float(np.mean(np.abs(dot)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Spatial overlap distance threshold (after unit normalization)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graphs...")
    graphs = load_graphs(args.checkpoint)
    print(f"  Loaded {len(graphs)} graphs: {list(graphs.keys())}")

    print("Loading taxonomy...")
    obj_to_cat = load_taxonomy(args.taxonomy)
    print(f"  Categories: {obj_to_cat}")

    graph_ids = sorted(graphs.keys())
    pairs = list(combinations(graph_ids, 2))
    print(f"\nComputing pairwise alignment for {len(pairs)} pairs...")

    results = []
    for ga_id, gb_id in pairs:
        ga = graphs[ga_id]
        gb = graphs[gb_id]

        spatial = compute_spatial_overlap(
            ga["pos"], gb["pos"], threshold=args.threshold
        )
        feature = compute_feature_similarity(
            ga["pos"], ga["x"], gb["pos"], gb["x"]
        )
        normal = compute_normal_alignment(
            ga["pos"], ga["norm"], gb["pos"], gb["norm"]
        )

        cat_a = obj_to_cat.get(ga_id, "unknown")
        cat_b = obj_to_cat.get(gb_id, "unknown")
        same_cat = cat_a == cat_b

        row = {
            "graph_a": ga_id,
            "graph_b": gb_id,
            "category_a": cat_a,
            "category_b": cat_b,
            "same_category": same_cat,
            "spatial_overlap": round(spatial, 4),
            "feature_similarity": round(feature, 4),
            "normal_alignment": round(normal, 4),
            "nodes_a": ga["n_nodes"],
            "nodes_b": gb["n_nodes"],
        }
        results.append(row)
        tag = "SAME" if same_cat else "DIFF"
        print(
            f"  [{tag}] {ga_id:15s} vs {gb_id:15s}: "
            f"spatial={spatial:.4f}  feature={feature:.4f}  normal={normal:.4f}"
        )

    # Separate intra vs inter category
    intra = [r for r in results if r["same_category"]]
    inter = [r for r in results if not r["same_category"]]

    def stats(rows, key):
        vals = [r[key] for r in rows]
        if not vals:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "n": len(vals),
        }

    summary = {
        "threshold": args.threshold,
        "n_graphs": len(graphs),
        "n_pairs": len(pairs),
        "n_intra_category": len(intra),
        "n_inter_category": len(inter),
        "intra_category": {
            "spatial_overlap": stats(intra, "spatial_overlap"),
            "feature_similarity": stats(intra, "feature_similarity"),
            "normal_alignment": stats(intra, "normal_alignment"),
        },
        "inter_category": {
            "spatial_overlap": stats(inter, "spatial_overlap"),
            "feature_similarity": stats(inter, "feature_similarity"),
            "normal_alignment": stats(inter, "normal_alignment"),
        },
    }

    # Compute separation (effect size)
    for metric in ["spatial_overlap", "feature_similarity", "normal_alignment"]:
        intra_mean = summary["intra_category"][metric]["mean"]
        inter_mean = summary["inter_category"][metric]["mean"]
        intra_std = summary["intra_category"][metric]["std"]
        inter_std = summary["inter_category"][metric]["std"]
        pooled_std = np.sqrt(
            (intra_std ** 2 + inter_std ** 2) / 2.0
        ) if (intra_std + inter_std) > 0 else 1.0
        cohens_d = (intra_mean - inter_mean) / pooled_std if pooled_std > 0 else 0.0
        summary[f"{metric}_separation_cohens_d"] = round(float(cohens_d), 4)

    print("\n=== SUMMARY ===")
    print(f"Intra-category pairs: {len(intra)}")
    print(f"Inter-category pairs: {len(inter)}")
    for metric in ["spatial_overlap", "feature_similarity", "normal_alignment"]:
        im = summary["intra_category"][metric]["mean"]
        em = summary["inter_category"][metric]["mean"]
        cd = summary[f"{metric}_separation_cohens_d"]
        print(f"  {metric}:")
        print(f"    intra-category mean: {im:.4f}")
        print(f"    inter-category mean: {em:.4f}")
        print(f"    Cohen's d (intra - inter): {cd:.4f}")

    # Verdict
    spatial_d = summary["spatial_overlap_separation_cohens_d"]
    feature_d = summary["feature_similarity_separation_cohens_d"]
    normal_d = summary["normal_alignment_separation_cohens_d"]
    any_strong = any(d > 0.8 for d in [spatial_d, feature_d, normal_d])
    any_moderate = any(d > 0.5 for d in [spatial_d, feature_d, normal_d])

    if any_strong:
        verdict = "PASS: Strong category separation in graph structure (d > 0.8)"
    elif any_moderate:
        verdict = "PARTIAL: Moderate category separation (0.5 < d < 0.8)"
    else:
        verdict = "FAIL: No meaningful category separation in graph structure (d < 0.5)"
    summary["verdict"] = verdict
    print(f"\nVerdict: {verdict}")

    # Save
    with open(output_dir / "pairwise_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
