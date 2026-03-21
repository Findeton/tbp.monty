"""Phase 2 Stage 3: Category Subgraph Extraction Prototype.

Given a checkpoint with multiple learned object graphs and a category taxonomy,
extracts the shared spatial substructure for each category. The shared subgraph
consists of nodes from graph A that have close spatial correspondences in graph B
(within the same category), along with their averaged features and normals.

This is the first concrete test of whether graph-structural category representations
can be built from existing Monty object memories.

Usage:
    python tools/phase2_category_subgraph_extraction.py \
        --checkpoint <path_to_model.pt> \
        --taxonomy <path_to_taxonomy.yaml> \
        --output <output_directory> \
        --threshold 0.05
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import KDTree


def load_graphs(checkpoint_path, lm_index=0):
    """Load graph memories from a Monty checkpoint.

    Auto-detects graph format (GraphObjectModel with 'patch' key or
    GridObjectModel with 'patch_0' key). Handles missing normals gracefully.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    gm = ckpt["lm_dict"][lm_index]["graph_memory"]
    graphs = {}
    for gid, g_dict in gm.items():
        # Auto-detect patch key
        if "patch" in g_dict:
            g = g_dict["patch"]
        elif "patch_0" in g_dict:
            g = g_dict["patch_0"]
        else:
            # Use first available key
            g = next(iter(g_dict.values()))

        # Extract pos/x as numpy arrays (works for both torch and numpy)
        pos = np.asarray(g.pos, dtype=np.float64)
        x = np.asarray(g.x, dtype=np.float64)

        # Handle optional normals
        try:
            norm = np.asarray(g.norm, dtype=np.float64)
        except (AttributeError, TypeError):
            norm = None

        graphs[gid] = {
            "pos": pos,
            "x": x,
            "norm": norm,
            "n_nodes": pos.shape[0],
        }
    return graphs


def load_taxonomy(taxonomy_path, mapping_key="object_to_category"):
    """Load category taxonomy and return object-to-category mapping."""
    with open(taxonomy_path) as f:
        tax = yaml.safe_load(f)
    return dict(tax[mapping_key])


def center_and_scale(pos):
    """Center point cloud at origin and normalize to unit scale."""
    centered = pos - pos.mean(axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered, pos.mean(axis=0), scale


def extract_shared_subgraph(graph_a, graph_b, threshold=0.05):
    """Extract nodes from A that have close correspondences in B.

    Both graphs are centered/scaled to unit space for alignment. For each node
    in A with a neighbor in B within threshold, the shared subgraph keeps:
    - the midpoint position (average of A and B positions in original space)
    - the averaged feature vector
    - the averaged surface normal

    Returns:
        dict with keys: pos, x, norm, n_nodes, correspondence_distances,
        fraction_a_covered, fraction_b_covered
    """
    pos_a_raw = graph_a["pos"]
    pos_b_raw = graph_b["pos"]

    pos_a_centered, center_a, scale_a = center_and_scale(pos_a_raw)
    pos_b_centered, center_b, scale_b = center_and_scale(pos_b_raw)

    tree_b = KDTree(pos_b_centered)
    dists, indices_b = tree_b.query(pos_a_centered)

    # Nodes in A that have a close match in B
    mask = dists < threshold

    if mask.sum() == 0:
        return {
            "pos": np.zeros((0, 3)),
            "x": np.zeros((0, graph_a["x"].shape[1])),
            "norm": np.zeros((0, 3)),
            "n_nodes": 0,
            "correspondence_distances": np.array([]),
            "fraction_a_covered": 0.0,
            "fraction_b_covered": 0.0,
        }

    matched_a_idx = np.where(mask)[0]
    matched_b_idx = indices_b[mask]

    # Average features from both graphs at matched positions
    avg_x = (graph_a["x"][matched_a_idx] + graph_b["x"][matched_b_idx]) / 2.0

    # Average normals if available
    if graph_a["norm"] is not None and graph_b["norm"] is not None:
        avg_norm = (graph_a["norm"][matched_a_idx] + graph_b["norm"][matched_b_idx]) / 2.0
        norm_lengths = np.linalg.norm(avg_norm, axis=1, keepdims=True)
        norm_lengths = np.where(norm_lengths > 0, norm_lengths, 1.0)
        avg_norm = avg_norm / norm_lengths
    else:
        avg_norm = None

    # Position: average of matched positions in centered/scaled space
    avg_pos = (pos_a_centered[matched_a_idx] + pos_b_centered[matched_b_idx]) / 2.0

    # Check reverse coverage (how many B nodes have a match in A)
    tree_a = KDTree(pos_a_centered)
    dists_ba, _ = tree_a.query(pos_b_centered)
    frac_b_covered = float(np.mean(dists_ba < threshold))

    return {
        "pos": avg_pos,
        "x": avg_x,
        "norm": avg_norm,
        "n_nodes": int(mask.sum()),
        "correspondence_distances": dists[mask],
        "fraction_a_covered": float(mask.mean()),
        "fraction_b_covered": frac_b_covered,
    }


def match_against_subgraph(subgraph, query_graph, threshold=0.05):
    """Score how well a query graph matches a category subgraph.

    Returns:
        dict with spatial_overlap, feature_similarity, normal_alignment
    """
    if subgraph["n_nodes"] == 0:
        return {
            "spatial_overlap": 0.0,
            "feature_similarity": 0.0,
            "normal_alignment": 0.0,
        }

    query_pos, _, _ = center_and_scale(query_graph["pos"])
    sub_pos = subgraph["pos"]  # already centered/scaled

    tree_query = KDTree(query_pos)
    dists, indices = tree_query.query(sub_pos)
    mask = dists < threshold

    spatial_overlap = float(mask.mean())

    if mask.sum() == 0:
        return {
            "spatial_overlap": spatial_overlap,
            "feature_similarity": 0.0,
            "normal_alignment": 0.0,
        }

    # Feature similarity at matched positions
    matched_sub_x = subgraph["x"][mask]
    matched_query_x = query_graph["x"][indices[mask]]
    dot = np.sum(matched_sub_x * matched_query_x, axis=1)
    norm_s = np.linalg.norm(matched_sub_x, axis=1)
    norm_q = np.linalg.norm(matched_query_x, axis=1)
    denom = norm_s * norm_q
    denom = np.where(denom > 0, denom, 1.0)
    feature_sim = float(np.mean(dot / denom))

    # Normal alignment at matched positions (if normals available)
    if subgraph["norm"] is not None and query_graph["norm"] is not None:
        matched_sub_norm = subgraph["norm"][mask]
        matched_query_norm = query_graph["norm"][indices[mask]]
        normal_dot = np.sum(matched_sub_norm * matched_query_norm, axis=1)
        normal_align = float(np.mean(np.abs(normal_dot)))
    else:
        normal_align = float("nan")

    return {
        "spatial_overlap": spatial_overlap,
        "feature_similarity": feature_sim,
        "normal_alignment": normal_align,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--mapping-key",
        default="object_to_category",
        help="Key in taxonomy YAML mapping objects to groups (default: object_to_category)",
    )
    parser.add_argument(
        "--lm-index",
        type=int,
        default=0,
        help="LM index to extract graphs from (default: 0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graphs...")
    graphs = load_graphs(args.checkpoint, lm_index=args.lm_index)
    print(f"  Loaded {len(graphs)} graphs: {list(graphs.keys())}")

    print("Loading taxonomy...")
    obj_to_cat = load_taxonomy(args.taxonomy, mapping_key=args.mapping_key)
    print(f"  Categories: {obj_to_cat}")

    # Group objects by category
    cat_to_objs = defaultdict(list)
    for obj, cat in obj_to_cat.items():
        if obj in graphs:
            cat_to_objs[cat].append(obj)
    print(f"  Training objects by category: {dict(cat_to_objs)}")

    # Extract shared subgraphs per category
    print(f"\n=== Extracting shared subgraphs (threshold={args.threshold}) ===")
    category_subgraphs = {}
    extraction_info = {}

    for cat, objs in cat_to_objs.items():
        if len(objs) < 2:
            print(f"  {cat}: only {len(objs)} object, skipping extraction")
            continue

        # For each pair, extract shared subgraph and then intersect
        pair_subgraphs = []
        for oa, ob in combinations(objs, 2):
            sub = extract_shared_subgraph(
                graphs[oa], graphs[ob], threshold=args.threshold
            )
            pair_subgraphs.append((oa, ob, sub))
            print(
                f"  {cat}: {oa} vs {ob} -> {sub['n_nodes']} shared nodes "
                f"(coverage: {sub['fraction_a_covered']:.1%} / "
                f"{sub['fraction_b_covered']:.1%})"
            )

        # Use the pair with the most shared nodes as the category prototype
        best_pair = max(pair_subgraphs, key=lambda x: x[2]["n_nodes"])
        category_subgraphs[cat] = best_pair[2]
        extraction_info[cat] = {
            "best_pair": f"{best_pair[0]} vs {best_pair[1]}",
            "n_shared_nodes": best_pair[2]["n_nodes"],
            "fraction_a_covered": best_pair[2]["fraction_a_covered"],
            "fraction_b_covered": best_pair[2]["fraction_b_covered"],
            "mean_correspondence_distance": float(
                np.mean(best_pair[2]["correspondence_distances"])
            )
            if len(best_pair[2]["correspondence_distances"]) > 0
            else 0.0,
        }

    # Now test: does each training object match its own category subgraph
    # better than other category subgraphs?
    print(f"\n=== Matching training objects against category subgraphs ===")
    match_results = []
    for obj_id in sorted(graphs.keys()):
        obj_cat = obj_to_cat.get(obj_id, "unknown")
        row = {"object": obj_id, "true_category": obj_cat}
        scores = {}
        for cat, sub in category_subgraphs.items():
            score = match_against_subgraph(sub, graphs[obj_id], threshold=args.threshold)
            scores[cat] = score
            row[f"{cat}_spatial"] = score["spatial_overlap"]
            row[f"{cat}_feature"] = score["feature_similarity"]
            row[f"{cat}_normal"] = score["normal_alignment"]

        # Which category subgraph matches best?
        best_cat = max(scores, key=lambda c: scores[c]["spatial_overlap"])
        row["predicted_category"] = best_cat
        row["correct"] = best_cat == obj_cat
        match_results.append(row)

        correct_tag = "OK" if row["correct"] else "WRONG"
        print(
            f"  {obj_id:15s} (true={obj_cat:8s}): "
            f"best match={best_cat:8s} [{correct_tag}]  "
            + "  ".join(
                f"{c}={scores[c]['spatial_overlap']:.3f}"
                for c in sorted(scores.keys())
            )
        )

    n_correct = sum(1 for r in match_results if r["correct"])
    print(
        f"\nTraining accuracy: {n_correct}/{len(match_results)} "
        f"({100 * n_correct / len(match_results):.1f}%)"
    )

    # Save results
    summary = {
        "threshold": args.threshold,
        "extraction_info": extraction_info,
        "match_results": match_results,
        "training_accuracy": n_correct / len(match_results),
    }
    with open(output_dir / "category_subgraph_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
