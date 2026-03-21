"""Phase 2: Inject category subgraphs into a Monty checkpoint.

Takes an existing training checkpoint and a category taxonomy, extracts shared
subgraphs for each category, and creates a new checkpoint where each category
subgraph is registered as an additional matchable graph entry alongside the
original instance graphs.

The category subgraphs use the same graph format as the source checkpoint
(GraphObjectModel or GridObjectModel) so they participate in evidence matching
identically to instance graphs.

Usage:
    python tools/phase2_inject_category_subgraphs.py \
        --checkpoint <path_to_model.pt> \
        --taxonomy <path_to_taxonomy.yaml> \
        --output <path_to_augmented_model.pt> \
        --threshold 0.05
"""

import argparse
import copy
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

from tbp.monty.frameworks.models.object_model import (
    GraphObjectModel,
    GridObjectModel,
)
from tbp.monty.frameworks.utils.object_model_utils import NumpyGraph


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def load_taxonomy(path, mapping_key="object_to_category"):
    import yaml

    with open(path) as f:
        tax = yaml.safe_load(f)
    return dict(tax[mapping_key])


def center_and_scale(pos):
    centered = pos - pos.mean(axis=0)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered, pos.mean(axis=0), scale


def detect_patch_key(g_dict):
    """Detect the patch key used in a graph memory entry."""
    for key in ("patch", "patch_0", "patch_1"):
        if key in g_dict:
            return key
    return next(iter(g_dict.keys()))


def get_model_arrays(model):
    """Extract pos, x, norm as numpy arrays from any model type."""
    pos = np.asarray(model.pos, dtype=np.float64)
    x = np.asarray(model.x, dtype=np.float64)
    try:
        norm = np.asarray(model.norm, dtype=np.float64)
    except (AttributeError, TypeError):
        norm = None
    return pos, x, norm


def extract_shared_subgraph(model_a, model_b, threshold=0.05):
    """Extract nodes from A that have close correspondences in B.

    Alignment is done in centered/scaled space, but the output positions are
    in the original world coordinate space (averaged between A and B).
    This ensures the category subgraph is in the same reference frame as
    instance graphs for evidence matching.
    """
    pos_a, x_a, norm_a = get_model_arrays(model_a)
    pos_b, x_b, norm_b = get_model_arrays(model_b)

    pos_a_c, _, _ = center_and_scale(pos_a)
    pos_b_c, _, _ = center_and_scale(pos_b)

    tree_b = KDTree(pos_b_c)
    dists, indices_b = tree_b.query(pos_a_c)
    mask = dists < threshold

    if mask.sum() == 0:
        return None

    matched_a = np.where(mask)[0]
    matched_b = indices_b[mask]

    # Average positions in ORIGINAL world coordinate space
    avg_pos = (pos_a[matched_a] + pos_b[matched_b]) / 2.0

    # Average features
    avg_x = (x_a[matched_a] + x_b[matched_b]) / 2.0

    # Average normals if available
    if norm_a is not None and norm_b is not None:
        avg_norm = (norm_a[matched_a] + norm_b[matched_b]) / 2.0
        lengths = np.linalg.norm(avg_norm, axis=1, keepdims=True)
        lengths = np.where(lengths > 0, lengths, 1.0)
        avg_norm = avg_norm / lengths
    else:
        avg_norm = None

    return {
        "pos": avg_pos,
        "x": avg_x,
        "norm": avg_norm,
        "n_nodes": int(mask.sum()),
    }


def build_graph_object_model(object_id, pos, x, norm, feature_mapping, k_n=20):
    """Build a GraphObjectModel from raw arrays (torch_geometric Data format)."""
    pos_t = torch.tensor(pos, dtype=torch.float32)
    x_t = torch.tensor(x, dtype=torch.float32)
    norm_t = torch.tensor(norm, dtype=torch.float32) if norm is not None else torch.zeros((pos.shape[0], 3), dtype=torch.float32)

    n = pos_t.shape[0]
    actual_k = min(k_n, n - 1)

    if actual_k > 0 and n > 1:
        adj = kneighbors_graph(pos, n_neighbors=actual_k, mode="connectivity")
        coo = adj.tocoo()
        edge_index = torch.tensor(
            np.stack([coo.row, coo.col]), dtype=torch.long
        )
        edge_attr = pos_t[edge_index[1]] - pos_t[edge_index[0]]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)

    data = Data(
        x=x_t,
        pos=pos_t,
        norm=norm_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        feature_mapping=dict(feature_mapping),
    )

    model = GraphObjectModel(object_id)
    model.set_graph(data)
    model.k_n = actual_k

    return model


def build_grid_object_model(object_id, pos, x, feature_mapping):
    """Build a GridObjectModel from raw arrays (NumpyGraph format).

    Creates a GridObjectModel with use_original_graph=True so the raw graph
    is stored directly without grid quantization.
    """
    numpy_graph = NumpyGraph(dict(
        pos=np.asarray(pos, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        feature_mapping=dict(feature_mapping),
    ))

    # GridObjectModel requires these init params; values don't matter when
    # use_original_graph=True since grids are bypassed.
    model = GridObjectModel(
        object_id,
        max_nodes=pos.shape[0],
        max_size=1.0,
        num_voxels_per_dim=64,
    )
    model.use_original_graph = True
    model.set_graph(numpy_graph)

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--prefix",
        default="category_",
        help="Prefix for category subgraph IDs (default: category_)",
    )
    parser.add_argument(
        "--mapping-key",
        default="object_to_category",
        help="Key in taxonomy YAML mapping objects to groups (default: object_to_category)",
    )
    parser.add_argument(
        "--lm-index",
        type=int,
        default=0,
        help="LM index to inject into (default: 0)",
    )
    args = parser.parse_args()

    print("Loading checkpoint...")
    ckpt = load_checkpoint(args.checkpoint)
    lm_idx = args.lm_index
    gm = ckpt["lm_dict"][lm_idx]["graph_memory"]
    target_to_graph = ckpt["lm_dict"][lm_idx]["target_to_graph_id"]
    graph_to_target = ckpt["lm_dict"][lm_idx]["graph_id_to_target"]

    print(f"  LM index: {lm_idx}")
    print(f"  Existing graphs: {list(gm.keys())}")

    print("Loading taxonomy...")
    obj_to_cat = load_taxonomy(args.taxonomy, mapping_key=args.mapping_key)

    # Detect checkpoint format from first graph
    first_g_dict = next(iter(gm.values()))
    patch_key = detect_patch_key(first_g_dict)
    first_model = first_g_dict[patch_key]
    is_grid_model = isinstance(first_model, GridObjectModel)
    model_type_name = "GridObjectModel" if is_grid_model else "GraphObjectModel"

    # Get feature mapping
    if hasattr(first_model, '_graph') and hasattr(first_model._graph, 'feature_mapping'):
        feature_mapping = dict(first_model._graph.feature_mapping)
    else:
        feature_mapping = dict(first_model.feature_mapping)
    k_n = getattr(first_model, "k_n", 20)

    print(f"  Model type: {model_type_name}")
    print(f"  Patch key: {patch_key}")
    print(f"  Feature mapping: {list(feature_mapping.keys())}")
    print(f"  k_n: {k_n}")

    # Group training objects by category
    cat_to_objs = defaultdict(list)
    for obj, cat in obj_to_cat.items():
        if obj in gm:
            cat_to_objs[cat].append(obj)
    print(f"  Training objects by category: {dict(cat_to_objs)}")

    # Deep copy the checkpoint
    new_ckpt = copy.deepcopy(ckpt)
    new_gm = new_ckpt["lm_dict"][lm_idx]["graph_memory"]
    new_t2g = new_ckpt["lm_dict"][lm_idx]["target_to_graph_id"]
    new_g2t = new_ckpt["lm_dict"][lm_idx]["graph_id_to_target"]

    # Extract and inject category subgraphs
    print(f"\n=== Extracting category subgraphs (threshold={args.threshold}) ===")
    injected = []

    for cat, objs in cat_to_objs.items():
        if len(objs) < 2:
            print(f"  {cat}: only {len(objs)} object, skipping")
            continue

        # Find best pair
        best_sub = None
        best_pair = None
        for oa, ob in combinations(objs, 2):
            sub = extract_shared_subgraph(
                gm[oa][patch_key], gm[ob][patch_key], threshold=args.threshold
            )
            if sub is not None and (best_sub is None or sub["n_nodes"] > best_sub["n_nodes"]):
                best_sub = sub
                best_pair = (oa, ob)

        if best_sub is None or best_sub["n_nodes"] < 10:
            print(f"  {cat}: insufficient shared structure, skipping")
            continue

        cat_id = f"{args.prefix}{cat}"
        print(
            f"  {cat}: {best_pair[0]} + {best_pair[1]} -> {best_sub['n_nodes']} "
            f"shared nodes -> graph id '{cat_id}'"
        )

        # Build the appropriate model type
        if is_grid_model:
            cat_model = build_grid_object_model(
                cat_id,
                best_sub["pos"],
                best_sub["x"],
                feature_mapping,
            )
        else:
            cat_model = build_graph_object_model(
                cat_id,
                best_sub["pos"],
                best_sub["x"],
                best_sub["norm"],
                feature_mapping,
                k_n=k_n,
            )

        # Inject into graph memory using the same patch key
        new_gm[cat_id] = {patch_key: cat_model}

        # Register in target/graph mappings.
        new_t2g[cat_id] = {cat_id}
        new_g2t[cat_id] = {cat_id}
        injected.append(cat_id)

    print(f"\n=== Augmented checkpoint ===")
    print(f"  Original graphs: {len(gm)}")
    print(f"  Injected category subgraphs: {len(injected)} ({injected})")
    print(f"  Total graphs: {len(new_gm)}")
    print(f"  All graph IDs: {list(new_gm.keys())}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"\n  Saved augmented checkpoint to {output_path}")


if __name__ == "__main__":
    main()
