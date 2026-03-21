"""Phase 2: Split a checkpoint into multiple LM checkpoints for voting.

Creates a multi-LM checkpoint where each LM gets a different subset of
instance graphs PLUS all category subgraphs. This enables lateral voting
through shared category-level graph IDs.

The voting mechanism: when LM_A has eliminated "category_utensil" and
"category_box" from its possible matches, it sends these as negative votes
to LM_B, which can then also exclude them. Since all LMs share the
category subgraph IDs, the votes are meaningful across LMs.

Usage:
    python tools/phase2_split_checkpoint_for_voting.py \
        --checkpoint <augmented_model.pt> \
        --taxonomy <taxonomy.yaml> \
        --output <multi_lm_model.pt> \
        --n-lms 2
"""

import argparse
import copy
from collections import defaultdict
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-lms", type=int, default=2)
    args = parser.parse_args()

    print("Loading augmented checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    gm = ckpt["lm_dict"][0]["graph_memory"]
    t2g = ckpt["lm_dict"][0]["target_to_graph_id"]
    g2t = ckpt["lm_dict"][0]["graph_id_to_target"]

    with open(args.taxonomy) as f:
        tax = yaml.safe_load(f)
    obj_to_cat = dict(tax["object_to_category"])

    # Separate instance graphs from category graphs
    instance_ids = [gid for gid in gm if not gid.startswith("category_")]
    category_ids = [gid for gid in gm if gid.startswith("category_")]

    print(f"Instance graphs: {instance_ids}")
    print(f"Category graphs: {category_ids}")

    # Group instances by category
    cat_to_instances = defaultdict(list)
    for gid in instance_ids:
        cat = obj_to_cat.get(gid, "unknown")
        cat_to_instances[cat].append(gid)

    print(f"Instances by category: {dict(cat_to_instances)}")

    # Split instances across LMs: each LM gets one instance per category
    # With 2 LMs and 2 instances per category, each LM gets exactly one
    n_lms = args.n_lms
    lm_instances = [[] for _ in range(n_lms)]

    for cat, instances in cat_to_instances.items():
        for i, inst in enumerate(instances):
            lm_idx = i % n_lms
            lm_instances[lm_idx].append(inst)

    for i in range(n_lms):
        print(f"LM {i} instances: {lm_instances[i]} + all category subgraphs")

    # Build multi-LM checkpoint
    new_ckpt = copy.deepcopy(ckpt)
    new_ckpt["lm_dict"] = {}

    for lm_idx in range(n_lms):
        lm_graph_ids = lm_instances[lm_idx] + category_ids
        lm_gm = {gid: copy.deepcopy(gm[gid]) for gid in lm_graph_ids}
        lm_t2g = {gid: {gid} for gid in lm_graph_ids}
        lm_g2t = {gid: {gid} for gid in lm_graph_ids}

        new_ckpt["lm_dict"][lm_idx] = {
            "graph_memory": lm_gm,
            "target_to_graph_id": lm_t2g,
            "graph_id_to_target": lm_g2t,
        }

    # Set up connectivity matrices
    # sm_to_lm_matrix: both LMs receive from the same SM (index 0)
    new_ckpt["sm_to_lm_matrix"] = [[0] for _ in range(n_lms)]

    # lm_to_lm_matrix: no hierarchical connections
    new_ckpt["lm_to_lm_matrix"] = None

    # lm_to_lm_vote_matrix: all LMs vote with each other
    vote_matrix = []
    for i in range(n_lms):
        vote_matrix.append([j for j in range(n_lms) if j != i])
    new_ckpt["lm_to_lm_vote_matrix"] = vote_matrix

    print(f"\nConnectivity:")
    print(f"  sm_to_lm_matrix: {new_ckpt['sm_to_lm_matrix']}")
    print(f"  lm_to_lm_vote_matrix: {vote_matrix}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"\nSaved multi-LM checkpoint to {output_path}")
    print(f"  {n_lms} LMs, each with {len(lm_instances[0])} instances + "
          f"{len(category_ids)} category subgraphs")


if __name__ == "__main__":
    main()
