"""Phase 2: Category-aggregated evidence scoring.

Instead of taking argmax over individual graph evidence (instance-level),
aggregates evidence by category and picks the category with highest total
evidence mass. This extracts the category signal that already exists in
the evidence distribution without requiring any new mechanism.

Usage:
    python tools/phase2_category_evidence_readout.py \
        --eval-stats <path_to_eval_stats.csv> \
        --taxonomy <path_to_taxonomy.yaml> \
        --output <output.json> \
        --mapping-key object_to_category
"""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_evidence_dict(s):
    """Parse evidence_per_graph column from CSV (stored as string repr of dict)."""
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return {}


def get_category(obj_id, obj_to_cat):
    """Get category for an object ID, handling category/alphabet prefixes."""
    if obj_id in obj_to_cat:
        return obj_to_cat[obj_id]
    # Handle category_X / alphabet_X prefixes
    for prefix in ("category_", "alphabet_", "baseshape_"):
        if obj_id.startswith(prefix):
            return obj_id[len(prefix):]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-stats", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mapping-key", default="object_to_category",
        help="Key in taxonomy YAML (default: object_to_category)",
    )
    args = parser.parse_args()

    # Load taxonomy
    with open(args.taxonomy) as f:
        tax = yaml.safe_load(f)
    obj_to_cat = dict(tax[args.mapping_key])

    # Auto-add synthetic entries for category/alphabet/baseshape prefixes
    known_cats = set(obj_to_cat.values())
    for prefix in ("category_", "alphabet_", "baseshape_"):
        for cat in known_cats:
            synth_id = f"{prefix}{cat}"
            if synth_id not in obj_to_cat:
                obj_to_cat[synth_id] = cat

    # Load eval stats
    df = pd.read_csv(args.eval_stats)
    if "evidence_per_graph" not in df.columns:
        print("ERROR: eval_stats.csv does not contain evidence_per_graph column.")
        print("Re-run the eval with the updated logging code.")
        return

    print(f"Loaded {len(df)} rows from {args.eval_stats}")

    # Compute category-aggregated evidence for each episode
    results = []
    for idx, row in df.iterrows():
        target = row["primary_target_object"]
        instance_winner = row["most_likely_object"]
        target_cat = get_category(target, obj_to_cat)

        evidence = parse_evidence_dict(row["evidence_per_graph"])
        if not evidence:
            results.append({
                "target": target,
                "target_category": target_cat,
                "instance_prediction": instance_winner,
                "category_prediction": None,
                "instance_correct": target == instance_winner,
                "category_correct": False,
                "category_evidence": {},
            })
            continue

        # Aggregate evidence by category
        cat_evidence = defaultdict(float)
        cat_count = defaultdict(int)
        for graph_id, ev in evidence.items():
            cat = get_category(graph_id, obj_to_cat)
            if cat is not None:
                cat_evidence[cat] += max(0, ev)  # Only count positive evidence
                cat_count[cat] += 1

        # Normalize by number of training objects per category
        cat_evidence_norm = {}
        for cat, ev in cat_evidence.items():
            count = cat_count.get(cat, 1)
            cat_evidence_norm[cat] = ev / count if count > 0 else ev

        # Category prediction = category with highest normalized evidence
        if cat_evidence_norm:
            cat_prediction = max(cat_evidence_norm, key=cat_evidence_norm.get)
        else:
            cat_prediction = None

        instance_cat = get_category(str(instance_winner), obj_to_cat)

        # Novelty detection: compute category evidence margin (normalized)
        margin = 0.0
        rel_margin = 0.0
        is_novel = True
        if len(cat_evidence_norm) >= 2:
            sorted_cats = sorted(
                cat_evidence_norm.items(), key=lambda x: x[1], reverse=True
            )
            best_ev = sorted_cats[0][1]
            second_ev = sorted_cats[1][1]
            margin = best_ev - second_ev
            rel_margin = margin / best_ev if best_ev > 0 else 0.0
            is_novel = rel_margin < 0.3

        results.append({
            "target": target,
            "target_category": target_cat,
            "instance_prediction": str(instance_winner),
            "instance_category": instance_cat,
            "category_prediction": cat_prediction,
            "instance_correct": target == instance_winner,
            "instance_category_correct": target_cat == instance_cat,
            "category_correct": target_cat == cat_prediction,
            "category_evidence": dict(cat_evidence),
            "category_margin": round(margin, 2),
            "category_relative_margin": round(rel_margin, 4),
            "category_novel": is_novel,
        })

    # Compute metrics
    n = len(results)
    instance_exact = sum(1 for r in results if r["instance_correct"])
    instance_cat_correct = sum(
        1 for r in results if r.get("instance_category_correct", False)
    )
    category_agg = sum(1 for r in results if r["category_correct"])

    # Novelty detection: split by confidence using instance prediction
    n_novel = sum(1 for r in results if r.get("category_novel", False))
    n_confident = n - n_novel
    inst_confident_correct = sum(
        1 for r in results
        if not r.get("category_novel", True)
        and r.get("instance_category_correct", False)
    )
    inst_novel_correct = sum(
        1 for r in results
        if r.get("category_novel", False)
        and r.get("instance_category_correct", False)
    )

    # Find optimal threshold via sweep
    thresholds = [i / 20.0 for i in range(1, 19)]
    best_th, best_separation = 0.3, 0.0
    for th in thresholds:
        conf = [r for r in results if r.get("category_relative_margin", 0) >= th]
        novel = [r for r in results if r.get("category_relative_margin", 0) < th]
        if conf and novel:
            conf_acc = sum(
                1 for r in conf if r.get("instance_category_correct", False)
            ) / len(conf)
            novel_acc = sum(
                1 for r in novel if r.get("instance_category_correct", False)
            ) / len(novel)
            sep = conf_acc - novel_acc
            if sep > best_separation:
                best_separation = sep
                best_th = th

    print(f"\n{'='*70}")
    print(f"{'METRIC':<50s} {'COUNT':>6s} {'ACC':>8s}")
    print(f"{'='*70}")
    print(f"{'Instance exact match':<50s} "
          f"{instance_exact:>6d} {100*instance_exact/n:>7.1f}%")
    print(f"{'Instance category (argmax winner category)':<50s} "
          f"{instance_cat_correct:>6d} {100*instance_cat_correct/n:>7.1f}%")
    print(f"{'Category-aggregated (normalized sum)':<50s} "
          f"{category_agg:>6d} {100*category_agg/n:>7.1f}%")
    print(f"{'-'*70}")
    print(f"  NOVELTY DETECTION (threshold={best_th:.2f})")
    if n_confident > 0:
        print(f"{'  Confident (instance cat, high margin)':<50s} "
              f"{n_confident:>6d} "
              f"{100*inst_confident_correct/n_confident:>7.1f}%")
    if n_novel > 0:
        print(f"{'  Novel/uncertain (instance cat, low margin)':<50s} "
              f"{n_novel:>6d} "
              f"{100*inst_novel_correct/n_novel:>7.1f}%")
    print(f"{'='*70}")

    print(f"\nOptimal threshold: {best_th:.2f} "
          f"(separation: {100*best_separation:.1f}pp)")

    # Margin distribution by category
    margins = defaultdict(list)
    for r in results:
        tc = r.get("target_category")
        if tc:
            margins[tc].append(r.get("category_relative_margin", 0))
    if margins:
        print(f"\n{'CATEGORY':<15s} {'AVG MARGIN':>10s} {'INST CAT':>9s} "
              f"{'NOVEL%':>8s} {'N':>4s}")
        print(f"{'-'*50}")
        for cat in sorted(margins.keys()):
            m_list = margins[cat]
            avg_m = np.mean(m_list)
            cat_correct = sum(
                1 for r in results
                if r.get("target_category") == cat
                and r.get("instance_category_correct", False)
            )
            cat_total = sum(
                1 for r in results if r.get("target_category") == cat
            )
            cat_novel_pct = 100 * sum(
                1 for m in m_list if m < best_th
            ) / len(m_list)
            print(f"{cat:<15s} {avg_m:>10.3f} "
                  f"{100*cat_correct/cat_total:>8.1f}% "
                  f"{cat_novel_pct:>7.1f}% "
                  f"{cat_total:>4d}")

    # Save
    summary = {
        "n_episodes": n,
        "instance_exact_accuracy": round(100 * instance_exact / n, 2),
        "instance_category_accuracy": round(100 * instance_cat_correct / n, 2),
        "category_aggregated_accuracy": round(100 * category_agg / n, 2),
        "novelty_threshold": best_th,
        "n_confident": n_confident,
        "n_novel": n_novel,
        "confident_accuracy": round(
            100 * inst_confident_correct / n_confident, 2
        ) if n_confident > 0 else None,
        "novel_accuracy": round(
            100 * inst_novel_correct / n_novel, 2
        ) if n_novel > 0 else None,
        "per_episode": results[:30],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
