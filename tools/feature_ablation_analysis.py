"""Feature ablation analysis for Extended YCB benchmark.

Compares category accuracy across ablation variants:
  - baseline (HSV + pose + curvature)
  - no_hsv (pose + curvature only)
  - no_curvature (HSV + pose only)
  - pose_only (pose vectors only)
  - catbias (baseline + category bias enabled)

Decomposes into shape-congruent vs shape-incongruent categories.

Usage:
    python tools/feature_ablation_analysis.py
"""

import ast
import csv
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path.home() / "tbp/results/monty/projects/phase2_review_runs"

TAXONOMY = {
    "011_banana": "fruit", "012_strawberry": "fruit", "013_apple": "fruit",
    "014_lemon": "fruit", "015_peach": "fruit", "016_pear": "fruit",
    "017_orange": "fruit", "018_plum": "fruit",
    "053_mini_soccer_ball": "ball", "054_softball": "ball",
    "055_baseball": "ball", "056_tennis_ball": "ball",
    "057_racquetball": "ball", "058_golf_ball": "ball",
    "003_cracker_box": "box", "004_sugar_box": "box",
    "008_pudding_box": "box", "009_gelatin_box": "box", "036_wood_block": "box",
    "002_master_chef_can": "can", "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can", "010_potted_meat_can": "can",
    "025_mug": "cup", "065-a_cups": "cup", "065-b_cups": "cup",
    "065-c_cups": "cup", "065-d_cups": "cup", "065-e_cups": "cup",
    "065-f_cups": "cup", "065-g_cups": "cup", "065-h_cups": "cup",
    "065-i_cups": "cup",
    "030_fork": "utensil", "031_spoon": "utensil", "032_knife": "utensil",
    "033_spatula": "utensil",
    "035_power_drill": "tool", "037_scissors": "tool",
    "042_adjustable_wrench": "tool", "043_phillips_screwdriver": "tool",
    "044_flat_screwdriver": "tool",
    "050_medium_clamp": "clamp", "051_large_clamp": "clamp",
    "052_extra_large_clamp": "clamp",
    "072-a_toy_airplane": "airplane", "072-b_toy_airplane": "airplane",
    "072-c_toy_airplane": "airplane", "072-d_toy_airplane": "airplane",
    "072-e_toy_airplane": "airplane",
}

SHAPE_CONGRUENT = {"airplane", "fruit", "cup", "utensil", "box", "clamp"}
SHAPE_INCONGRUENT = {"ball", "tool", "can"}

VARIANTS = {
    "baseline": "extended_ycb_eval_holdout",
    "no_hsv": "extended_ycb_eval_holdout_no_hsv",
    "no_curvature": "extended_ycb_eval_holdout_no_curvature",
    "pose_only": "extended_ycb_eval_holdout_pose_only",
    "catbias": "extended_ycb_eval_holdout_catbias",
}


def analyze_variant(name, run_name):
    csv_path = RESULTS_DIR / run_name / "eval_stats.csv"
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    for row in rows:
        target = row["primary_target_object"]
        predicted = row["most_likely_object"]
        true_cat = TAXONOMY.get(target)
        pred_cat = TAXONOMY.get(predicted)
        if true_cat:
            cat_total[true_cat] += 1
            if true_cat == pred_cat:
                cat_correct[true_cat] += 1

    # Compute splits
    sc_correct = sum(cat_correct[c] for c in SHAPE_CONGRUENT if c in cat_total)
    sc_total = sum(cat_total[c] for c in SHAPE_CONGRUENT if c in cat_total)
    si_correct = sum(cat_correct[c] for c in SHAPE_INCONGRUENT if c in cat_total)
    si_total = sum(cat_total[c] for c in SHAPE_INCONGRUENT if c in cat_total)
    total_correct = sc_correct + si_correct
    total_all = sc_total + si_total

    return {
        "name": name,
        "total": f"{100 * total_correct / total_all:.1f}%" if total_all else "N/A",
        "shape_congruent": f"{100 * sc_correct / sc_total:.1f}%" if sc_total else "N/A",
        "shape_incongruent": f"{100 * si_correct / si_total:.1f}%" if si_total else "N/A",
        "per_category": {
            c: f"{100 * cat_correct[c] / cat_total[c]:.1f}%"
            for c in sorted(cat_total.keys())
        },
        "n_episodes": total_all,
    }


def main():
    print(f"{'VARIANT':<18s} {'OVERALL':>8s} {'SHAPE-CONG':>11s} {'SHAPE-INC':>10s}")
    print(f"{'-' * 50}")

    results = {}
    for name, run_name in VARIANTS.items():
        r = analyze_variant(name, run_name)
        if r:
            results[name] = r
            print(f"{name:<18s} {r['total']:>8s} {r['shape_congruent']:>11s} "
                  f"{r['shape_incongruent']:>10s}")
        else:
            print(f"{name:<18s} {'(no data)':>8s}")

    if len(results) >= 2:
        print(f"\n{'CATEGORY':<15s}", end="")
        for name in results:
            print(f" {name:>12s}", end="")
        print()
        print("-" * (15 + 13 * len(results)))

        all_cats = sorted(
            set().union(*(r["per_category"].keys() for r in results.values()))
        )
        for cat in all_cats:
            regime = "SC" if cat in SHAPE_CONGRUENT else "SI"
            print(f"{cat:<12s}({regime})", end="")
            for name in results:
                val = results[name]["per_category"].get(cat, "N/A")
                print(f" {val:>12s}", end="")
            print()


if __name__ == "__main__":
    main()
