#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Comprehensive CorticalColumnTorch benchmark: single-column, voting,
heterarchy, and auto-label.

Runs four benchmark scenarios:
1. Single-column YCB (5 objects x 4 rotations) — baseline
2. Two-LM Hopfield voting (hierarchical, animated model)
3. Three-LM heterarchy (parent + 2 children, animated vs static)
4. Auto-label discrimination (train without names, verify recall)

Usage::

    conda run -n tbp.monty python scripts/run_full_torch_benchmark.py
"""

import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

from tbp.monty.frameworks.models.cortical_column_torch.experiment import (
    Panda3DTorchExperiment,
)
from tbp.monty.simulators.panda3d.cortical_column_torch_evaluation import (
    CorticalColumnTorchEvalHarness,
)

# ── Constants ────────────────────────────────────────────────────────
TESTS_ROOT = Path(__file__).resolve().parents[1] / "tests"
ASSET_DIR = str(
    TESTS_ROOT / "unit" / "simulators" / "panda3d" / "test_assets" / "animated"
)
FOX_PATH = str(Path(ASSET_DIR) / "Fox.glb")

YCB_OBJECTS = [
    "011_banana",
    "025_mug",
    "003_cracker_box",
    "013_apple",
    "035_power_drill",
]
YCB_ROTATIONS = [
    (0.0, 0.0, 0.0),
    (0.0, 45.0, 0.0),
    (30.0, 0.0, 60.0),
    (0.0, 90.0, 0.0),
]

CHILD_COLUMN_KW = {
    "n_minicolumns": 1024,
    "n_cells_per_minicolumn": 8,
    "sparsity": 0.04,
    "beta": 12.0,
    "max_settle_iters": 10,
    "evidence_decay": 0.01,
    "seed": 42,
}


# ── Benchmark 1: Single-column YCB ──────────────────────────────────
def benchmark_single_column():
    """5 YCB objects x 4 rotations, single CorticalColumnTorch."""
    print("\n" + "=" * 60)
    print("Benchmark 1: Single-Column YCB (5 objects x 4 rotations)")
    print("=" * 60)

    harness = CorticalColumnTorchEvalHarness(
        object_names=YCB_OBJECTS,
        eval_rotations=YCB_ROTATIONS,
        train_steps=120,
        eval_steps=60,
        train_episodes=2,
        resolution=(64, 64),
        fov=90.0,
        orbit_radius=0.5,
        column_kwargs={
            "n_minicolumns": 2048,
            "n_cells_per_minicolumn": 8,
            "sparsity": 0.03,
            "beta": 12.0,
            "max_settle_iters": 10,
            "evidence_decay": 0.01,
        },
        sm_features=["on_object", "hsv", "principal_curvatures_log"],
        evidence_threshold=2.0,
        seed=42,
    )
    try:
        result = harness.run()
        print(result.summary())
        for ep in result.episodes:
            rot_str = (f"({ep.rotation[0]:.0f},{ep.rotation[1]:.0f},"
                       f"{ep.rotation[2]:.0f})")
            status = "OK" if ep.correct else f"WRONG({ep.detected_object})"
            print(f"  {ep.object_name:20s} rot={rot_str}: {status} "
                  f"ev={ep.max_evidence:.2f}")
        n_correct = sum(1 for e in result.episodes if e.correct)
        n_episodes = len(result.episodes)
        return {
            "accuracy": result.accuracy,
            "n_correct": n_correct,
            "n_episodes": n_episodes,
            "wall_clock": result.wall_clock_total,
        }
    finally:
        harness.close()


# ── Benchmark 2: Two-LM Hopfield Voting ─────────────────────────────
def benchmark_voting():
    """Two-LM surprise-gated Hopfield voting on animated model."""
    print("\n" + "=" * 60)
    print("Benchmark 2: Two-LM Hopfield Voting (animated)")
    print("=" * 60)

    if not os.path.isfile(FOX_PATH):
        print("  SKIP — Fox.glb not found")
        return {"status": "skipped"}

    exp = Panda3DTorchExperiment(
        model_path=FOX_PATH,
        hierarchical=True,
        resolution=(32, 32),
        initial_distance=2.0,
        object_scale=(0.01, 0.01, 0.01),
        asset_search_paths=[ASSET_DIR],
        column_kwargs=CHILD_COLUMN_KW,
        hopfield_voting=True,
        hopfield_surprise_threshold=0.3,
    )
    try:
        exp._setup()
        anim = list(exp._anim_obj.animation_names)[0]

        t0 = time.monotonic()
        exp.train("fox_walk", anim_name=anim, n_steps=30)
        exp.train("fox_static", n_steps=30)
        train_time = time.monotonic() - t0

        # Eval animated — should detect fox_walk
        t1 = time.monotonic()
        result_walk = exp.evaluate(anim_name=anim, n_steps=20)
        eval_time_walk = time.monotonic() - t1

        # Eval static — should detect fox_static
        result_static = exp.evaluate(n_steps=20)

        walk_det = result_walk.get("graph_id")
        static_det = result_static.get("graph_id")

        morph_ev = result_walk.get("evidence", {}).get("lm_morphology", {})
        behav_ev = result_walk.get("evidence", {}).get("lm_behavior", {})
        parent_ev = result_walk.get("evidence", {}).get("lm_parent", {})

        print(f"  Animated eval: detected={walk_det} "
              f"({'OK' if walk_det == 'fox_walk' else 'WRONG'})")
        print(f"  Static eval:   detected={static_det} "
              f"({'OK' if static_det == 'fox_static' else 'WRONG'})")
        print(f"  LM morphology evidence: {morph_ev}")
        print(f"  LM behavior evidence:   {behav_ev}")
        print(f"  LM parent evidence:     {parent_ev}")
        print(f"  Train: {train_time:.1f}s  Eval: {eval_time_walk:.1f}s")

        children_have_evidence = all(
            len(ev) > 0 for ev in [morph_ev, behav_ev]
        )
        print(f"  Child LMs have evidence: {children_have_evidence}")

        return {
            "walk_detected": walk_det,
            "walk_correct": walk_det == "fox_walk",
            "static_detected": static_det,
            "static_correct": static_det == "fox_static",
            "children_have_evidence": children_have_evidence,
            "morph_evidence": morph_ev,
            "behavior_evidence": behav_ev,
            "train_time": train_time,
            "eval_time": eval_time_walk,
        }
    finally:
        exp.close()


# ── Benchmark 3: Three-LM Heterarchy ────────────────────────────────
def benchmark_heterarchy():
    """3-LM heterarchy: CameraSM + ChangeDetectingSM + parent column.

    Trains Fox animated and Fox static as two distinct objects. Tests that
    the heterarchy correctly routes context between children and parent,
    and that children accumulate evidence through the hierarchy.
    """
    print("\n" + "=" * 60)
    print("Benchmark 3: Three-LM Heterarchy (parent + 2 children)")
    print("=" * 60)

    if not os.path.isfile(FOX_PATH):
        print("  SKIP — Fox.glb not found")
        return {"status": "skipped"}

    exp = Panda3DTorchExperiment(
        model_path=FOX_PATH,
        hierarchical=True,
        resolution=(32, 32),
        initial_distance=2.0,
        object_scale=(0.01, 0.01, 0.01),
        asset_search_paths=[ASSET_DIR],
        column_kwargs=CHILD_COLUMN_KW,
    )
    try:
        exp._setup()
        anim = list(exp._anim_obj.animation_names)[0]

        t0 = time.monotonic()
        exp.train("fox_walk", anim_name=anim, n_steps=30)
        exp.train("fox_static", n_steps=30)
        train_time = time.monotonic() - t0

        # Report what each LM learned
        for lm in exp.monty.learning_modules:
            known = lm.get_all_known_object_ids()
            print(f"  {lm.learning_module_id}: known={known}")

        # Eval animated
        t1 = time.monotonic()
        result_walk = exp.evaluate(anim_name=anim, n_steps=20)
        eval_time = time.monotonic() - t1

        walk_det = result_walk.get("graph_id")
        morph_ev = result_walk.get("evidence", {}).get("lm_morphology", {})
        behav_ev = result_walk.get("evidence", {}).get("lm_behavior", {})
        parent_ev = result_walk.get("evidence", {}).get("lm_parent", {})

        print(f"  Animated eval: detected={walk_det} "
              f"({'OK' if walk_det == 'fox_walk' else 'WRONG'})")
        print(f"    morphology: {morph_ev}")
        print(f"    behavior:   {behav_ev}")
        print(f"    parent:     {parent_ev}")

        # Eval static
        result_static = exp.evaluate(n_steps=20)
        static_det = result_static.get("graph_id")
        morph_ev2 = result_static.get("evidence", {}).get("lm_morphology", {})
        behav_ev2 = result_static.get("evidence", {}).get("lm_behavior", {})

        print(f"  Static eval:   detected={static_det} "
              f"({'OK' if static_det == 'fox_static' else 'WRONG'})")
        print(f"    morphology: {morph_ev2}")
        print(f"    behavior:   {behav_ev2}")
        print(f"  Train: {train_time:.1f}s  Eval: {eval_time:.1f}s")

        children_have_evidence = (len(morph_ev) > 0 and len(behav_ev) > 0)

        return {
            "walk_detected": walk_det,
            "walk_correct": walk_det == "fox_walk",
            "static_detected": static_det,
            "static_correct": static_det == "fox_static",
            "children_have_evidence": children_have_evidence,
            "train_time": train_time,
            "eval_time": eval_time,
        }
    finally:
        exp.close()


# ── Benchmark 4: Auto-Label Discrimination ──────────────────────────
def benchmark_auto_label():
    """Train without external labels on animated and static Fox.

    The column should auto-generate distinct labels for the two visual
    experiences and discriminate them at eval time.
    """
    print("\n" + "=" * 60)
    print("Benchmark 4: Auto-Label Discrimination")
    print("=" * 60)

    if not os.path.isfile(FOX_PATH):
        print("  SKIP — Fox.glb not found")
        return {"status": "skipped"}

    exp = Panda3DTorchExperiment(
        model_path=FOX_PATH,
        hierarchical=False,
        resolution=(32, 32),
        initial_distance=2.0,
        object_scale=(0.01, 0.01, 0.01),
        asset_search_paths=[ASSET_DIR],
        column_kwargs=CHILD_COLUMN_KW,
    )
    try:
        exp._setup()
        anim = list(exp._anim_obj.animation_names)[0]

        # Train animated Fox (no label)
        t0 = time.monotonic()
        exp.train(None, anim_name=anim, n_steps=30)
        # Train static Fox (no label) — different visual input
        exp.train(None, n_steps=30)
        train_time = time.monotonic() - t0

        lm = exp.monty.learning_modules[0]
        known = lm.get_all_known_object_ids()
        print(f"  Auto-generated labels: {known}")
        print(f"  Number of distinct labels: {len(known)}")

        # Eval animated
        t1 = time.monotonic()
        result_anim = exp.evaluate(anim_name=anim, n_steps=20)
        anim_det = result_anim.get("graph_id")
        anim_ev = dict(lm.evidence)
        print(f"  Animated eval: detected={anim_det}, evidence={anim_ev}")

        # Eval static
        result_static = exp.evaluate(n_steps=20)
        static_det = result_static.get("graph_id")
        static_ev = dict(lm.evidence)
        eval_time = time.monotonic() - t1
        print(f"  Static eval:   detected={static_det}, evidence={static_ev}")

        discriminated = len(known) >= 2 and anim_det != static_det
        print(f"  Discrimination: {discriminated}")
        print(f"  Train: {train_time:.1f}s  Eval: {eval_time:.1f}s")

        return {
            "n_auto_labels": len(known),
            "auto_labels": known,
            "anim_detected": anim_det,
            "static_detected": static_det,
            "discriminated": discriminated,
            "train_time": train_time,
            "eval_time": eval_time,
        }
    finally:
        exp.close()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    all_results = {}
    t_total = time.monotonic()

    all_results["single_column_ycb"] = benchmark_single_column()
    all_results["two_lm_voting"] = benchmark_voting()
    all_results["three_lm_heterarchy"] = benchmark_heterarchy()
    all_results["auto_label"] = benchmark_auto_label()

    total_time = time.monotonic() - t_total

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FULL BENCHMARK SUMMARY")
    print("=" * 60)

    ycb = all_results["single_column_ycb"]
    print(f"\n1. Single-Column YCB: {ycb['accuracy']*100:.0f}% "
          f"({ycb['n_correct']}/{ycb['n_episodes']}) "
          f"in {ycb['wall_clock']:.1f}s")

    vt = all_results["two_lm_voting"]
    if "status" not in vt:
        print(f"\n2. Two-LM Voting:")
        print(f"   animated={vt['walk_detected']} "
              f"({'OK' if vt['walk_correct'] else 'WRONG'}), "
              f"static={vt['static_detected']} "
              f"({'OK' if vt['static_correct'] else 'WRONG'})")
        print(f"   children_evidence={vt['children_have_evidence']}")
    else:
        print(f"\n2. Two-LM Voting: {vt['status']}")

    het = all_results["three_lm_heterarchy"]
    if "status" not in het:
        print(f"\n3. Heterarchy:")
        print(f"   animated={het['walk_detected']} "
              f"({'OK' if het['walk_correct'] else 'WRONG'}), "
              f"static={het['static_detected']} "
              f"({'OK' if het['static_correct'] else 'WRONG'})")
        print(f"   children_evidence={het['children_have_evidence']}")
    else:
        print(f"\n3. Heterarchy: {het['status']}")

    auto = all_results["auto_label"]
    if "status" not in auto:
        print(f"\n4. Auto-Label: {auto['n_auto_labels']} labels, "
              f"discriminated={auto['discriminated']}")
    else:
        print(f"\n4. Auto-Label: {auto['status']}")

    print(f"\nTotal benchmark time: {total_time:.1f}s")

    out_path = Path("benchmark_results_full_torch.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
