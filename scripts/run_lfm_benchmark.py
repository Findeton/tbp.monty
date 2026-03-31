#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
# MIT License

"""Compare LocationFeatureMemory vs cortical attractor evidence on YCB.

Runs the same 5-object YCB benchmark with both evidence modes:
  1. Cortical attractors (Phase 10 default): EMA prototypes + softmax recall
  2. Location-feature memory (Phase 11): many patterns + attention-weighted evidence

Usage::

    conda run -n tbp.monty python scripts/run_lfm_benchmark.py
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from tbp.monty.simulators.panda3d.cortical_column_torch_evaluation import (
    CorticalColumnTorchEvalHarness,
)

OBJECTS = [
    "011_banana",
    "025_mug",
    "003_cracker_box",
    "013_apple",
    "035_power_drill",
]

ROTATIONS = [
    (0.0, 0.0, 0.0),
    (0.0, 45.0, 0.0),
    (30.0, 0.0, 60.0),
    (0.0, 90.0, 0.0),
]

BASE_COLUMN_KWARGS = {
    "n_minicolumns": 2048,
    "n_cells_per_minicolumn": 8,
    "sparsity": 0.03,
    "beta": 12.0,
    "max_settle_iters": 10,
    "evidence_decay": 0.01,
}

HARNESS_KWARGS = dict(
    object_names=OBJECTS,
    eval_rotations=ROTATIONS,
    train_steps=120,
    eval_steps=60,
    train_episodes=2,
    resolution=(64, 64),
    fov=90.0,
    orbit_radius=0.5,
    sm_features=["on_object", "hsv", "principal_curvatures_log"],
    evidence_threshold=2.0,
    seed=42,
)


def run_mode(label, column_kwargs):
    print(f"\n{'=' * 60}")
    print(f"Running: {label}")
    print(f"{'=' * 60}")

    harness = CorticalColumnTorchEvalHarness(
        column_kwargs=column_kwargs,
        **HARNESS_KWARGS,
    )

    try:
        t0 = time.time()
        result = harness.run()
        elapsed = time.time() - t0

        print(f"\n{label}: {result.summary()} ({elapsed:.1f}s)")

        obj_results = {}
        for ep in result.episodes:
            obj_results.setdefault(ep.object_name, []).append(ep)

        for obj_name, eps in sorted(obj_results.items()):
            n_correct = sum(1 for e in eps if e.correct)
            print(f"  {obj_name}: {n_correct}/{len(eps)}")
            for e in eps:
                rot_str = (
                    f"({e.rotation[0]:.0f},{e.rotation[1]:.0f},{e.rotation[2]:.0f})"
                )
                status = "OK" if e.correct else f"WRONG({e.detected_object})"
                print(f"    rot={rot_str}: {status} ev={e.max_evidence:.2f}")

        return result
    finally:
        harness.close()


def main():
    # Mode 1: Cortical attractors (baseline)
    baseline_kw = dict(BASE_COLUMN_KWARGS)
    baseline_result = run_mode("Cortical Attractors (Phase 10)", baseline_kw)

    # Mode 2: Location-feature memory
    lfm_kw = dict(BASE_COLUMN_KWARGS)
    lfm_kw["use_location_feature_memory"] = True
    lfm_result = run_mode("Location-Feature Memory (Phase 11)", lfm_kw)

    # Summary comparison
    b_correct = sum(1 for e in baseline_result.episodes if e.correct)
    l_correct = sum(1 for e in lfm_result.episodes if e.correct)
    b_total = len(baseline_result.episodes)
    l_total = len(lfm_result.episodes)

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Cortical Attractors: {b_correct}/{b_total} "
          f"({100 * b_correct / b_total:.0f}%)")
    print(f"  Location-Feature Mem: {l_correct}/{l_total} "
          f"({100 * l_correct / l_total:.0f}%)")


if __name__ == "__main__":
    main()
