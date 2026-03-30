#!/usr/bin/env python
# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Run CorticalColumnTorch YCB benchmark and save results.

Trains on 5 YCB objects at canonical rotation, then evaluates at 4 rotations
each (20 episodes total). Identical protocol to the SDR CorticalColumn
benchmark in benchmark_results.json.

Usage::

    conda run -n tbp.monty python scripts/run_torch_benchmark.py
"""

import json
import logging
import sys
from pathlib import Path

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


def main():
    harness = CorticalColumnTorchEvalHarness(
        object_names=OBJECTS,
        eval_rotations=ROTATIONS,
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

        print("\n" + "=" * 60)
        print("CorticalColumnTorch Benchmark Results")
        print("=" * 60)
        print(result.summary())
        print()

        # Per-object breakdown
        obj_results = {}
        for ep in result.episodes:
            obj_results.setdefault(ep.object_name, []).append(ep)

        for obj_name, eps in obj_results.items():
            n_correct = sum(1 for e in eps if e.correct)
            print(f"  {obj_name}: {n_correct}/{len(eps)} correct")
            for e in eps:
                rot_str = f"({e.rotation[0]:.0f},{e.rotation[1]:.0f},{e.rotation[2]:.0f})"
                status = "OK" if e.correct else f"WRONG({e.detected_object})"
                print(f"    rot={rot_str}: {status} ev={e.max_evidence:.2f} "
                      f"steps={e.steps_to_converge} "
                      f"settle={e.mean_settling_iterations:.1f}")

        # Save results
        out_path = Path("benchmark_results_torch.json")
        result.save(out_path)
        print(f"\nResults saved to {out_path}")

    finally:
        harness.close()


if __name__ == "__main__":
    main()
