# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Benchmark comparison: CorticalColumn (SDR) vs EvidenceGraphLM (dense).

Runs both architectures on the same set of YCB objects through the Panda3D
eval pipeline and prints a side-by-side comparison of accuracy, speed,
and memory usage.

Usage::

    python -m tbp.monty.simulators.panda3d.benchmark_comparison

Or from Python::

    from tbp.monty.simulators.panda3d.benchmark_comparison import run_comparison
    run_comparison(n_objects=3, train_steps=40, eval_steps=40)
"""

from __future__ import annotations

import logging
import sys
import time

from tbp.monty.simulators.panda3d.ycb import YCB_EVAL_OBJECTS

logger = logging.getLogger(__name__)


def run_comparison(
    n_objects: int = 3,
    train_steps: int = 40,
    eval_steps: int = 40,
    resolution: tuple[int, int] = (64, 64),
    seed: int = 42,
) -> dict:
    """Run both architectures and return comparison metrics.

    Parameters
    ----------
    n_objects : int
        Number of YCB objects to use.
    train_steps : int
        Orbital steps per training episode.
    eval_steps : int
        Orbital steps per evaluation episode.
    resolution : tuple[int, int]
        Render resolution.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Comparison results with keys "cortical_column" and "evidence_graph_lm".
    """
    objects = YCB_EVAL_OBJECTS[:n_objects]
    results = {}

    # --- CorticalColumn (SDR) ---
    print(f"\n{'='*60}")
    print("  CorticalColumn (SDR-based)")
    print(f"{'='*60}")

    from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
        CorticalColumnEvalHarness,
    )

    cc_harness = CorticalColumnEvalHarness(
        object_names=objects,
        train_steps=train_steps,
        eval_steps=eval_steps,
        resolution=resolution,
        seed=seed,
    )
    t0 = time.monotonic()
    cc_result = cc_harness.run()
    cc_time = time.monotonic() - t0

    column = cc_harness.get_column()
    cc_memory = column.memory.memory_bytes()
    cc_segments = column.dendrites.total_segments

    cc_harness.close()

    results["cortical_column"] = {
        "accuracy": cc_result.accuracy,
        "mean_steps": cc_result.mean_steps_to_converge,
        "wall_clock": cc_time,
        "memory_bytes": cc_memory,
        "n_segments": cc_segments,
        "episodes": [
            {
                "object": e.object_name,
                "rotation": e.rotation,
                "detected": e.detected_object,
                "correct": e.correct,
                "steps": e.steps_to_converge,
                "evidence": e.max_evidence,
            }
            for e in cc_result.episodes
        ],
    }

    # --- EvidenceGraphLM (dense) ---
    print(f"\n{'='*60}")
    print("  EvidenceGraphLM (dense feature grid)")
    print(f"{'='*60}")

    from tbp.monty.simulators.panda3d.evaluation import Panda3DEvalHarness

    lm_harness = Panda3DEvalHarness(
        object_names=objects,
        train_steps=train_steps,
        eval_steps=eval_steps,
        resolution=resolution,
        seed=seed,
    )
    t0 = time.monotonic()
    lm_result = lm_harness.run()
    lm_time = time.monotonic() - t0

    # Estimate LM memory
    lm = lm_harness.get_lm()
    lm_memory = 0
    for graph_id in lm.get_all_known_object_ids():
        graph = lm.graph_memory.get_graph(graph_id)
        if graph is not None:
            lm_memory += graph.number_of_nodes() * 200  # rough estimate

    lm_harness.close()

    results["evidence_graph_lm"] = {
        "accuracy": lm_result.accuracy,
        "mean_steps": lm_result.mean_steps_to_converge,
        "wall_clock": lm_time,
        "memory_bytes": lm_memory,
        "episodes": [
            {
                "object": e.object_name,
                "rotation": e.rotation,
                "detected": e.detected_object,
                "correct": e.correct,
                "steps": e.steps_to_converge,
                "evidence": e.max_evidence,
            }
            for e in lm_result.episodes
        ],
    }

    # --- Print comparison ---
    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Objects: {objects}")
    print(f"Train steps: {train_steps}, Eval steps: {eval_steps}")
    print(f"Resolution: {resolution}")
    print()

    cc = results["cortical_column"]
    lm = results["evidence_graph_lm"]

    print(f"{'Metric':<30} {'CorticalColumn':>15} {'EvidenceGraphLM':>15}")
    print("-" * 62)
    print(f"{'Accuracy':<30} {cc['accuracy']:>14.1%} {lm['accuracy']:>14.1%}")
    print(
        f"{'Mean steps to converge':<30} "
        f"{str(cc['mean_steps'] or 'N/A'):>15} "
        f"{str(lm['mean_steps'] or 'N/A'):>15}"
    )
    print(f"{'Wall clock (s)':<30} {cc['wall_clock']:>14.1f} {lm['wall_clock']:>14.1f}")
    print(
        f"{'Memory (KB)':<30} "
        f"{cc['memory_bytes']/1024:>14.1f} "
        f"{lm['memory_bytes']/1024:>14.1f}"
    )
    if "n_segments" in cc:
        print(f"{'Dendrite segments':<30} {cc['n_segments']:>15}")
    print()

    # Per-episode comparison
    print(f"{'Object':<20} {'Rot':<20} {'CC':>8} {'LM':>8}")
    print("-" * 58)
    for cc_ep, lm_ep in zip(cc["episodes"], lm["episodes"]):
        rot_str = str(cc_ep["rotation"])
        cc_ok = "Y" if cc_ep["correct"] else "N"
        lm_ok = "Y" if lm_ep["correct"] else "N"
        print(f"{cc_ep['object']:<20} {rot_str:<20} {cc_ok:>8} {lm_ok:>8}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    n_objects = 3
    if len(sys.argv) > 1:
        n_objects = int(sys.argv[1])

    run_comparison(n_objects=n_objects)


if __name__ == "__main__":
    main()
