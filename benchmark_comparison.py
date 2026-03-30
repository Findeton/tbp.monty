"""Head-to-head comparison: CorticalColumnLM vs EvidenceGraphLM.

Uses the existing eval harnesses with YCB objects and identical Panda3D
rendering + CameraSM pipeline. Measures accuracy, speed, and memory.
"""
import json
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("comparison")

# Suppress noisy loggers
for name in [
    "tbp.monty.frameworks.models.evidence_matching",
    "tbp.monty.frameworks.models.graph_matching",
    "tbp.monty.simulators.panda3d",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


# --- Configuration ---
# Use 5 geometrically diverse objects for a meaningful comparison
OBJECTS = [
    "011_banana",
    "025_mug",
    "003_cracker_box",
    "013_apple",
    "035_power_drill",
]

# Same rotations for both architectures
EVAL_ROTATIONS = [
    (0.0, 0.0, 0.0),
    (0.0, 45.0, 0.0),
    (30.0, 0.0, 60.0),
    (0.0, 90.0, 0.0),
]

# Same observation pipeline
TRAIN_STEPS = 60
EVAL_STEPS = 60
RESOLUTION = (64, 64)
FOV = 90.0
ORBIT_RADIUS = 0.5
SEED = 42

# ---- Run CorticalColumn Eval ----
def run_cortical_column():
    from tbp.monty.simulators.panda3d.cortical_column_evaluation import (
        CorticalColumnEvalHarness,
    )

    logger.info("=" * 60)
    logger.info("CORTICAL COLUMN (SDR-based)")
    logger.info("=" * 60)

    harness = CorticalColumnEvalHarness(
        object_names=OBJECTS,
        eval_rotations=EVAL_ROTATIONS,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
        resolution=RESOLUTION,
        fov=FOV,
        orbit_radius=ORBIT_RADIUS,
        seed=SEED,
        column_kwargs={
            "n_minicolumns": 2048,
            "n_cells_per_minicolumn": 8,
            "sparsity": 0.03,
            "use_weight_memory": True,
            "use_attractor": True,
        },
        evidence_threshold=2.0,
    )
    try:
        result = harness.run()
        logger.info("\n%s", result.summary())
        return result
    finally:
        harness.close()


# ---- Run EvidenceGraphLM Eval ----
def run_evidence_graph():
    from tbp.monty.simulators.panda3d.evaluation import Panda3DEvalHarness

    logger.info("=" * 60)
    logger.info("EVIDENCE GRAPH LM (dense graph matching)")
    logger.info("=" * 60)

    harness = Panda3DEvalHarness(
        object_names=OBJECTS,
        eval_rotations=EVAL_ROTATIONS,
        train_steps=TRAIN_STEPS,
        eval_steps=EVAL_STEPS,
        resolution=RESOLUTION,
        fov=FOV,
        orbit_radius=ORBIT_RADIUS,
        seed=SEED,
    )
    try:
        result = harness.run()
        logger.info("\n%s", result.summary())
        return result
    finally:
        harness.close()


def print_comparison(cc_result, eg_result):
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    print(f"Objects: {OBJECTS}")
    print(f"Rotations: {len(EVAL_ROTATIONS)} per object")
    print(f"Episodes: {len(OBJECTS) * len(EVAL_ROTATIONS)} total")
    print(f"Train steps: {TRAIN_STEPS}, Eval steps: {EVAL_STEPS}")
    print(f"Resolution: {RESOLUTION}, FOV: {FOV}°")
    print()

    header = f"{'Metric':<35} {'CorticalColumn':>18} {'EvidenceGraphLM':>18}"
    print(header)
    print("-" * len(header))

    def row(label, cc_val, eg_val, fmt=".1f"):
        cc_str = f"{cc_val:{fmt}}" if cc_val is not None else "N/A"
        eg_str = f"{eg_val:{fmt}}" if eg_val is not None else "N/A"
        print(f"{label:<35} {cc_str:>18} {eg_str:>18}")

    row("Accuracy (%)", 
        cc_result.accuracy * 100 if cc_result else None,
        eg_result.accuracy * 100 if eg_result else None,
        ".1f")

    row("Mean steps to converge",
        cc_result.mean_steps_to_converge if cc_result else None,
        eg_result.mean_steps_to_converge if eg_result else None,
        ".1f")

    row("Mean rotation error (deg)",
        cc_result.mean_rotation_error_deg if cc_result else None,
        eg_result.mean_rotation_error_deg if eg_result else None,
        ".1f")

    row("Wall clock total (s)",
        cc_result.wall_clock_total if cc_result else None,
        eg_result.wall_clock_total if eg_result else None,
        ".1f")

    # Memory from config
    cc_mem = cc_result.config.get("memory_bytes") if cc_result else None
    eg_mem = eg_result.config.get("memory_bytes") if eg_result else None
    row("Memory (KB)",
        cc_mem / 1024 if cc_mem else None,
        eg_mem / 1024 if eg_mem else None,
        ".1f")

    print()

    # Per-object breakdown
    print("PER-OBJECT ACCURACY:")
    print(f"{'Object':<25} {'CC':>8} {'EG':>8}")
    print("-" * 45)
    for obj_name in OBJECTS:
        cc_eps = [e for e in (cc_result.episodes if cc_result else []) 
                  if e.object_name == obj_name]
        eg_eps = [e for e in (eg_result.episodes if eg_result else [])
                  if e.object_name == obj_name]
        cc_acc = sum(1 for e in cc_eps if e.correct) / max(len(cc_eps), 1) * 100
        eg_acc = sum(1 for e in eg_eps if e.correct) / max(len(eg_eps), 1) * 100
        cc_mark = "✓" if cc_acc == 100 else f"{cc_acc:.0f}%"
        eg_mark = "✓" if eg_acc == 100 else f"{eg_acc:.0f}%"
        print(f"  {obj_name:<23} {cc_mark:>8} {eg_mark:>8}")

    print()

    # Per-rotation breakdown  
    print("PER-ROTATION ACCURACY:")
    print(f"{'Rotation':<25} {'CC':>8} {'EG':>8}")
    print("-" * 45)
    for rot in EVAL_ROTATIONS:
        cc_eps = [e for e in (cc_result.episodes if cc_result else [])
                  if e.rotation == rot]
        eg_eps = [e for e in (eg_result.episodes if eg_result else [])
                  if e.rotation == rot]
        cc_acc = sum(1 for e in cc_eps if e.correct) / max(len(cc_eps), 1) * 100
        eg_acc = sum(1 for e in eg_eps if e.correct) / max(len(eg_eps), 1) * 100
        print(f"  {str(rot):<23} {cc_acc:>7.0f}% {eg_acc:>7.0f}%")

    print()

    # Detailed episode results
    print("DETAILED EPISODE RESULTS:")
    print(f"{'Object':<18} {'Rotation':<18} {'CC det':<12} {'CC ok':>5} {'CC ev':>7} {'EG det':<12} {'EG ok':>5} {'EG ev':>7}")
    print("-" * 95)
    for obj in OBJECTS:
        for rot in EVAL_ROTATIONS:
            cc_ep = next((e for e in (cc_result.episodes if cc_result else [])
                         if e.object_name == obj and e.rotation == rot), None)
            eg_ep = next((e for e in (eg_result.episodes if eg_result else [])
                         if e.object_name == obj and e.rotation == rot), None)
            cc_det = cc_ep.detected_object[:10] if cc_ep and cc_ep.detected_object else "?"
            cc_ok = "✓" if cc_ep and cc_ep.correct else "✗"
            cc_ev = f"{cc_ep.max_evidence:.2f}" if cc_ep else "?"
            eg_det = eg_ep.detected_object[:10] if eg_ep and eg_ep.detected_object else "?"
            eg_ok = "✓" if eg_ep and eg_ep.correct else "✗"
            eg_ev = f"{eg_ep.max_evidence:.2f}" if eg_ep else "?"
            rot_str = f"({rot[0]:.0f},{rot[1]:.0f},{rot[2]:.0f})"
            print(f"  {obj:<16} {rot_str:<18} {cc_det:<12} {cc_ok:>5} {cc_ev:>7} {eg_det:<12} {eg_ok:>5} {eg_ev:>7}")


if __name__ == "__main__":
    # Run both architectures
    cc_result = run_cortical_column()
    eg_result = run_evidence_graph()
    print_comparison(cc_result, eg_result)

    # Save raw results
    results = {
        "cortical_column": cc_result.to_dict() if cc_result else None,
        "evidence_graph": eg_result.to_dict() if eg_result else None,
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to benchmark_results.json")
