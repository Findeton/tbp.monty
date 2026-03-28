"""Real Habitat temporal learning test.

Runs a full multi-episode Habitat eval with HPC enabled and measures:
1. What concepts the HPC learns from real LM recognition
2. Whether temporal predictions are correct
3. Whether HPC context priming affects recognition
4. Prediction accuracy over episodes

This is the REAL test — full sensor → motor → LM → HPC pipeline with
active exploration (LM directs where to look via goal state generator).

Usage:
    conda run -n tbp.monty python tools/run_temporal_learning_test.py
"""

import csv
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MONTY_LOGS", os.path.expanduser("~/tbp/results/monty"))
os.environ.setdefault("MONTY_DATA", os.path.expanduser("~/tbp/data"))
os.environ.setdefault(
    "MONTY_MODELS",
    os.path.expanduser("~/tbp/results/monty/pretrained_models"),
)
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

from tbp.monty.hydra import register_resolvers  # noqa: E402

register_resolvers()

from hydra import compose, initialize  # noqa: E402
import hydra  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Category mapping
CATEGORIES = {
    "015_peach": "fruit", "018_plum": "fruit",
    "057_racquetball": "ball", "058_golf_ball": "ball",
    "010_potted_meat_can": "can",
    "009_gelatin_box": "box", "036_wood_block": "box",
    "065-g_cups": "cup", "065-h_cups": "cup", "065-i_cups": "cup",
    "032_knife": "utensil",
    "043_phillips_screwdriver": "tool", "044_flat_screwdriver": "tool",
    "052_extra_large_clamp": "clamp",
    "072-d_toy_airplane": "airplane", "072-e_toy_airplane": "airplane",
    # Training objects (for HPC concept mapping)
    "011_banana": "fruit", "012_strawberry": "fruit",
    "013_apple": "fruit", "014_lemon": "fruit",
    "016_pear": "fruit", "017_orange": "fruit",
    "053_mini_soccer_ball": "ball", "054_softball": "ball",
    "055_baseball": "ball", "056_tennis_ball": "ball",
    "002_master_chef_can": "can", "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can",
    "003_cracker_box": "box", "004_sugar_box": "box",
    "008_pudding_box": "box",
    "025_mug": "cup",
    "030_fork": "utensil", "031_spoon": "utensil",
    "033_spatula": "utensil",
    "035_power_drill": "tool", "037_scissors": "tool",
    "042_adjustable_wrench": "tool",
    "050_medium_clamp": "clamp", "051_large_clamp": "clamp",
    "072-a_toy_airplane": "airplane",
    "072-b_toy_airplane": "airplane",
    "072-c_toy_airplane": "airplane",
}

# A diverse subset: 2 fruits, 2 balls, 1 can, 1 cup, 1 tool
TEST_OBJECTS = [
    "015_peach",
    "018_plum",
    "057_racquetball",
    "058_golf_ball",
    "010_potted_meat_can",
    "065-g_cups",
    "043_phillips_screwdriver",
]


def main():
    with initialize(config_path="../src/tbp/monty/conf", version_base=None):
        cfg = compose(
            config_name="experiment",
            overrides=["experiment=extended_ycb_eval_holdout_hpc"],
        )

    # Quick test: 7 objects, 2 rotations each = 14 episodes
    cfg.experiment.config.n_eval_epochs = 2
    cfg.experiment.config.eval_env_interface_args.object_names = TEST_OBJECTS

    out = (
        Path(cfg.experiment.config.logging.output_dir)
        / "temporal_learning_test_scale"
    )
    out.mkdir(exist_ok=True, parents=True)
    cfg.experiment.config.logging.output_dir = str(out)
    cfg.experiment.config.logging.run_name = "temporal_learning_test_scale"

    # Enable scale-invariant matching so the LM can actually recognize
    # holdout objects (different sizes than training objects)
    from omegaconf import OmegaConf
    scale_overrides = {
        "learning_module_configs": {
            "learning_module_0": {
                "learning_module_args": {
                    "scale_invariant_matching": True,
                    "max_match_distance": 0.01,
                    "feature_weights": {
                        "patch": {
                            "hsv": [1.0, 2.0, 0.5],
                            "pose_vectors": [1.0, 1.0, 1.0],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                }
            }
        }
    }
    OmegaConf.update(
        cfg, "experiment.config.monty_config", scale_overrides,
        merge=True, force_add=True,
    )

    logger.info(
        f"=== Temporal Learning Test ==="
        f"\n  Objects: {TEST_OBJECTS}"
        f"\n  Epochs: 2"
        f"\n  Episodes: {len(TEST_OBJECTS) * 2}"
    )

    experiment = hydra.utils.instantiate(cfg.experiment)
    start = time.time()

    with experiment:
        model = experiment.model
        lm = model.learning_modules[0]
        hpc = model.learning_modules[1]

        logger.info(
            f"  LM0: {type(lm).__name__} "
            f"({len(lm.get_all_known_object_ids())} objects in memory)"
        )
        logger.info(f"  LM1: {type(hpc).__name__}")
        logger.info(f"  sm_to_lm: {model.sm_to_lm_matrix}")
        logger.info(f"  lm_to_lm: {model.lm_to_lm_matrix}")

        experiment.run()

    elapsed = time.time() - start

    # ---- Analyze HPC state ----
    logger.info(f"\n{'=' * 60}")
    logger.info("HPC TEMPORAL LEARNING RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total time: {elapsed:.0f}s")

    # What concepts did HPC learn?
    logger.info(f"\nConcept SDRs learned: {len(hpc._concept_sdrs)}")
    for concept in sorted(hpc._concept_sdrs.keys()):
        cat = CATEGORIES.get(concept, "?")
        logger.info(f"  {concept:30s} (category: {cat})")

    # Episode sequence
    logger.info(f"\nEpisode sequence ({len(hpc.episodic_memory)} episodes):")
    terminals = []
    for i, ep in enumerate(hpc.episodic_memory):
        if ep:
            last = ep[-1]
            concepts = last.get("active_concepts", [])
            terminal = concepts[0] if concepts else "?"
        else:
            terminal = "?"
        terminals.append(terminal)
        target_obj = TEST_OBJECTS[i % len(TEST_OBJECTS)]
        target_cat = CATEGORIES.get(target_obj, "?")
        recog_cat = CATEGORIES.get(terminal, "?")
        match = "OK" if target_cat == recog_cat else "MISS"
        logger.info(
            f"  ep{i:2d}: target={target_obj:25s} ({target_cat:8s}) "
            f"recognized={terminal:25s} ({recog_cat:8s}) [{match}]"
        )

    # Temporal predictions
    logger.info(f"\nTemporal association matrix norm: "
                f"{float(hpc._temporal_W.sum()):.1f}")

    if hpc._last_episode_terminal:
        preds = hpc.get_temporal_predictions()
        logger.info(f"\nPredictions from '{hpc._last_episode_terminal}':")
        for concept, prob in sorted(preds.items(), key=lambda x: -x[1])[:5]:
            cat = CATEGORIES.get(concept, "?")
            logger.info(f"  {concept:30s} p={prob:.3f} ({cat})")

    # Prediction accuracy
    acc = hpc.get_prediction_accuracy()
    logger.info(f"\nPrediction accuracy:")
    logger.info(f"  Total predictions: {acc['total']}")
    logger.info(f"  Hits (observed in predicted set): {acc['hits']}")
    logger.info(f"  Top-1 correct: {acc['top_correct']}")
    if acc["total"] > 0:
        logger.info(f"  Hit rate: {acc['accuracy']:.1%}")
        logger.info(f"  Top-1 accuracy: {acc['top_accuracy']:.1%}")

    # Per-episode prediction detail
    if acc["history"]:
        logger.info(f"\nPrediction history:")
        for i, entry in enumerate(acc["history"]):
            top = entry["top_prediction"]
            obs = entry["observed"]
            hit = "HIT" if entry["hit"] else "MISS"
            top_ok = "TOP_OK" if entry["top_correct"] else "TOP_MISS"
            top_p = entry["predicted"].get(top, 0)
            logger.info(
                f"  pred{i:2d}: predicted={top:25s} (p={top_p:.2f}) "
                f"observed={obs:25s} [{hit}] [{top_ok}]"
            )

    # Context signal analysis
    logger.info(f"\nHPC context signal state:")
    signal = hpc.get_context_signal()
    if signal:
        tp = signal.get("temporal_predictions", {})
        if tp:
            logger.info(f"  Temporal predictions in context: {len(tp)}")
            for c, p in sorted(tp.items(), key=lambda x: -x[1])[:3]:
                logger.info(f"    {c}: {p:.3f}")
        assoc = signal.get("association_strengths", {})
        if assoc:
            logger.info(f"  Association strengths: {len(assoc)} entries")
            for c, s in sorted(assoc.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {c}: {s:.3f}")
    else:
        logger.info("  No context signal (no active concepts)")

    # Read eval CSV for comparison
    csv_path = out / "eval_stats.csv"
    if csv_path.exists():
        logger.info(f"\nEval stats CSV: {csv_path}")
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            logger.info(f"  Columns: {header}")
            for row in reader:
                if len(row) > 7:
                    mlh = row[5].strip().strip("'\"[] ")
                    target = row[7].strip().strip("'\"[] ")
                    steps = row[3].strip() if len(row) > 3 else "?"
                    mlh_cat = CATEGORIES.get(mlh, "?")
                    tgt_cat = CATEGORIES.get(target, "?")
                    match = "OK" if mlh_cat == tgt_cat else "MISS"
                    logger.info(
                        f"  target={target:25s} mlh={mlh:25s} "
                        f"steps={steps:4s} [{match}]"
                    )

    logger.info(f"\n{'=' * 60}")
    logger.info("DONE")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
