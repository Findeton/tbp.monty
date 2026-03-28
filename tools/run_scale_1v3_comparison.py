"""Compare 1 LM vs 3 LM scale matching on a small object subset.

Tests whether multi-LM voting with diverse scale factors improves
category accuracy over a single LM, using a quick subset of objects
(balls, cans, fruit) with 2 epochs.

Usage:
    conda run -n tbp.monty python tools/run_scale_1v3_comparison.py
"""

import logging
import os
import time

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
from omegaconf import OmegaConf  # noqa: E402
import csv  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Category mapping for accuracy computation
CATEGORIES = {
    "015_peach": "fruit", "018_plum": "fruit",
    "057_racquetball": "ball", "058_golf_ball": "ball",
    "010_potted_meat_can": "can",
    "011_banana": "fruit", "012_strawberry": "fruit", "013_apple": "fruit",
    "014_lemon": "fruit", "016_pear": "fruit", "017_orange": "fruit",
    "053_mini_soccer_ball": "ball", "054_softball": "ball",
    "055_baseball": "ball", "056_tennis_ball": "ball",
    "002_master_chef_can": "can", "005_tomato_soup_can": "can",
    "007_tuna_fish_can": "can",
}

# Quick subset: balls + cans + 2 fruits (tests scale matching + feature discrimination)
TEST_OBJECTS = [
    "057_racquetball",
    "058_golf_ball",
    "010_potted_meat_can",
    "015_peach",
    "018_plum",
]


def compute_category_accuracy(csv_path):
    """Compute category accuracy from eval CSV."""
    total = correct = 0
    cat_stats = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            mlh = row[5].strip().strip("'\"[] ")
            target = row[7].strip().strip("'\"[] ")
            if mlh.startswith("["):
                mlh = mlh.split("'")[1] if "'" in mlh else mlh

            mlh_cat = CATEGORIES.get(mlh, "?")
            target_cat = CATEGORIES.get(target, "?")

            total += 1
            cat_correct = mlh_cat == target_cat
            if cat_correct:
                correct += 1

            if target_cat not in cat_stats:
                cat_stats[target_cat] = {"total": 0, "correct": 0}
            cat_stats[target_cat]["total"] += 1
            if cat_correct:
                cat_stats[target_cat]["correct"] += 1

    return correct, total, cat_stats


def run_experiment(name, lm_overrides, n_epochs=2):
    """Run an eval with the given LM config overrides."""
    from pathlib import Path

    with initialize(config_path="../src/tbp/monty/conf", version_base=None):
        cfg = compose(
            config_name="experiment",
            overrides=["experiment=extended_ycb_eval_holdout_scale"],
        )

    # Override to quick subset
    cfg.experiment.config.n_eval_epochs = n_epochs
    cfg.experiment.config.logging.run_name = f"scale_compare_{name}"

    out = (
        Path(cfg.experiment.config.logging.output_dir)
        / f"scale_compare_{name}"
    )
    out.mkdir(exist_ok=True, parents=True)
    cfg.experiment.config.logging.output_dir = str(out)

    # Override object list to quick subset
    cfg.experiment.config.eval_env_interface_args.object_names = TEST_OBJECTS

    # Apply LM overrides (force_add needed for keys like scale_factors_seed
    # that don't exist in the base structured config)
    OmegaConf.update(cfg, "experiment.config.monty_config", lm_overrides,
                     merge=True, force_add=True)

    logger.info(f"=== Running {name} ({n_epochs} epochs, {len(TEST_OBJECTS)} objects) ===")

    experiment = hydra.utils.instantiate(cfg.experiment)
    start = time.time()
    with experiment:
        # Log scale factors per LM
        for i, lm in enumerate(experiment.model.learning_modules):
            sf = getattr(lm.hypotheses_updater, "scale_factors", None)
            logger.info(f"  LM{i} scale_factors: {sf}")
        experiment.run()

    elapsed = time.time() - start
    csv_path = out / "eval_stats.csv"
    correct, total, cats = compute_category_accuracy(csv_path)
    pct = correct * 100 / total if total else 0

    logger.info(f"=== {name}: {correct}/{total} = {pct:.1f}% in {elapsed:.0f}s ===")
    for cat in sorted(cats.keys()):
        s = cats[cat]
        p = s["correct"] * 100 / s["total"] if s["total"] else 0
        logger.info(f"  {cat:10s}: {s['correct']}/{s['total']} = {p:.0f}%")

    return correct, total, cats, elapsed


def main():
    # --- 1 LM config ---
    lm_1lm = {
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
                },
            },
        },
    }

    # --- 3 LM config: each LM has a different scale_factors_seed ---
    lm_3lm = {
        "learning_module_configs": {
            "learning_module_0": {
                "learning_module_args": {
                    "scale_invariant_matching": True,
                    "scale_factors_seed": 0,
                    "max_match_distance": 0.01,
                    "feature_weights": {
                        "patch": {
                            "hsv": [1.0, 2.0, 0.5],
                            "pose_vectors": [1.0, 1.0, 1.0],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                },
            },
            "learning_module_1": {
                "learning_module_class": "${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}",
                "learning_module_args": {
                    "scale_invariant_matching": True,
                    "scale_factors_seed": 42,
                    "max_match_distance": 0.01,
                    "tolerances": {
                        "patch": {
                            "hsv": [0.1, 0.2, 0.2],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                    "feature_weights": {
                        "patch": {
                            "hsv": [1.0, 2.0, 0.5],
                            "pose_vectors": [1.0, 1.0, 1.0],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                },
            },
            "learning_module_2": {
                "learning_module_class": "${monty.class:tbp.monty.frameworks.models.evidence_matching.learning_module.EvidenceGraphLM}",
                "learning_module_args": {
                    "scale_invariant_matching": True,
                    "scale_factors_seed": 99,
                    "max_match_distance": 0.01,
                    "tolerances": {
                        "patch": {
                            "hsv": [0.1, 0.2, 0.2],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                    "feature_weights": {
                        "patch": {
                            "hsv": [1.0, 2.0, 0.5],
                            "pose_vectors": [1.0, 1.0, 1.0],
                            "principal_curvatures_log": [1.0, 1.0],
                        }
                    },
                },
            },
        },
        "sm_to_lm_matrix": [[0], [0], [0]],
        # No per-step voting: with a single shared sensor, all LMs get
        # identical observations → correlated errors → voting amplifies
        # noise instead of correcting it.  Each LM runs independently;
        # we compare per-LM results offline.
        "lm_to_lm_vote_matrix": None,
    }

    # --- 3 LM config with conditional voting ---
    import copy
    lm_3lm_cond = copy.deepcopy(lm_3lm)
    lm_3lm_cond["lm_to_lm_vote_matrix"] = [[1, 2], [0, 2], [0, 1]]
    lm_3lm_cond["monty_args"] = {
        "conditional_voting": True,
        "vote_confident_threshold": 2,
        "vote_after_steps": 100,
    }

    # Run experiments (skip completed ones)
    results = {}
    from pathlib import Path

    base_dir = Path(os.path.expanduser(
        "~/tbp/results/monty/projects/phase2_review_runs"
    ))

    for name, overrides in [
        ("1lm", lm_1lm),
        ("3lm", lm_3lm),
        ("3lm_cond", lm_3lm_cond),
    ]:
        csv_path = base_dir / f"scale_compare_{name}" / "eval_stats.csv"
        if csv_path.exists():
            correct, total, cats = compute_category_accuracy(csv_path)
            results[name] = (correct, total, cats, 0)
            logger.info(f"{name} already done: {correct}/{total}")
        else:
            r = run_experiment(name, overrides, n_epochs=2)
            results[name] = r

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    for name, (correct, total, cats, elapsed) in results.items():
        pct = correct * 100 / total if total else 0
        logger.info(f"{name}: {correct}/{total} = {pct:.1f}% ({elapsed:.0f}s)")
        for cat in sorted(cats.keys()):
            s = cats[cat]
            p = s["correct"] * 100 / s["total"] if s["total"] else 0
            logger.info(f"  {cat}: {s['correct']}/{s['total']} = {p:.0f}%")


if __name__ == "__main__":
    main()
