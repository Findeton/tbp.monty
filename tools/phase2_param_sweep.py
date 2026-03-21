"""Parameter sweep for single-LM Omniglot evaluation.

Tests different max_match_distance, tolerances, and feature weights
on the existing trained checkpoint. No retraining needed.

Usage:
    conda run -n tbp.monty python tools/phase2_param_sweep.py
"""

import ast
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tbp.monty.frameworks.run_env import setup_env
setup_env()

from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)


def run_eval(config_overrides, run_name):
    """Run a single eval with the given parameter overrides."""
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from tbp.monty.hydra import register_resolvers
    register_resolvers()

    # Build config programmatically
    with hydra.initialize(config_path="../src/tbp/monty/conf", version_base=None):
        cfg = hydra.compose(
            config_name="experiment",
            overrides=[
                "experiment=phase2_omniglot_scaled6_eval_v2",
                f"experiment.config.logging.run_name={run_name}",
            ] + config_overrides,
        )

    output_dir = (
        Path(cfg.experiment.config.logging.output_dir)
        / cfg.experiment.config.logging.run_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.experiment.config.logging.output_dir = str(output_dir)

    experiment = hydra.utils.instantiate(cfg.experiment)
    with experiment:
        experiment.run()

    # Read results
    csv_path = output_dir / "eval_stats.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def analyze_results(df, taxonomy):
    """Compute exact and alphabet accuracy from eval stats."""
    n = len(df)
    exact = 0
    alpha = 0
    for _, row in df.iterrows():
        target = row["primary_target_object"]
        pred = str(row["most_likely_object"])
        if target == pred:
            exact += 1
        target_cat = taxonomy.get(target, target)
        pred_cat = taxonomy.get(pred, pred)
        if target_cat == pred_cat:
            alpha += 1
    return exact / n, alpha / n


def main():
    taxonomy = {}
    for alpha in [
        "Alphabet_of_the_Magi", "Anglo-Saxon_Futhorc", "Arcadian",
        "Armenian", "Asomtavruli_(Georgian)",
    ]:
        for i in range(1, 7):
            taxonomy[f"{alpha}_{i}"] = alpha

    configs = {
        # Baseline (current settings)
        "baseline": [],

        # Tighter spatial matching
        "mmd3": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.max_match_distance=3",
        ],
        "mmd2": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.max_match_distance=2",
        ],

        # Tighter pose tolerance
        "pose30": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.tolerances"
            ".patch.pose_vectors=${np.array:[30, 30, 30]}",
        ],
        "pose20": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.tolerances"
            ".patch.pose_vectors=${np.array:[20, 20, 20]}",
        ],

        # Tighter curvature tolerance
        "curv_half": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.tolerances"
            ".patch.principal_curvatures_log=${np.ones:2}*0.5",
        ],

        # Weight surface normal (dim 0) in addition to curvature direction
        "fw110": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.feature_weights"
            ".patch.pose_vectors=[1, 1, 0]",
        ],
        "fw111": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.feature_weights"
            ".patch.pose_vectors=[1, 1, 1]",
        ],

        # Combined: tighter spatial + tighter pose
        "mmd3_pose30": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.max_match_distance=3",
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.tolerances"
            ".patch.pose_vectors=${np.array:[30, 30, 30]}",
        ],

        # Combined: tighter spatial + all pose weights
        "mmd3_fw110": [
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.max_match_distance=3",
            "experiment.config.monty_config.learning_module_configs"
            ".learning_module_0.learning_module_args.feature_weights"
            ".patch.pose_vectors=[1, 1, 0]",
        ],
    }

    results = {}
    for name, overrides in configs.items():
        run_name = f"sweep_{name}"
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            df = run_eval(overrides, run_name)
            if df is not None:
                exact, alpha = analyze_results(df, taxonomy)
                results[name] = (exact, alpha, len(df))
                print(f"  -> Exact: {100*exact:.1f}%, Alphabet: {100*alpha:.1f}%")
            else:
                print(f"  -> No results CSV found")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results[name] = (0, 0, 0)

    print(f"\n{'='*60}")
    print(f"{'Config':20s} {'Exact':>8s} {'Alphabet':>10s} {'N':>5s}")
    print(f"{'='*60}")
    for name, (exact, alpha, n) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"{name:20s} {100*exact:7.1f}% {100*alpha:9.1f}% {n:5d}")


if __name__ == "__main__":
    main()
