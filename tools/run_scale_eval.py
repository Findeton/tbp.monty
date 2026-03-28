"""Run extended YCB eval with scale-invariant matching.

Usage:
    conda run -n tbp.monty python tools/run_scale_eval.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    with initialize(config_path="../src/tbp/monty/conf", version_base=None):
        cfg = compose(
            config_name="experiment",
            overrides=["experiment=extended_ycb_eval_holdout_scale"],
        )

    from pathlib import Path

    output_dir = (
        Path(cfg.experiment.config.logging.output_dir)
        / cfg.experiment.config.logging.run_name
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    cfg.experiment.config.logging.output_dir = str(output_dir)

    logger.info("Instantiating experiment...")
    experiment = hydra.utils.instantiate(cfg.experiment)

    start = time.time()
    logger.info("Starting scale-invariant eval run...")
    with experiment:
        logger.info(f"Model: {type(experiment.model).__name__}")
        logger.info(f"Num LMs: {len(experiment.model.learning_modules)}")
        lm = experiment.model.learning_modules[0]
        logger.info(f"  scale_invariant_matching: {lm.scale_invariant_matching}")
        experiment.run()

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
