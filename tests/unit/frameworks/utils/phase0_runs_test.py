from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from tbp.monty.frameworks.experiments.monty_experiment import ExperimentMode, MontyExperiment
from tbp.monty.frameworks.utils.phase0_runs import (
    _launch_parts,
    build_manifest,
    format_episode_spec,
    get_status,
    load_manifest,
    summarize,
)


class FormatEpisodeSpecTest(unittest.TestCase):
    def test_formats_ranges(self):
        self.assertEqual(format_episode_spec([0, 1, 2, 4, 6, 7]), "0:3,4,6:8")

    def test_formats_singletons(self):
        self.assertEqual(format_episode_spec([3]), "3")

    def test_formats_empty(self):
        self.assertEqual(format_episode_spec([]), "")

    def test_launch_parts_quotes_multi_segment_episode_spec(self):
        parts = _launch_parts(
            experiment="test/eval",
            overrides=[],
            num_parallel=1,
            episodes="47:57,62:84",
        )

        self.assertIn("episodes='47:57,62:84'", parts)


class BuildManifestTest(unittest.TestCase):
    def test_build_manifest_for_pretraining_writes_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/supervised_pre_training",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )

            self.assertEqual(manifest.run_mode, "parallel_train")
            self.assertIn("pretrained/model.pt", "\n".join(manifest.expected_artifacts))
            self.assertNotIn("train_stats.csv", "\n".join(manifest.expected_artifacts))
            self.assertTrue(Path(manifest.manifest_path).exists())
            self.assertTrue(Path(manifest.config_path).exists())
            self.assertTrue((Path(manifest.output_dir) / "phase0_launch.sh").exists())

    def test_build_manifest_for_eval_predicts_eval_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )

            self.assertEqual(manifest.run_mode, "parallel_eval")
            self.assertIn("eval_stats.csv", "\n".join(manifest.expected_artifacts))


class StatusDetectionTest(unittest.TestCase):
    def test_status_reports_partial_for_missing_eval_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            parallel_dir = base_dir / f"{manifest.run_name}-parallel_eval_episode_0"
            parallel_dir.mkdir(parents=True, exist_ok=True)
            (parallel_dir / "eval_stats.csv").write_text("episode,primary_performance\n0,correct\n")

            status = get_status(Path(manifest.manifest_path))

            self.assertEqual(status["state"], "partial")
            self.assertTrue(len(status["completed_episodes"]) >= 1)
            self.assertTrue(status["missing_spec"])

    def test_status_reports_completed_when_final_eval_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            (base_dir / "eval_stats.csv").write_text("episode,primary_performance\n0,correct\n")
            (base_dir / "parallel_log.txt").write_text("ok\n")

            status = get_status(Path(manifest.manifest_path))

            self.assertEqual(status["state"], "completed")

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            loaded = load_manifest(Path(manifest.manifest_path))
            payload = json.loads(Path(manifest.manifest_path).read_text())

            self.assertEqual(loaded.run_name, manifest.run_name)
            self.assertEqual(payload["experiment"], "test/eval")

    def test_status_re_registers_resolvers_after_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            (base_dir / "eval_stats.csv").write_text("episode,primary_performance\n0,correct\n")
            (base_dir / "parallel_log.txt").write_text("ok\n")

            OmegaConf.clear_resolvers()
            status = get_status(Path(manifest.manifest_path))

            self.assertEqual(status["state"], "completed")

    def test_summarize_re_registers_resolvers_after_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            (base_dir / "eval_stats.csv").write_text(
                "episode,primary_performance\n0,correct\n1,correct_mlh\n2,incorrect\n"
            )
            (base_dir / "parallel_log.txt").write_text("ok\n")

            OmegaConf.clear_resolvers()
            summary = summarize(Path(manifest.manifest_path))

            self.assertTrue(summary["eval_stats_exists"])
            self.assertEqual(summary["eval_rows"], 3)
            self.assertAlmostEqual(summary["percent_correct"], 66.66666666666666)

    def test_completed_training_status_does_not_report_missing_after_merge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/supervised_pre_training",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            pretrained_dir = base_dir / "pretrained"
            pretrained_dir.mkdir(parents=True, exist_ok=True)
            (pretrained_dir / "model.pt").write_text("ok\n")
            (base_dir / "parallel_log.txt").write_text("ok\n")

            status = get_status(Path(manifest.manifest_path))

            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["missing_episodes"], [])

    def test_final_eval_file_with_leftover_parallel_dirs_is_partial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_manifest(
                experiment="test/eval",
                overrides=[f"++experiment.config.logging.output_dir={tmpdir}"],
                num_parallel=1,
            )
            base_dir = Path(manifest.output_dir)
            (base_dir / "eval_stats.csv").write_text("episode,primary_performance\n0,correct\n")
            (base_dir / "parallel_log.txt").write_text("ok\n")
            leftover_dir = base_dir / f"{manifest.run_name}-parallel_eval_episode_0"
            leftover_dir.mkdir(parents=True, exist_ok=True)
            (leftover_dir / "eval_stats.csv").write_text("episode,primary_performance\n0,correct\n")

            status = get_status(Path(manifest.manifest_path))

            self.assertEqual(status["state"], "partial")
            self.assertTrue(status["completed_episodes"])


class EvalPostEpochSaveBehaviorTest(unittest.TestCase):
    def test_post_epoch_skips_state_save_during_eval(self):
        experiment = object.__new__(MontyExperiment)
        experiment.experiment_mode = ExperimentMode.EVAL
        experiment.config = {"seed": 42}
        experiment._rng_seed_history = []
        experiment.total_train_steps = 0
        experiment.train_episodes = 0
        experiment.train_epochs = 0
        experiment.total_eval_steps = 0
        experiment.eval_episodes = 0
        experiment.env_interface = None
        experiment.output_dir = Path("/tmp/unused")
        experiment.save_state_dict = MagicMock()
        experiment.logger_handler = SimpleNamespace(post_epoch=MagicMock())
        experiment.eval_epochs = 0
        experiment.eval_env_interface = SimpleNamespace(post_epoch=MagicMock())

        experiment.post_epoch()

        experiment.save_state_dict.assert_not_called()
        experiment.logger_handler.post_epoch.assert_called_once()
        experiment.eval_env_interface.post_epoch.assert_called_once_with()
        self.assertEqual(experiment.eval_epochs, 1)


if __name__ == "__main__":
    unittest.main()