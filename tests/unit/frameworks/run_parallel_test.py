# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.


import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from omegaconf import OmegaConf

from tbp.monty.frameworks.loggers.monty_handlers import BasicCSVStatsHandler
from tbp.monty.frameworks.run_parallel import (
    is_partial_eval_run,
    parse_episode_spec,
    post_parallel_eval,
)


class ParseEpisodeSpecTest(unittest.TestCase):
    def test_selects_all(self):
        self.assertEqual(parse_episode_spec(None, total=5), [0, 1, 2, 3, 4])
        self.assertEqual(parse_episode_spec("", total=3), [0, 1, 2])
        self.assertEqual(parse_episode_spec(":", total=3), [0, 1, 2])
        self.assertEqual(parse_episode_spec("all", total=4), [0, 1, 2, 3])

    def test_single_index_valid(self):
        self.assertEqual(parse_episode_spec("3", total=6), [3])

    def test_single_index_out_of_bounds_raises(self):
        with self.assertRaisesRegex(ValueError, "not a valid index"):
            parse_episode_spec("5", total=5)

    def test_closed_range(self):
        self.assertEqual(parse_episode_spec("2:5", total=10), [2, 3, 4])

    def test_closed_range_equal_start_end_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("3:3", total=10)

    def test_closed_range_start_greater_than_end_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("5:3", total=10)

    def test_closed_range_end_equals_total_ok(self):
        self.assertEqual(parse_episode_spec("0:4", total=4), [0, 1, 2, 3])

    def test_closed_range_end_greater_than_total_raises(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("0:6", total=5)

    def test_open_left(self):
        self.assertEqual(parse_episode_spec(":3", total=10), [0, 1, 2])

    def test_open_left_requires_positive_end(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec(":0", total=10)

    def test_open_right(self):
        self.assertEqual(parse_episode_spec("7:", total=10), [7, 8, 9])

    def test_open_right_start_zero_ok(self):
        self.assertEqual(parse_episode_spec("0:", total=3), [0, 1, 2])

    def test_open_right_start_equals_total_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid range"):
            parse_episode_spec("5:", total=5)

    def test_mixed_selection(self):
        self.assertEqual(
            parse_episode_spec("0,3,5:8", total=10),
            [0, 3, 5, 6, 7],
        )

    def test_duplicates_eliminated_and_sorted(self):
        self.assertEqual(
            parse_episode_spec("2,2,1:3,0,0", total=5),
            [0, 1, 2],
        )

    def test_whitespace_is_ignored(self):
        self.assertEqual(
            parse_episode_spec("  0 ,  2 : 4 , 3  ", total=10),
            [0, 2, 3],
        )

    def test_letters_in_closed_range_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("a:b", total=10)

    def test_double_colon_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("3::5", total=10)

    def test_word_in_list_invalid(self):
        with self.assertRaisesRegex(ValueError, "not a valid selection"):
            parse_episode_spec("0,foo,2", total=10)

    def test_total_zero_all_is_empty(self):
        self.assertEqual(parse_episode_spec("all", total=0), [])

    def test_total_zero_none_is_empty(self):
        self.assertEqual(parse_episode_spec(None, total=0), [])

    def test_total_zero_any_specific_raises(self):
        with self.assertRaises(ValueError):
            parse_episode_spec("0", total=0)
        with self.assertRaises(ValueError):
            parse_episode_spec(":1", total=0)
        with self.assertRaises(ValueError):
            parse_episode_spec("0:", total=0)


class PartialEvalRunTest(unittest.TestCase):
    def test_detects_full_eval_selection(self):
        cfg = OmegaConf.create(
            {
                "episodes": "all",
                "experiment": {"config": {"do_train": False, "n_eval_epochs": 4}},
            }
        )

        self.assertFalse(is_partial_eval_run(cfg))

    def test_detects_subset_eval_selection(self):
        cfg = OmegaConf.create(
            {
                "episodes": "1:3",
                "experiment": {"config": {"do_train": False, "n_eval_epochs": 4}},
            }
        )

        self.assertTrue(is_partial_eval_run(cfg))

    def test_training_run_is_not_partial_eval(self):
        cfg = OmegaConf.create(
            {
                "episodes": "1:3",
                "experiment": {"config": {"do_train": True, "n_eval_epochs": 4}},
            }
        )

        self.assertFalse(is_partial_eval_run(cfg))


class PostParallelEvalTest(unittest.TestCase):
    def test_resume_eval_merges_existing_eval_stats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            existing_eval = base_dir / "eval_stats.csv"
            parallel_dir = base_dir / "parallel_eval_episode_1"
            parallel_dir.mkdir()

            pd.DataFrame(
                [
                    {"episode_seed": 100, "lm_id": "LM_0", "primary_performance": "correct"},
                    {"episode_seed": 100, "lm_id": "LM_1", "primary_performance": "correct"},
                ]
            ).to_csv(existing_eval, index=False)
            pd.DataFrame(
                [
                    {"episode_seed": 200, "lm_id": "LM_0", "primary_performance": "correct_mlh"},
                    {"episode_seed": 200, "lm_id": "LM_1", "primary_performance": "correct_mlh"},
                ]
            ).to_csv(parallel_dir / "eval_stats.csv", index=False)

            experiments = [
                {
                    "config": {
                        "logging": {
                            "output_dir": str(parallel_dir),
                            "detailed_save_per_episode": False,
                            "monty_handlers": [BasicCSVStatsHandler],
                            "python_log_to_file": False,
                        }
                    }
                }
            ]

            with mock.patch("hydra.utils.instantiate", return_value=object()):
                post_parallel_eval(
                    experiments,
                    base_dir,
                    merge_existing_eval_stats=True,
                )

            merged = pd.read_csv(existing_eval)

            self.assertEqual(len(merged), 4)
            self.assertEqual(sorted(merged["episode_seed"].unique().tolist()), [100, 200])
            self.assertFalse(parallel_dir.exists())


if __name__ == "__main__":
    unittest.main()
