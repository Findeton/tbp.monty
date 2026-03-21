from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tbp.monty.frameworks.utils.category_metrics import (
    load_category_taxonomy,
    summarize_category_eval,
)


class CategoryMetricsTest(unittest.TestCase):
    def test_load_category_taxonomy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "taxonomy.yaml"
            path.write_text(
                "object_to_category:\n  mug: cup\n  fork: utensil\ntrain_objects:\n  - mug\nholdout_objects:\n  - fork\n"
            )

            taxonomy = load_category_taxonomy(path)

            self.assertEqual(taxonomy["object_to_category"]["mug"], "cup")
            self.assertEqual(taxonomy["holdout_objects"], ["fork"])

    def test_summarize_category_eval(self):
        eval_stats = pd.DataFrame(
            [
                {
                    "primary_target_object": "c_cups",
                    "most_likely_object": "mug",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "knife",
                    "most_likely_object": "fork",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "pudding_box",
                    "most_likely_object": "cracker_box",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "d_cups",
                    "most_likely_object": "fork",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "d_cups",
                    "most_likely_object": None,
                    "primary_performance": "no_match",
                },
            ]
        )
        taxonomy = {
            "object_to_category": {
                "mug": "cup",
                "c_cups": "cup",
                "d_cups": "cup",
                "knife": "utensil",
                "fork": "utensil",
                "pudding_box": "box",
                "cracker_box": "box",
            },
            "train_objects": ["mug", "fork", "cracker_box"],
            "holdout_objects": ["c_cups", "d_cups", "knife", "pudding_box"],
        }

        summary = summarize_category_eval(eval_stats, taxonomy)

        self.assertEqual(summary["rows"], 5)
        self.assertAlmostEqual(summary["exact_accuracy"], 0.0)
        self.assertAlmostEqual(summary["category_accuracy"], 60.0)
        self.assertAlmostEqual(summary["within_category_confusion_rate"], 60.0)
        self.assertAlmostEqual(summary["cross_category_confusion_rate"], 20.0)
        self.assertAlmostEqual(summary["no_prediction_rate"], 20.0)

    def test_missing_targets_raise(self):
        eval_stats = pd.DataFrame(
            [
                {
                    "primary_target_object": "unknown_obj",
                    "most_likely_object": "mug",
                }
            ]
        )
        taxonomy = {"object_to_category": {"mug": "cup"}}

        with self.assertRaisesRegex(ValueError, "Targets missing from taxonomy"):
            summarize_category_eval(eval_stats, taxonomy)


if __name__ == "__main__":
    unittest.main()