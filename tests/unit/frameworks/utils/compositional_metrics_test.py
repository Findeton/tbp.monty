from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tbp.monty.frameworks.utils.compositional_metrics import (
    load_compositional_taxonomy,
    summarize_compositional_eval,
)


class CompositionalMetricsTest(unittest.TestCase):
    def test_load_compositional_taxonomy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "taxonomy.yaml"
            path.write_text(
                "object_to_base:\n  002_cube_tbp_horz: cube\nobject_to_logo:\n  002_cube_tbp_horz: tbp\ntrain_objects:\n  - 002_cube_tbp_horz\nholdout_objects:\n  - 004_cube_numenta_horz\n",
                encoding="utf-8",
            )

            taxonomy = load_compositional_taxonomy(path)

            self.assertEqual(taxonomy["object_to_base"]["002_cube_tbp_horz"], "cube")
            self.assertEqual(taxonomy["holdout_objects"], ["004_cube_numenta_horz"])

    def test_summarize_compositional_eval(self):
        eval_stats = pd.DataFrame(
            [
                {
                    "primary_target_object": "004_cube_numenta_horz",
                    "most_likely_object": "002_cube_tbp_horz",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "007_disk_tbp_horz",
                    "most_likely_object": "007_disk_tbp_horz",
                    "primary_performance": "correct",
                },
                {
                    "primary_target_object": "014_cylinder_numenta_horz",
                    "most_likely_object": "019_sphere_numenta_horz",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "017_sphere_tbp_horz",
                    "most_likely_object": "011_cylinder",
                    "primary_performance": "confused",
                },
                {
                    "primary_target_object": "026_mug_numenta_horz",
                    "most_likely_object": None,
                    "primary_performance": "no_match",
                },
            ]
        )
        taxonomy = {
            "object_to_base": {
                "002_cube_tbp_horz": "cube",
                "004_cube_numenta_horz": "cube",
                "007_disk_tbp_horz": "disk",
                "014_cylinder_numenta_horz": "cylinder",
                "017_sphere_tbp_horz": "sphere",
                "019_sphere_numenta_horz": "sphere",
                "011_cylinder": "cylinder",
                "026_mug_numenta_horz": "mug",
            },
            "object_to_logo": {
                "002_cube_tbp_horz": "tbp",
                "004_cube_numenta_horz": "numenta",
                "007_disk_tbp_horz": "tbp",
                "014_cylinder_numenta_horz": "numenta",
                "017_sphere_tbp_horz": "tbp",
                "019_sphere_numenta_horz": "numenta",
                "011_cylinder": "bare",
                "026_mug_numenta_horz": "numenta",
            },
            "train_objects": ["002_cube_tbp_horz"],
            "holdout_objects": ["004_cube_numenta_horz"],
        }

        summary = summarize_compositional_eval(eval_stats, taxonomy)

        self.assertEqual(summary["rows"], 5)
        self.assertAlmostEqual(summary["exact_accuracy"], 20.0)
        self.assertAlmostEqual(summary["base_accuracy"], 40.0)
        self.assertAlmostEqual(summary["logo_accuracy"], 40.0)
        self.assertAlmostEqual(summary["both_parts_accuracy"], 20.0)
        self.assertAlmostEqual(summary["base_only_confusion_rate"], 20.0)
        self.assertAlmostEqual(summary["logo_only_confusion_rate"], 20.0)
        self.assertAlmostEqual(summary["cross_part_confusion_rate"], 20.0)
        self.assertAlmostEqual(summary["no_prediction_rate"], 20.0)

    def test_missing_targets_raise(self):
        eval_stats = pd.DataFrame(
            [
                {
                    "primary_target_object": "unknown_obj",
                    "most_likely_object": "002_cube_tbp_horz",
                }
            ]
        )
        taxonomy = {
            "object_to_base": {"002_cube_tbp_horz": "cube"},
            "object_to_logo": {"002_cube_tbp_horz": "tbp"},
        }

        with self.assertRaisesRegex(ValueError, "Targets missing from object_to_base"):
            summarize_compositional_eval(eval_stats, taxonomy)


if __name__ == "__main__":
    unittest.main()