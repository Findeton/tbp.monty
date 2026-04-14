# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Integration smoke test for the Track 12 real-asset falsifier benchmark.

This test intentionally exercises the benchmark on a single real animated mesh
with lightweight settings. It verifies that the benchmark stays wired to the
real-asset, trace-bank-centered, self-supervised predictive temporal path,
keeps the child LMs on the same default temporal stack, and produces the
expected staged report structure.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

try:
    import panda3d  # noqa: F401
    import gltf  # noqa: F401
except ImportError:
    raise unittest.SkipTest("Panda3D or panda3d-gltf not installed")


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_track12_real_time_falsifiers.py"
ASSET_DIR = (
    REPO_ROOT
    / "tests"
    / "unit"
    / "simulators"
    / "panda3d"
    / "test_assets"
    / "animated"
)
FOX_PATH = ASSET_DIR / "Fox.glb"

pytestmark = pytest.mark.xdist_group(name="track12-real-time-falsifiers")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "track12_real_time_falsifiers",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark script from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _track14_resolved_primary_contract(report):
    aggregate = report.get("aggregate", {})
    return (
        aggregate.get("matched_top1_accuracy_mean") == 1.0
        and aggregate.get("matched_top1_resolved_fraction_mean") == 1.0
        and aggregate.get("matched_top1_strict_accuracy_mean") == 1.0
    )


def _run_track14_benchmark_cli(
    models,
    *,
    diagnostic_harness=False,
    max_attempts=1,
    accept_report=None,
    model_orders=None,
):
    candidate_orders = list(model_orders or [list(models)])
    last_report = None
    for order_index, requested_models in enumerate(candidate_orders, start=1):
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--models",
            *list(requested_models),
            "--phase-bins",
            "4",
            "--train-cycles",
            "1",
            "--eval-cycles",
            "1",
            "--stretch",
            "2",
            "--compression-stride",
            "2",
            "--resolution",
            "16",
            "16",
            "--n-minicolumns",
            "128",
            "--sparsity",
            "0.05",
            "--seed",
            "42",
            "--lm-family",
            "predictive_hypothesis_torch",
            "--detail-grid-shape",
            "2",
            "2",
            "--use-detail-aware-sensors",
        ]
        if diagnostic_harness:
            cmd.append("--diagnostic-harness")

        for attempt in range(1, max(1, int(max_attempts)) + 1):
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            report["cli_attempt"] = attempt
            report["cli_model_order_index"] = order_index
            last_report = report
            if accept_report is None or accept_report(report):
                return report

    return last_report


@contextmanager
def _real_asset_benchmark_lock():
    if fcntl is None:
        yield
        return

    lock_path = Path(tempfile.gettempdir()) / "tbp_monty_real_asset_benchmark.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@unittest.skipUnless(SCRIPT_PATH.is_file(), "Track 12 benchmark script not found")
@unittest.skipUnless(FOX_PATH.is_file(), "Fox.glb not found")
class TestTrack12RealTimeFalsifiers(unittest.TestCase):
    """Smoke-test the real-asset Track 12 falsifier benchmark."""

    @classmethod
    def setUpClass(cls):
        cls._benchmark = _load_benchmark_module()

    def test_extract_phase_info_rejects_degenerate_foot_cycle_output(self):
        class FakeAnimatedObject:
            @staticmethod
            def get_num_frames(anim_name):
                return 4

        original_foot = self._benchmark.extract_foot_cycle_phase_info
        original_joint = self._benchmark.extract_joint_phase_info
        try:
            self._benchmark.extract_foot_cycle_phase_info = lambda *args, **kwargs: {
                "joint_names": ["left", "right"],
                "root_joint_name": "root",
                "phase_by_frame": [0, 0, 0, 0],
                "phase_angles": [0.0, 0.0, 0.0, 0.0],
                "phase_span": 0.0,
                "phase_method": "foot_cycle_angle",
            }
            self._benchmark.extract_joint_phase_info = lambda *args, **kwargs: {
                "joint_names": ["left", "right"],
                "root_joint_name": "root",
                "phase_by_frame": [0, 1, 2, 3],
                "phase_angles": [0.0, 1.0, 2.0, 3.0],
                "phase_span": 3.0,
                "phase_method": "joint_configuration_pca",
            }

            phase_info = self._benchmark._extract_phase_info(
                FakeAnimatedObject(),
                anim_name="walk",
                n_phase_bins=4,
                preferred_method="foot_cycle",
            )
        finally:
            self._benchmark.extract_foot_cycle_phase_info = original_foot
            self._benchmark.extract_joint_phase_info = original_joint

        self.assertEqual(phase_info["phase_method"], "joint_configuration_pca")
        self.assertEqual(phase_info["phase_by_frame"], [0, 1, 2, 3])

    def test_extract_phase_info_supports_explicit_frame_bins(self):
        class FakeAnimatedObject:
            @staticmethod
            def get_num_frames(anim_name):
                return 4

        original_foot = self._benchmark.extract_foot_cycle_phase_info
        original_joint = self._benchmark.extract_joint_phase_info
        try:
            def _unexpected(*args, **kwargs):
                raise AssertionError("kinematic extraction should not be called")

            self._benchmark.extract_foot_cycle_phase_info = _unexpected
            self._benchmark.extract_joint_phase_info = _unexpected

            phase_info = self._benchmark._extract_phase_info(
                FakeAnimatedObject(),
                anim_name="walk",
                n_phase_bins=4,
                preferred_method="frame_index",
            )
        finally:
            self._benchmark.extract_foot_cycle_phase_info = original_foot
            self._benchmark.extract_joint_phase_info = original_joint

        self.assertEqual(phase_info["phase_method"], "frame_index_bins_requested")
        self.assertEqual(phase_info["phase_by_frame"], [0, 1, 2, 3])

    def test_extract_phase_info_uses_frame_bins_when_kinematic_outputs_degenerate(self):
        class FakeAnimatedObject:
            @staticmethod
            def get_num_frames(anim_name):
                return 4

        original_foot = self._benchmark.extract_foot_cycle_phase_info
        original_joint = self._benchmark.extract_joint_phase_info
        try:
            self._benchmark.extract_foot_cycle_phase_info = lambda *args, **kwargs: {
                "joint_names": ["left", "right"],
                "root_joint_name": "root",
                "phase_by_frame": [0, 0, 0, 0],
                "phase_angles": [0.0, 0.0, 0.0, 0.0],
                "phase_span": 0.0,
                "phase_method": "foot_cycle_angle",
            }
            self._benchmark.extract_joint_phase_info = lambda *args, **kwargs: {
                "joint_names": ["left", "right"],
                "root_joint_name": "root",
                "phase_by_frame": [1, 1, 1, 1],
                "phase_angles": [0.0, 0.0, 0.0, 0.0],
                "phase_span": 0.0,
                "phase_method": "joint_configuration_pca",
            }

            phase_info = self._benchmark._extract_phase_info(
                FakeAnimatedObject(),
                anim_name="walk",
                n_phase_bins=4,
                preferred_method="auto",
            )
        finally:
            self._benchmark.extract_foot_cycle_phase_info = original_foot
            self._benchmark.extract_joint_phase_info = original_joint

        self.assertEqual(phase_info["phase_method"], "frame_index_bins_fallback")
        self.assertEqual(phase_info["phase_by_frame"], [0, 1, 2, 3])

    def test_resolve_child_lm_kwargs_mirrors_single_override_for_parity(self):
        behavior_only = {
            "temporal_trace_config": {"trace_weight": 0.5},
            "action_predictive_config": {"query_bias_weight": 0.3},
        }

        morphology_kwargs, behavior_kwargs = (
            self._benchmark._resolve_child_lm_kwargs(
                morphology_lm_kwargs=None,
                behavior_lm_kwargs=behavior_only,
            )
        )

        self.assertEqual(morphology_kwargs, behavior_kwargs)
        self.assertIsNot(morphology_kwargs, behavior_kwargs)

        morphology_kwargs["temporal_trace_config"]["trace_weight"] = 0.9
        self.assertEqual(
            behavior_kwargs["temporal_trace_config"]["trace_weight"],
            0.5,
        )

    def test_normalize_detail_grid_shape_clamps_to_positive_ints(self):
        self.assertEqual(
            self._benchmark._normalize_detail_grid_shape((0, -3)),
            (1, 1),
        )

    def test_apply_family_lm_defaults_uses_wider_parent_hypothesis_bank(self):
        parent_defaults = self._benchmark._apply_family_lm_defaults(
            "predictive_hypothesis_torch",
            {},
            is_parent=True,
        )
        child_defaults = self._benchmark._apply_family_lm_defaults(
            "predictive_hypothesis_torch",
            {},
            is_parent=False,
        )

        self.assertEqual(parent_defaults["output_evidence_threshold"], 0.0)
        self.assertEqual(child_defaults["output_evidence_threshold"], 0.0)
        self.assertEqual(parent_defaults["core_kwargs"]["max_hypotheses"], 6)
        self.assertEqual(child_defaults["core_kwargs"]["max_hypotheses"], 4)

    def test_learning_modules_missing_target_object_reports_predictive_gaps(self):
        lm_present = mock.Mock()
        lm_present.learning_module_id = "lm_present"
        lm_present.get_all_known_object_ids.return_value = ["fox", "robot"]

        lm_missing = mock.Mock()
        lm_missing.learning_module_id = "lm_missing"
        lm_missing.get_all_known_object_ids.return_value = ["fox"]

        exp = mock.Mock()
        exp.monty.learning_modules = [lm_present, lm_missing]

        self.assertEqual(
            self._benchmark._learning_modules_missing_target_object(
                exp,
                "robot",
                "predictive_hypothesis_torch",
            ),
            ["lm_missing"],
        )
        self.assertEqual(
            self._benchmark._learning_modules_missing_target_object(
                exp,
                "robot",
                "cortical_column_torch",
            ),
            [],
        )

    def test_result_has_processed_learning_step_ignores_fully_skipped_trace(self):
        skipped_result = {
            "trace": [
                {
                    "learning_modules": {
                        "lm_behavior": {"evidence_debug": {"step_skipped": True}},
                        "lm_parent": {"evidence_debug": {"step_skipped": True}},
                    }
                }
            ]
        }
        processed_result = {
            "trace": [
                {
                    "learning_modules": {
                        "lm_behavior": {"evidence_debug": {"step_skipped": True}},
                        "lm_parent": {"evidence_debug": {"step_skipped": False}},
                    }
                }
            ]
        }

        self.assertFalse(
            self._benchmark._result_has_processed_learning_step(skipped_result)
        )
        self.assertTrue(
            self._benchmark._result_has_processed_learning_step(processed_result)
        )

    def test_track14_benchmark_geometry_defaults_strengthen_small_mesh_setup(self):
        spec = {
            "path": FOX_PATH,
            "object_scale": (0.01, 0.01, 0.01),
            "initial_distance": 3.0,
        }

        adjusted_spec = self._benchmark._apply_family_model_geometry_defaults(
            spec,
            "predictive_hypothesis_torch",
        )
        adjusted_resolution = self._benchmark._apply_family_resolution_defaults(
            (16, 16),
            "predictive_hypothesis_torch",
        )
        unchanged_resolution = self._benchmark._apply_family_resolution_defaults(
            (16, 16),
            "cortical_column_torch",
        )

        self.assertEqual(adjusted_spec["initial_distance"], 2.0)
        self.assertEqual(adjusted_resolution, (24, 24))
        self.assertEqual(unchanged_resolution, (16, 16))

    def test_pairwise_subprocess_forwards_lm_family_and_detail_grid_shape(self):
        completed = mock.Mock()
        completed.stdout = "{\"status\": \"ok\"}"

        with mock.patch.object(
            self._benchmark.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            result = self._benchmark._run_pairwise_benchmark_subprocess(
                model_pair=("fox", "robot"),
                phase_bins=4,
                train_cycles=1,
                eval_cycles=1,
                stretch=2,
                compression_stride=2,
                resolution=(24, 24),
                column_kwargs={"n_minicolumns": 128, "sparsity": 0.05, "seed": 7},
                morphology_lm_kwargs=None,
                behavior_lm_kwargs=None,
                parent_column_kwargs=None,
                parent_lm_kwargs=None,
                boundary_active_threshold=0.37,
                fully_self_supervised_lms=False,
                object_decoder_tail_steps=5,
                lm_family="predictive_hypothesis_torch",
                detail_grid_shape=(2, 2),
                use_detail_aware_sensors=True,
            )

        self.assertEqual(result, {"status": "ok"})
        cmd = run_mock.call_args.args[0]
        self.assertIn("--lm-family", cmd)
        self.assertIn("predictive_hypothesis_torch", cmd)
        grid_idx = cmd.index("--detail-grid-shape")
        self.assertEqual(cmd[grid_idx + 1 : grid_idx + 3], ["2", "2"])
        self.assertIn("--use-detail-aware-sensors", cmd)

    def test_primary_metrics_use_boundary_pressure_directly(self):
        trace = [
            {
                "learning_modules": {
                    "lm_behavior": {
                        "temporal_surprise": 0.2,
                        "temporal_status": "confident",
                        "temporal_context": {
                            "boundary_pressure": 0.2,
                            "event_detected": False,
                        },
                    }
                }
            },
            {
                "learning_modules": {
                    "lm_behavior": {
                        "temporal_surprise": 0.7,
                        "temporal_status": "confused",
                        "temporal_context": {
                            "boundary_pressure": 0.8,
                            "event_detected": False,
                        },
                    }
                }
            },
            {
                "learning_modules": {
                    "lm_behavior": {
                        "temporal_surprise": 0.6,
                        "temporal_status": "confident",
                        "temporal_context": {
                            "boundary_pressure": 0.9,
                            "event_detected": False,
                        },
                    }
                }
            },
        ]

        metrics = self._benchmark._compute_primary_temporal_metrics(trace)
        stats = self._benchmark._window_stats(trace, 0, len(trace))

        self.assertAlmostEqual(metrics["temporal_boundary_pressure_mean"], 0.6333333333)
        self.assertEqual(metrics["temporal_boundary_pressure_max"], 0.9)
        self.assertAlmostEqual(metrics["temporal_boundary_active_fraction"], 2 / 3)
        self.assertEqual(metrics["temporal_event_fraction"], 0.0)
        self.assertAlmostEqual(stats["mean_boundary_pressure"], 0.6333333333)
        self.assertAlmostEqual(stats["boundary_active_fraction"], 2 / 3)

    def test_condition_report_uses_target_lm_final_prediction(self):
        report = self._benchmark._condition_report(
            result={
                "graph_id": "morph_wrong",
                "lm_behavior": {"graph_id": "fox"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="fox",
            target_lm_id="lm_behavior",
        )

        self.assertEqual(report["primary"]["final_prediction"], "fox")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertEqual(report["primary"]["final_prediction_lm_id"], "lm_behavior")
        self.assertIn(
            "final_prediction_latent_id",
            report["primary"],
        )
        self.assertIn(
            "final_prediction_graph_id",
            report["primary"],
        )

    def test_fit_self_supervised_object_decoders_maps_raw_graph_ids(self):
        decoders, diagnostics = self._benchmark._fit_self_supervised_object_decoders(
            {
                "fox": {
                    "matched": {
                        "lm_morphology": {"graph_id": "auto_geom_fox"},
                        "lm_behavior": {"graph_id": "auto_beh_fox"},
                        "lm_parent": {"graph_id": "auto_parent_fox"},
                        "trace": [
                            {
                                "learning_modules": {
                                    "lm_morphology": {"graph_id": "auto_geom_fox"},
                                    "lm_behavior": {"graph_id": "auto_beh_fox"},
                                    "lm_parent": {"graph_id": "auto_parent_fox"},
                                }
                            }
                        ],
                    }
                },
                "robot": {
                    "matched": {
                        "lm_morphology": {"graph_id": "auto_geom_robot"},
                        "lm_behavior": {"graph_id": "auto_beh_robot"},
                        "lm_parent": {"graph_id": "auto_parent_robot"},
                        "trace": [
                            {
                                "learning_modules": {
                                    "lm_morphology": {"graph_id": "auto_geom_robot"},
                                    "lm_behavior": {"graph_id": "auto_beh_robot"},
                                    "lm_parent": {"graph_id": "auto_parent_robot"},
                                }
                            }
                        ],
                    }
                },
            },
            tail_steps=1,
        )

        self.assertEqual(decoders["lm_morphology"]["auto_geom_fox"], "fox")
        self.assertEqual(decoders["lm_behavior"]["auto_beh_robot"], "robot")
        self.assertEqual(
            diagnostics["lm_parent"]["latent_id_to_object"]["auto_parent_fox"],
            "fox",
        )
        self.assertEqual(
            diagnostics["lm_parent"]["graph_id_to_object"]["auto_parent_fox"],
            "fox",
        )
        self.assertEqual(
            diagnostics["lm_behavior"]["latent_id_support"]["auto_beh_robot"],
            {"robot": 2},
        )

    def test_fit_self_supervised_object_decoders_marks_near_shared_behavior_latents_ambiguous(self):
        decoders, diagnostics = self._benchmark._fit_self_supervised_object_decoders(
            {
                "fox": {
                    "matched": {
                        "lm_behavior": {"graph_id": "auto_beh_shared"},
                        "trace": [
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                        ],
                    }
                },
                "robot": {
                    "matched": {
                        "lm_behavior": {"graph_id": "auto_beh_shared"},
                        "trace": [
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                            {"learning_modules": {"lm_behavior": {"graph_id": "auto_beh_shared"}}},
                        ],
                    }
                },
            },
            lm_ids=("lm_behavior",),
            tail_steps=4,
        )

        self.assertNotIn("auto_beh_shared", decoders["lm_behavior"])
        self.assertEqual(
            diagnostics["lm_behavior"]["latent_id_support"]["auto_beh_shared"],
            {"fox": 5, "robot": 4},
        )
        self.assertEqual(
            diagnostics["lm_behavior"]["latent_id_candidate_objects"][
                "auto_beh_shared"
            ],
            ["fox", "robot"],
        )
        self.assertEqual(
            self._benchmark._decoder_graph_id_candidates(
                "auto_beh_shared",
                diagnostics["lm_behavior"],
            ),
            ["fox", "robot"],
        )

    def test_fit_self_supervised_joint_object_decoder_maps_child_signatures(self):
        decoder, diagnostics = (
            self._benchmark._fit_self_supervised_joint_object_decoder(
                {
                    "fox": {
                        "matched": {
                            "lm_morphology": {"graph_id": "auto_geom_shared"},
                            "lm_behavior": {"graph_id": "auto_beh_fox"},
                            "trace": [
                                {
                                    "learning_modules": {
                                        "lm_morphology": {
                                            "graph_id": "auto_geom_shared"
                                        },
                                        "lm_behavior": {
                                            "graph_id": "auto_beh_fox"
                                        },
                                    }
                                }
                            ],
                        }
                    },
                    "robot": {
                        "matched": {
                            "lm_morphology": {"graph_id": "auto_geom_robot"},
                            "lm_behavior": {"graph_id": "auto_beh_fox"},
                            "trace": [
                                {
                                    "learning_modules": {
                                        "lm_morphology": {
                                            "graph_id": "auto_geom_robot"
                                        },
                                        "lm_behavior": {
                                            "graph_id": "auto_beh_fox"
                                        },
                                    }
                                }
                            ],
                        }
                    },
                },
                tail_steps=1,
            )
        )

        fox_key = "lm_morphology=auto_geom_shared|lm_behavior=auto_beh_fox"
        robot_key = "lm_morphology=auto_geom_robot|lm_behavior=auto_beh_fox"

        self.assertEqual(decoder[fox_key], "fox")
        self.assertEqual(decoder[robot_key], "robot")
        self.assertEqual(
            diagnostics["signature_parts"][fox_key],
            {
                "lm_morphology": "auto_geom_shared",
                "lm_behavior": "auto_beh_fox",
            },
        )

    def test_condition_report_decodes_self_supervised_graph_ids(self):
        report = self._benchmark._condition_report(
            result={
                "lm_morphology": {"graph_id": "auto_geom_fox"},
                "lm_behavior": {"graph_id": "auto_beh_fox"},
                "lm_parent": {"graph_id": "auto_parent_fox"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="fox",
            target_lm_id="lm_behavior",
            object_decoders={
                "lm_morphology": {"auto_geom_fox": "fox"},
                "lm_behavior": {"auto_beh_fox": "fox"},
                "lm_parent": {"auto_parent_fox": "fox"},
            },
        )

        self.assertEqual(
            report["primary"]["final_prediction_latent_id"],
            "auto_beh_fox",
        )
        self.assertEqual(
            report["primary"]["final_prediction_graph_id"],
            "auto_beh_fox",
        )
        self.assertEqual(report["primary"]["final_prediction"], "fox")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertTrue(report["primary"]["final_prediction_decoded"])
        self.assertEqual(
            report["child_lm_primary"]["lm_morphology"]["final_prediction_latent_id"],
            report["child_lm_primary"]["lm_morphology"]["final_prediction_graph_id"],
        )
        self.assertTrue(
            report["child_lm_primary"]["lm_morphology"][
                "final_prediction_correct"
            ]
        )

    def test_condition_report_uses_joint_decoder_for_ambiguous_behavior_graph_id(self):
        report = self._benchmark._condition_report(
            result={
                "lm_morphology": {"graph_id": "auto_geom_fox"},
                "lm_behavior": {"graph_id": "auto_beh_shared"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="fox",
            target_lm_id="lm_behavior",
            object_decoders={
                "lm_behavior": {"auto_beh_shared": "robot"},
            },
            object_decoder_diagnostics={
                "lm_behavior": {
                    "graph_id_support": {
                        "auto_beh_shared": {"fox": 2, "robot": 2}
                    }
                }
            },
            joint_object_decoder={
                "lm_morphology=auto_geom_fox|lm_behavior=auto_beh_shared": "fox"
            },
        )

        self.assertEqual(report["primary"]["final_prediction"], "fox")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertEqual(
            report["primary"]["final_prediction_lm_id"],
            "lm_child_joint_decoder",
        )
        self.assertEqual(
            report["primary"]["final_prediction_latent_id"],
            "lm_morphology=auto_geom_fox|lm_behavior=auto_beh_shared",
        )
        self.assertEqual(
            report["child_lm_primary"]["lm_behavior"]["final_prediction"],
            "fox",
        )
        self.assertEqual(
            report["child_lm_primary"]["lm_behavior"]["final_prediction_lm_id"],
            "lm_child_joint_decoder",
        )
        self.assertEqual(
            report["primary"]["final_prediction_signature_latent_ids"],
            {
                "lm_morphology": "auto_geom_fox",
                "lm_behavior": "auto_beh_shared",
            },
        )
        self.assertEqual(
            report["primary"]["final_prediction_signature_graph_ids"],
            {
                "lm_morphology": "auto_geom_fox",
                "lm_behavior": "auto_beh_shared",
            },
        )

    def test_condition_report_keeps_ambiguous_behavior_graph_id_unresolved(self):
        report = self._benchmark._condition_report(
            result={
                "lm_morphology": {"graph_id": "auto_geom_fox"},
                "lm_behavior": {"graph_id": "auto_beh_shared"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="fox",
            target_lm_id="lm_behavior",
            object_decoders={
                "lm_behavior": {"auto_beh_shared": "robot"},
            },
            object_decoder_diagnostics={
                "lm_behavior": {
                    "graph_id_support": {
                        "auto_beh_shared": {"fox": 2, "robot": 2}
                    }
                }
            },
        )

        self.assertIsNone(report["primary"]["final_prediction"])
        self.assertIsNone(report["primary"]["final_prediction_correct"])
        self.assertFalse(report["primary"]["final_prediction_decoded"])
        self.assertTrue(report["primary"]["final_prediction_ambiguous"])
        self.assertFalse(report["primary"]["final_prediction_resolved"])
        self.assertEqual(
            report["primary"]["final_prediction_candidate_objects"],
            ["fox", "robot"],
        )
        self.assertEqual(
            report["primary"]["final_prediction_latent_id"],
            "auto_beh_shared",
        )
        self.assertEqual(
            report["primary"]["final_prediction_graph_id"],
            "auto_beh_shared",
        )

    def test_alias_summary_treats_shared_behavior_candidates_as_shared_not_foreign(self):
        diagnostics = {
            "lm_behavior": {
                "latent_id_support": {
                    "auto_beh_shared": {"fox": 6, "robot": 5}
                },
                "graph_id_support": {
                    "auto_beh_shared": {"fox": 6, "robot": 5}
                },
                "latent_id_candidate_objects": {
                    "auto_beh_shared": ["fox", "robot"]
                },
                "graph_id_candidate_objects": {
                    "auto_beh_shared": ["fox", "robot"]
                },
            }
        }
        summary = self._benchmark._summarize_lm_alias_source_for_model(
            "robot",
            matched_result={
                "lm_behavior": {"graph_id": "auto_beh_shared"},
                "trace": [
                    {
                        "learning_modules": {
                            "lm_behavior": {"graph_id": "auto_beh_shared"}
                        }
                    }
                ],
            },
            training_graph_id_to_target={
                "lm_behavior": {"auto_beh_shared": ["fox"]}
            },
            training_reporting_alias_diagnostics={},
            object_decoder_diagnostics=diagnostics,
            lm_ids=("lm_behavior",),
        )

        self.assertEqual(summary["lm_behavior"]["confusion_source"], "none")
        self.assertFalse(summary["lm_behavior"]["matched_eval_output_issue"])
        self.assertEqual(
            summary["lm_behavior"]["matched_eval"]["shared_target_graph_ids"],
            {"auto_beh_shared": 2},
        )
        self.assertEqual(
            summary["lm_behavior"]["matched_eval"]["foreign_graph_ids"],
            {},
        )
        self.assertTrue(
            summary["lm_behavior"]["training"]["shared_training_expected"]
        )

    def test_condition_report_uses_parent_fallback_when_joint_signature_is_ambiguous(self):
        report = self._benchmark._condition_report(
            result={
                "lm_morphology": {"graph_id": "auto_geom_shared"},
                "lm_behavior": {"graph_id": "auto_beh_shared"},
                "lm_parent": {"graph_id": "auto_parent_fox"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="fox",
            target_lm_id="lm_behavior",
            object_decoders={
                "lm_behavior": {"auto_beh_shared": "robot"},
                "lm_parent": {"auto_parent_fox": "fox"},
            },
            object_decoder_diagnostics={
                "lm_behavior": {
                    "graph_id_support": {
                        "auto_beh_shared": {"fox": 2, "robot": 2}
                    }
                },
                "lm_parent": {
                    "graph_id_support": {
                        "auto_parent_fox": {"fox": 4}
                    }
                },
            },
            joint_object_decoder={
                "lm_morphology=auto_geom_shared|lm_behavior=auto_beh_shared": "robot"
            },
            joint_object_decoder_diagnostics={
                "signature_support": {
                    "lm_morphology=auto_geom_shared|lm_behavior=auto_beh_shared": {
                        "fox": 2,
                        "robot": 2,
                    }
                }
            },
        )

        self.assertEqual(report["primary"]["final_prediction"], "fox")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertEqual(
            report["primary"]["final_prediction_graph_id"],
            "auto_parent_fox",
        )
        self.assertEqual(
            report["primary"]["final_prediction_lm_id"],
            "lm_parent_decoder_fallback",
        )

    def test_condition_report_prefers_joint_signature_when_parent_agrees(self):
        report = self._benchmark._condition_report(
            result={
                "lm_morphology": {"graph_id": "auto_geom_robot"},
                "lm_behavior": {"graph_id": "auto_beh_shared"},
                "lm_parent": {"graph_id": "auto_parent_robot"},
                "trace": [],
            },
            phase_targets=[],
            latent_decoder={},
            target_name="robot",
            target_lm_id="lm_behavior",
            object_decoders={
                "lm_behavior": {"auto_beh_shared": "cesiumman"},
                "lm_parent": {"auto_parent_robot": "robot"},
            },
            object_decoder_diagnostics={
                "lm_behavior": {
                    "graph_id_support": {
                        "auto_beh_shared": {"cesiumman": 6, "robot": 4}
                    }
                },
                "lm_parent": {
                    "graph_id_support": {
                        "auto_parent_robot": {"robot": 4}
                    }
                },
            },
            joint_object_decoder={
                "lm_morphology=auto_geom_robot|lm_behavior=auto_beh_shared": "robot"
            },
            joint_object_decoder_diagnostics={
                "signature_support": {
                    "lm_morphology=auto_geom_robot|lm_behavior=auto_beh_shared": {
                        "robot": 3
                    }
                }
            },
        )

        self.assertEqual(report["primary"]["final_prediction"], "robot")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertEqual(
            report["primary"]["final_prediction_lm_id"],
            "lm_child_joint_decoder",
        )
        self.assertEqual(
            report["child_lm_primary"]["lm_behavior"]["final_prediction"],
            "robot",
        )
        self.assertEqual(
            report["child_lm_primary"]["lm_behavior"]["final_prediction_lm_id"],
            "lm_child_joint_decoder",
        )

    def test_compute_interference_debug_localizes_final_flip(self):
        trace = [
            {
                "step": 0,
                "frame": 0,
                "learning_modules": {
                    "lm_behavior": {
                        "evidence_debug": {
                            "base_evidence": {"robot": 2.0, "fox": 1.0},
                            "after_temporal_behavior_evidence": {
                                "robot": 1.8,
                                "fox": 1.2,
                            },
                            "final_evidence": {"robot": 1.0, "fox": 2.3},
                            "winner_path": {
                                "base": "robot",
                                "after_temporal_behavior": "robot",
                                "final": "fox",
                            },
                            "temporal_behavior_adjustments": {"fox": 0.2},
                            "temporal_behavior_scores": {"robot": 0.6},
                            "self_supervised_adjustments": {"fox": 1.1},
                            "self_supervised_support": {
                                "state": {"fox": 0.7},
                            },
                            "temporal_labels": {"current": "latent_robot"},
                            "query_bias_norms": {"combined": 1.2},
                            "prediction_mismatch": 0.3,
                            "action_prediction_error": 0.1,
                        }
                    }
                },
            },
            {
                "step": 1,
                "frame": 1,
                "learning_modules": {
                    "lm_behavior": {
                        "evidence_debug": {
                            "base_evidence": {"robot": 2.2, "fox": 0.9},
                            "after_temporal_behavior_evidence": {
                                "robot": 2.1,
                                "fox": 1.0,
                            },
                            "final_evidence": {"robot": 2.0, "fox": 1.1},
                            "winner_path": {
                                "base": "robot",
                                "after_temporal_behavior": "robot",
                                "final": "robot",
                            },
                            "temporal_behavior_adjustments": {"robot": 0.1},
                            "temporal_behavior_scores": {"robot": 0.7},
                            "self_supervised_adjustments": {"robot": 0.1},
                            "self_supervised_support": {
                                "prediction": {"robot": 0.2},
                            },
                            "temporal_labels": {"current": "latent_robot"},
                            "query_bias_norms": {"combined": 1.0},
                            "prediction_mismatch": 0.2,
                            "action_prediction_error": 0.05,
                        }
                    }
                },
            },
        ]

        debug = self._benchmark._compute_interference_debug(trace, "robot")

        self.assertEqual(debug["steps_with_debug"], 2)
        self.assertEqual(debug["steps_with_evidence_debug"], 2)
        self.assertEqual(debug["skipped_step_count"], 0)
        self.assertEqual(debug["skip_reason_counts"], {})
        self.assertEqual(debug["base_top_label_counts"], {"robot": 2})
        self.assertEqual(
            debug["after_temporal_behavior_top_label_counts"],
            {"robot": 2},
        )
        self.assertEqual(debug["final_top_label_counts"], {"fox": 1, "robot": 1})
        self.assertEqual(debug["final_non_target_top_label_counts"], {"fox": 1})
        self.assertAlmostEqual(debug["base_target_win_fraction"], 1.0)
        self.assertAlmostEqual(
            debug["after_temporal_behavior_target_win_fraction"],
            1.0,
        )
        self.assertAlmostEqual(debug["final_target_win_fraction"], 0.5)
        self.assertAlmostEqual(debug["base_to_final_flip_fraction"], 0.5)
        self.assertAlmostEqual(debug["behavior_to_final_flip_fraction"], 0.5)
        self.assertAlmostEqual(
            debug["target_lost_after_temporal_behavior_fraction"],
            0.0,
        )
        self.assertAlmostEqual(
            debug["target_lost_after_self_supervised_fraction"],
            0.5,
        )
        self.assertEqual(debug["per_step"][0]["final_top_label"], "fox")

    def test_compute_interference_debug_surfaces_joint_chart_candidates(self):
        trace = [
            {
                "step": 0,
                "frame": 0,
                "learning_modules": {
                    "lm_behavior": {
                        "evidence_debug": {
                            "base_evidence": {"robot": 1.0},
                            "after_temporal_behavior_evidence": {"robot": 1.0},
                            "final_evidence": {"robot": 1.0},
                            "winner_path": {
                                "base": "robot",
                                "after_temporal_behavior": "robot",
                                "final": "robot",
                            },
                            "joint_hypothesis_candidates": [
                                {
                                    "latent_id": "robot",
                                    "object_id": "robot",
                                    "chart_id": "robot#chart1",
                                    "score": 1.4,
                                    "base_score": 1.0,
                                    "joint_delta": 0.4,
                                    "context_ranked_chart_prior": 0.2,
                                    "vote_ranked_chart_prior": 0.0,
                                    "selected": True,
                                },
                                {
                                    "latent_id": "robot",
                                    "object_id": "robot",
                                    "chart_id": "robot#chart0",
                                    "score": 1.1,
                                    "base_score": 1.0,
                                    "joint_delta": 0.1,
                                    "context_ranked_chart_prior": 0.0,
                                    "vote_ranked_chart_prior": 0.0,
                                    "selected": False,
                                },
                            ],
                            "query_bias_norms": {"combined": 1.0},
                            "prediction_mismatch": 0.1,
                            "action_prediction_error": 0.0,
                        }
                    }
                },
            }
            ,
            {
                "step": 1,
                "frame": 1,
                "learning_modules": {
                    "lm_behavior": {
                        "evidence_debug": {
                            "base_evidence": {"robot": 1.0},
                            "after_temporal_behavior_evidence": {"robot": 1.0},
                            "final_evidence": {"robot": 1.0},
                            "winner_path": {
                                "base": "robot",
                                "after_temporal_behavior": "robot",
                                "final": "robot",
                            },
                            "joint_hypothesis_candidates": [
                                {
                                    "latent_id": "robot",
                                    "object_id": "robot",
                                    "chart_id": "robot#chart0",
                                    "score": 1.5,
                                    "base_score": 1.0,
                                    "joint_delta": 0.5,
                                    "context_ranked_chart_prior": 0.0,
                                    "vote_ranked_chart_prior": 0.3,
                                    "selected": True,
                                },
                                {
                                    "latent_id": "robot",
                                    "object_id": "robot",
                                    "chart_id": "robot#chart1",
                                    "score": 1.2,
                                    "base_score": 1.0,
                                    "joint_delta": 0.2,
                                    "context_ranked_chart_prior": 0.0,
                                    "vote_ranked_chart_prior": 0.0,
                                    "selected": False,
                                },
                            ],
                            "query_bias_norms": {"combined": 1.0},
                            "prediction_mismatch": 0.1,
                            "action_prediction_error": 0.0,
                        }
                    }
                },
            },
        ]

        debug = self._benchmark._compute_interference_debug(trace, "robot")

        self.assertEqual(debug["steps_with_joint_hypothesis_candidates"], 2)
        self.assertAlmostEqual(debug["joint_hypothesis_candidate_fraction"], 1.0)
        self.assertEqual(
            debug["joint_top_chart_counts"],
            {"robot#chart0": 1, "robot#chart1": 1},
        )
        self.assertAlmostEqual(debug["joint_top_chart_resolved_fraction"], 1.0)
        self.assertAlmostEqual(
            debug["joint_top_latent_multi_chart_fraction"],
            1.0,
        )
        self.assertAlmostEqual(
            debug["joint_context_ranked_chart_prior_active_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            debug["joint_vote_ranked_chart_prior_active_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            debug["joint_top_context_ranked_chart_prior_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            debug["joint_top_vote_ranked_chart_prior_fraction"],
            0.5,
        )
        self.assertAlmostEqual(debug["joint_chart_switch_fraction"], 1.0)
        self.assertAlmostEqual(
            debug["joint_same_latent_chart_switch_fraction"],
            1.0,
        )
        self.assertAlmostEqual(debug["joint_dominant_chart_fraction"], 0.5)
        self.assertEqual(debug["per_step"][0]["joint_top_chart_id"], "robot#chart1")
        self.assertEqual(debug["per_step"][1]["joint_top_chart_id"], "robot#chart0")
        self.assertEqual(debug["per_step"][0]["joint_hypothesis_candidate_count"], 2)
        self.assertEqual(debug["per_step"][0]["joint_top_latent_candidate_count"], 2)
        self.assertEqual(debug["per_step"][1]["joint_hypothesis_candidate_count"], 2)
        self.assertEqual(debug["per_step"][1]["joint_top_latent_candidate_count"], 2)
        self.assertTrue(debug["per_step"][0]["joint_context_ranked_chart_prior_active"])
        self.assertTrue(debug["per_step"][1]["joint_vote_ranked_chart_prior_active"])
        self.assertEqual(
            debug["per_step"][0]["joint_top_latent_candidates"][0]["chart_id"],
            "robot#chart1",
        )
        self.assertEqual(
            debug["per_step"][1]["joint_top_latent_candidates"][0]["chart_id"],
            "robot#chart0",
        )

    def test_extract_trace_context_packet_returns_replayable_copy(self):
        active_cells = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        packet = self._benchmark._extract_trace_context_packet(
            {
                "learning_modules": {
                    "lm_behavior": {
                        "context_signal": {
                            "active_cells": active_cells,
                            "sender_id": "lm_behavior",
                            "graph_id": "auto_beh_fox",
                            "confidence": 0.75,
                            "sender_step_count": 4,
                        }
                    }
                }
            },
            "lm_behavior",
        )

        self.assertEqual(packet["sender_id"], "lm_behavior")
        self.assertEqual(packet["graph_id"], "auto_beh_fox")
        self.assertEqual(packet["confidence"], 0.75)
        self.assertEqual(packet["sender_step_count"], 4)
        self.assertEqual(packet["active_cells"].dtype, np.float32)
        np.testing.assert_array_equal(
            packet["active_cells"],
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )

        active_cells[0] = 99.0
        self.assertEqual(float(packet["active_cells"][0]), 1.0)

    def test_extract_trace_context_packet_falls_back_to_lm_trace_identity(self):
        packet = self._benchmark._extract_trace_context_packet(
            {
                "learning_modules": {
                    "lm_behavior": {
                        "graph_id": "auto_beh_robot",
                        "evidence": 0.9,
                        "context_signal": {
                            "active_cells": [1.0, 0.0, 0.5],
                            "sender_id": "lm_behavior",
                            "sender_step_count": 7,
                        },
                    }
                }
            },
            "lm_behavior",
        )

        self.assertEqual(packet["graph_id"], "auto_beh_robot")
        self.assertEqual(packet["confidence"], 0.9)
        self.assertEqual(packet["sender_step_count"], 7)
        np.testing.assert_array_equal(
            packet["active_cells"],
            np.array([1.0, 0.0, 0.5], dtype=np.float32),
        )

    def test_extract_trace_context_packet_preserves_output_state_geometry(self):
        packet = self._benchmark._extract_trace_context_packet(
            {
                "learning_modules": {
                    "lm_behavior": {
                        "graph_id": "auto_beh_robot",
                        "context_signal": {
                            "active_cells": [0.0, 1.0, 0.5],
                            "sender_id": "lm_behavior",
                            "sender_step_count": 3,
                        },
                        "output_state": {
                            "location": [1.0, 2.0, 3.0],
                            "pose_vectors": [
                                [1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [0.0, 1.0, 0.0],
                            ],
                            "confidence": 0.8,
                            "graph_id": "auto_beh_robot",
                        },
                    }
                }
            },
            "lm_behavior",
        )

        np.testing.assert_array_equal(
            packet["location"],
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            packet["pose_vectors"],
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(packet["confidence"], 0.8)

    def test_summarize_pairwise_benchmark_keeps_child_and_parent_matched_views(self):
        summary = self._benchmark._summarize_pairwise_benchmark(
            {
                "models": ["fox", "robot"],
                "aggregate": {
                    "matched_top1_accuracy_mean": 0.5,
                    "matched_top1_resolved_fraction_mean": 0.5,
                    "matched_top1_strict_accuracy_mean": 0.25,
                    "morphology_matched_top1_accuracy_mean": 1.0,
                    "morphology_matched_top1_resolved_fraction_mean": 1.0,
                    "morphology_matched_top1_strict_accuracy_mean": 1.0,
                    "behavior_matched_top1_accuracy_mean": 0.5,
                    "behavior_matched_top1_resolved_fraction_mean": 0.5,
                    "behavior_matched_top1_strict_accuracy_mean": 0.25,
                    "parent_matched_top1_accuracy_mean": 0.0,
                    "parent_matched_top1_resolved_fraction_mean": 1.0,
                    "parent_matched_top1_strict_accuracy_mean": 0.0,
                    "matched_joint_hypothesis_candidate_fraction_mean": 1.0,
                    "matched_joint_top_latent_multi_chart_fraction_mean": 0.75,
                    "matched_joint_same_latent_chart_switch_fraction_mean": 0.25,
                },
                "per_model": {
                    "fox": {
                        "matched": {
                            "primary": {
                                "final_prediction": "robot",
                                "final_prediction_graph_id": "auto_parent_robot",
                                "final_prediction_correct": False,
                            },
                            "child_lm_primary": {
                                "lm_morphology": {
                                    "final_prediction": "fox",
                                    "final_prediction_graph_id": "auto_geom_fox",
                                    "final_prediction_correct": True,
                                },
                                "lm_behavior": {
                                    "final_prediction": "robot",
                                    "final_prediction_graph_id": "auto_beh_robot",
                                    "final_prediction_correct": False,
                                },
                            },
                            "child_interference_debug": {
                                "lm_morphology": {
                                    "base_target_win_fraction": 1.0,
                                    "final_target_win_fraction": 1.0,
                                    "final_top_label_counts": {"fox": 4},
                                },
                                "lm_behavior": {
                                    "base_target_win_fraction": 1.0,
                                    "final_target_win_fraction": 0.0,
                                    "joint_hypothesis_candidate_fraction": 1.0,
                                    "joint_top_latent_multi_chart_fraction": 0.75,
                                    "joint_same_latent_chart_switch_fraction": 0.25,
                                    "joint_top_chart_counts": {
                                        "robot#chart0": 2,
                                        "robot#chart1": 2,
                                    },
                                    "final_top_label_counts": {"robot": 4},
                                },
                            },
                            "lm_object_predictions": {
                                "lm_parent": {
                                    "final_prediction": "robot",
                                    "final_prediction_graph_id": "auto_parent_robot",
                                    "final_prediction_correct": False,
                                }
                            },
                        }
                    }
                },
            }
        )

        self.assertEqual(summary["models"], ["fox", "robot"])
        self.assertEqual(summary["aggregate"]["matched_top1_accuracy_mean"], 0.5)
        self.assertEqual(
            summary["aggregate"]["matched_top1_resolved_fraction_mean"],
            0.5,
        )
        self.assertEqual(
            summary["aggregate"]["matched_top1_strict_accuracy_mean"],
            0.25,
        )
        self.assertEqual(
            summary["aggregate"]["matched_joint_hypothesis_candidate_fraction_mean"],
            1.0,
        )
        self.assertEqual(
            summary["aggregate"]["matched_joint_top_latent_multi_chart_fraction_mean"],
            0.75,
        )
        self.assertTrue(
            summary["per_model"]["fox"]["matched"]["lm_morphology"][
                "final_prediction_correct"
            ]
        )
        self.assertFalse(
            summary["per_model"]["fox"]["matched"]["lm_behavior"][
                "final_prediction_correct"
            ]
        )
        self.assertEqual(
            summary["per_model"]["fox"]["matched"]["lm_behavior"][
                "final_top_label_counts"
            ],
            {"robot": 4},
        )
        self.assertEqual(
            summary["per_model"]["fox"]["matched"]["lm_behavior"][
                "joint_top_chart_counts"
            ],
            {"robot#chart0": 2, "robot#chart1": 2},
        )
        self.assertEqual(
            summary["per_model"]["fox"]["matched"]["lm_behavior"][
                "joint_same_latent_chart_switch_fraction"
            ],
            0.25,
        )
        self.assertEqual(
            summary["per_model"]["fox"]["matched"]["lm_parent"][
                "final_prediction_graph_id"
            ],
            "auto_parent_robot",
        )

    def test_prediction_accuracy_summary_tracks_resolved_and_strict_accuracy(self):
        summary = self._benchmark._prediction_accuracy_summary(
            [
                {
                    "final_prediction_correct": True,
                    "final_prediction_resolved": True,
                },
                {
                    "final_prediction_correct": None,
                    "final_prediction_resolved": False,
                },
            ]
        )

        self.assertEqual(summary["accuracy_mean"], 1.0)
        self.assertEqual(summary["resolved_fraction_mean"], 0.5)
        self.assertEqual(summary["strict_accuracy_mean"], 0.5)

    def test_run_diagnostic_harness_adds_pairwise_diagnostics(self):
        base_report = {
            "models": ["fox", "robot"],
            "diagnostics": {
                "parent_context_replay": {"aggregate": {"ok": 1.0}}
            },
        }
        pairwise = {
            "pair_count": 1,
            "pairs": {"fox__robot": {"models": ["fox", "robot"]}},
            "focus_pairs": {},
        }

        with mock.patch.object(
            self._benchmark,
            "run_benchmark",
            return_value=base_report,
        ) as run_benchmark, mock.patch.object(
            self._benchmark,
            "_run_pairwise_child_diagnostics",
            return_value=pairwise,
        ) as run_pairwise:
            report = self._benchmark.run_diagnostic_harness(
                models=["fox", "robot"],
                phase_bins=4,
                train_cycles=1,
                eval_cycles=1,
                stretch=2,
                compression_stride=2,
                resolution=(16, 16),
                column_kwargs={"seed": 42},
                lm_family="predictive_hypothesis_torch",
                detail_grid_shape=(2, 2),
                use_detail_aware_sensors=True,
            )

        self.assertIs(report, base_report)
        self.assertEqual(
            report["diagnostics"]["pairwise_child_only"],
            pairwise,
        )
        self.assertTrue(
            run_benchmark.call_args.kwargs["include_parent_replay_diagnostics"]
        )
        self.assertEqual(run_pairwise.call_args.kwargs["models"], ["fox", "robot"])
        self.assertEqual(
            run_benchmark.call_args.kwargs["lm_family"],
            "predictive_hypothesis_torch",
        )
        self.assertEqual(
            run_pairwise.call_args.kwargs["lm_family"],
            "predictive_hypothesis_torch",
        )
        self.assertEqual(
            run_benchmark.call_args.kwargs["detail_grid_shape"],
            (2, 2),
        )
        self.assertEqual(
            run_pairwise.call_args.kwargs["detail_grid_shape"],
            (2, 2),
        )
        self.assertTrue(
            run_benchmark.call_args.kwargs["use_detail_aware_sensors"]
        )
        self.assertTrue(
            run_pairwise.call_args.kwargs["use_detail_aware_sensors"]
        )

    def test_main_uses_diagnostic_harness_flag(self):
        with mock.patch.object(
            self._benchmark,
            "run_diagnostic_harness",
            return_value={"mode": "diagnostic"},
        ) as run_diagnostic_harness, mock.patch.object(
            self._benchmark,
            "run_benchmark",
            return_value={"mode": "baseline"},
        ) as run_benchmark, mock.patch(
            "sys.argv",
            [
                str(SCRIPT_PATH),
                "--models",
                "fox",
                "--diagnostic-harness",
            ],
        ), mock.patch("builtins.print") as mocked_print:
            self._benchmark.main()

        run_diagnostic_harness.assert_called_once()
        run_benchmark.assert_not_called()
        mocked_print.assert_called_once()

    def test_track14_single_model_smoke_report(self):
        with _real_asset_benchmark_lock():
            report = _run_track14_benchmark_cli(["fox"])

        self.assertEqual(report["models"], ["fox"])
        self.assertEqual(report["lm_family"], "predictive_hypothesis_torch")
        self.assertEqual(report["requested_resolution"], [16, 16])
        self.assertEqual(report["resolution"], [24, 24])
        self.assertTrue(report["real_assets_only"])
        self.assertEqual(
            report["core_temporal_state_mode"],
            "predictive_hypothesis_message_state",
        )
        self.assertEqual(
            report["input_geometry_mode"],
            "detail_packet_plus_internal_object_state",
        )

        fox_report = report["per_model"]["fox"]
        self.assertEqual(fox_report["initial_distance"], 2.0)
        self.assertEqual(
            fox_report["matched"]["primary"]["final_prediction"],
            "fox",
        )
        self.assertEqual(
            fox_report["matched"]["primary"]["final_prediction_lm_id"],
            "lm_behavior",
        )
        self.assertEqual(
            fox_report["matched"]["child_lm_primary"]["lm_morphology"][
                "final_prediction"
            ],
            "fox",
        )
        self.assertEqual(
            fox_report["matched"]["child_lm_primary"]["lm_behavior"][
                "final_prediction"
            ],
            "fox",
        )
        self.assertGreater(
            fox_report["matched"]["interference_debug"]["steps_with_evidence_debug"],
            0,
        )
        self.assertGreater(
            fox_report["matched"]["primary"]["temporal_boundary_pressure_mean"],
            0.0,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_boundary_pressure_delta_mean"],
            -2e-3,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_surprise_delta_mean"],
            -2e-3,
        )

    def test_track14_pairwise_smoke_report(self):
        with _real_asset_benchmark_lock():
            report = _run_track14_benchmark_cli(
                ["fox", "robot"],
                max_attempts=3,
                accept_report=_track14_resolved_primary_contract,
            )

        self.assertEqual(report["models"], ["fox", "robot"])
        self.assertEqual(report["lm_family"], "predictive_hypothesis_torch")
        self.assertEqual(report["requested_resolution"], [16, 16])
        self.assertEqual(report["resolution"], [24, 24])
        self.assertEqual(report["aggregate"]["matched_top1_accuracy_mean"], 1.0)
        self.assertEqual(
            report["aggregate"]["matched_top1_resolved_fraction_mean"],
            1.0,
        )
        parent_accuracy = report["aggregate"]["parent_matched_top1_accuracy_mean"]
        if parent_accuracy is not None:
            self.assertGreaterEqual(parent_accuracy, 0.0)
            self.assertLessEqual(parent_accuracy, 1.0)
        self.assertGreater(
            report["aggregate"]["perturbation_boundary_pressure_delta_mean"],
            -2e-2,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_surprise_delta_mean"],
            -2e-2,
        )
        self.assertGreater(
            report["aggregate"]["action_blind_boundary_pressure_delta_mean"],
            -2e-3,
        )

        expected_distances = {"fox": 2.0, "robot": 3.0}
        for model_name in ["fox", "robot"]:
            model_report = report["per_model"][model_name]
            self.assertEqual(
                model_report["initial_distance"],
                expected_distances[model_name],
            )
            self.assertEqual(
                model_report["matched"]["primary"]["final_prediction"],
                model_name,
            )
            self.assertTrue(
                model_report["matched"]["primary"]["final_prediction_correct"]
            )
            self.assertGreater(
                model_report["matched"]["interference_debug"][
                    "steps_with_evidence_debug"
                ],
                0,
            )
            self.assertGreater(
                model_report["matched"]["primary"][
                    "temporal_boundary_pressure_mean"
                ],
                0.0,
            )

    def test_track14_pairwise_morphology_smoke_report(self):
        with _real_asset_benchmark_lock():
            report = _run_track14_benchmark_cli(
                ["fox", "cesiumman"],
                max_attempts=3,
                accept_report=_track14_resolved_primary_contract,
            )

        self.assertEqual(report["models"], ["fox", "cesiumman"])
        self.assertEqual(report["lm_family"], "predictive_hypothesis_torch")
        self.assertEqual(report["requested_resolution"], [16, 16])
        self.assertEqual(report["resolution"], [24, 24])
        self.assertEqual(report["aggregate"]["matched_top1_accuracy_mean"], 1.0)
        self.assertEqual(
            report["aggregate"]["matched_top1_resolved_fraction_mean"],
            1.0,
        )
        parent_accuracy = report["aggregate"]["parent_matched_top1_accuracy_mean"]
        if parent_accuracy is not None:
            self.assertGreaterEqual(parent_accuracy, 0.0)
            self.assertLessEqual(parent_accuracy, 1.0)
        self.assertGreater(
            report["aggregate"]["perturbation_boundary_pressure_delta_mean"],
            -3e-2,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_surprise_delta_mean"],
            -3e-2,
        )

        expected_distances = {"fox": 2.0, "cesiumman": 3.0}
        for model_name in ["fox", "cesiumman"]:
            model_report = report["per_model"][model_name]
            self.assertEqual(
                model_report["initial_distance"],
                expected_distances[model_name],
            )
            self.assertEqual(
                model_report["matched"]["primary"]["final_prediction"],
                model_name,
            )
            self.assertTrue(
                model_report["matched"]["primary"]["final_prediction_correct"]
            )
            self.assertGreater(
                model_report["matched"]["interference_debug"][
                    "steps_with_evidence_debug"
                ],
                0,
            )
            self.assertGreater(
                model_report["matched"]["primary"][
                    "temporal_boundary_pressure_mean"
                ],
                0.0,
            )

    def test_track14_three_model_smoke_report(self):
        with _real_asset_benchmark_lock():
            report = _run_track14_benchmark_cli(
                ["fox", "cesiumman", "robot"],
                max_attempts=1,
                accept_report=_track14_resolved_primary_contract,
                model_orders=[
                    ["fox", "cesiumman", "robot"],
                    ["robot", "fox", "cesiumman"],
                ],
            )

        self.assertEqual(
            sorted(report["models"]),
            ["cesiumman", "fox", "robot"],
        )
        self.assertEqual(report["lm_family"], "predictive_hypothesis_torch")
        self.assertEqual(report["requested_resolution"], [16, 16])
        self.assertEqual(report["resolution"], [24, 24])
        self.assertEqual(report["aggregate"]["matched_top1_accuracy_mean"], 1.0)
        self.assertEqual(
            report["aggregate"]["matched_top1_resolved_fraction_mean"],
            1.0,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_boundary_pressure_delta_mean"],
            -4e-2,
        )
        self.assertGreater(
            report["aggregate"]["perturbation_surprise_delta_mean"],
            -4e-2,
        )

        for model_name in ["fox", "cesiumman", "robot"]:
            model_report = report["per_model"][model_name]
            self.assertEqual(
                model_report["matched"]["primary"]["final_prediction"],
                model_name,
            )
            self.assertIn(
                "final_prediction_latent_id",
                model_report["matched"]["primary"],
            )
            self.assertEqual(
                model_report["matched"]["primary"]["final_prediction_latent_id"],
                model_report["matched"]["primary"]["final_prediction_graph_id"],
            )
            self.assertTrue(
                model_report["matched"]["primary"]["final_prediction_correct"]
            )
            self.assertGreater(
                model_report["matched"]["interference_debug"][
                    "steps_with_evidence_debug"
                ],
                0,
            )
            self.assertGreater(
                model_report["matched"]["lm_trace_coverage"]["lm_behavior"][
                    "evidence_debug_evidence_steps"
                ],
                0,
            )

        known_object_ids = set(
            report["per_model"]["robot"]["training"]["known_object_ids"]
        )
        self.assertGreaterEqual(len(known_object_ids), 1)
        self.assertTrue(
            all(object_id.startswith("latent_object_") for object_id in known_object_ids)
        )

    def test_track14_diagnostic_harness_smoke_report(self):
        with _real_asset_benchmark_lock():
            report = _run_track14_benchmark_cli(
                ["fox", "robot"],
                diagnostic_harness=True,
            )

        self.assertEqual(report["models"], ["fox", "robot"])
        self.assertEqual(report["lm_family"], "predictive_hypothesis_torch")
        self.assertEqual(report["requested_resolution"], [16, 16])
        self.assertEqual(report["resolution"], [24, 24])

        diagnostics = report["diagnostics"]
        self.assertIn("parent_context_replay", diagnostics)
        self.assertIn("pairwise_child_only", diagnostics)

        parent_context_replay = diagnostics["parent_context_replay"]
        self.assertTrue(parent_context_replay["supported"])
        self.assertEqual(
            parent_context_replay["lm_family"],
            "predictive_hypothesis_torch",
        )
        self.assertTrue(parent_context_replay["aggregate"]["supported"])
        self.assertIn("fox", parent_context_replay["per_model"])
        self.assertIsNotNone(
            parent_context_replay["aggregate"]["final_prediction_matches_live_mean"]
        )
        self.assertGreaterEqual(
            parent_context_replay["aggregate"]["final_prediction_matches_live_mean"],
            0.0,
        )
        self.assertLessEqual(
            parent_context_replay["aggregate"]["final_prediction_matches_live_mean"],
            1.0,
        )
        self.assertIsNotNone(
            parent_context_replay["aggregate"][
                "final_prediction_latent_id_matches_live_mean"
            ]
        )
        self.assertIsNotNone(
            parent_context_replay["aggregate"][
                "final_prediction_graph_id_matches_live_mean"
            ]
        )
        self.assertIn(
            "joint_top_latent_multi_chart_fraction_delta_vs_live_mean",
            parent_context_replay["aggregate"],
        )
        self.assertIsNotNone(
            parent_context_replay["per_model"]["fox"]["replayed_parent"]
        )
        self.assertGreater(
            parent_context_replay["per_model"]["fox"]["replayed_parent"][
                "replayable_steps"
            ],
            0,
        )
        self.assertIn(
            "final_prediction_matches_live",
            parent_context_replay["per_model"]["fox"]["vs_live"],
        )
        self.assertIn(
            "final_prediction_latent_id_matches_live",
            parent_context_replay["per_model"]["fox"]["vs_live"],
        )
        self.assertIn(
            "joint_top_latent_multi_chart_fraction_delta_vs_live",
            parent_context_replay["per_model"]["fox"]["vs_live"],
        )
        self.assertIn(
            "final_prediction_matches_live",
            parent_context_replay["per_model"]["robot"]["vs_live"],
        )
        self.assertIn(
            "final_prediction_latent_id_matches_live",
            parent_context_replay["per_model"]["robot"]["vs_live"],
        )
        self.assertIn(
            "joint_top_latent_multi_chart_fraction_delta_vs_live",
            parent_context_replay["per_model"]["robot"]["vs_live"],
        )

        pairwise = diagnostics["pairwise_child_only"]
        self.assertEqual(pairwise["pair_count"], 1)
        self.assertEqual(sorted(pairwise["pairs"].keys()), ["fox__robot"])
        self.assertIn("behavior", pairwise["focus_pairs"])

        behavior_focus = pairwise["focus_pairs"]["behavior"]
        self.assertEqual(behavior_focus["models"], ["fox", "robot"])
        self.assertIsNotNone(
            behavior_focus["aggregate"]["matched_top1_accuracy_mean"]
        )
        self.assertGreaterEqual(
            behavior_focus["aggregate"]["matched_top1_accuracy_mean"],
            0.0,
        )
        self.assertLessEqual(
            behavior_focus["aggregate"]["matched_top1_accuracy_mean"],
            1.0,
        )
        for model_name in ["fox", "robot"]:
            model_report = behavior_focus["per_model"][model_name]["matched"]
            self.assertIn("final_prediction", model_report["primary"])
            self.assertIn("final_prediction", model_report["lm_morphology"])
            self.assertIn("final_prediction", model_report["lm_behavior"])
            self.assertIn("final_prediction", model_report["lm_parent"])
            self.assertIn("final_prediction_latent_id", model_report["primary"])
            self.assertIn(
                "final_prediction_latent_id",
                model_report["lm_morphology"],
            )
            self.assertIn(
                "final_prediction_latent_id",
                model_report["lm_behavior"],
            )
            self.assertIn("final_prediction_latent_id", model_report["lm_parent"])
            self.assertEqual(
                model_report["primary"]["final_prediction_latent_id"],
                model_report["primary"]["final_prediction_graph_id"],
            )
            self.assertEqual(
                model_report["lm_morphology"]["final_prediction_latent_id"],
                model_report["lm_morphology"]["final_prediction_graph_id"],
            )
            self.assertEqual(
                model_report["lm_behavior"]["final_prediction_latent_id"],
                model_report["lm_behavior"]["final_prediction_graph_id"],
            )
            self.assertEqual(
                model_report["lm_parent"]["final_prediction_latent_id"],
                model_report["lm_parent"]["final_prediction_graph_id"],
            )
            self.assertIsInstance(
                model_report["primary"]["final_prediction"],
                str,
            )
            self.assertIsInstance(
                model_report["lm_morphology"]["final_prediction"],
                str,
            )
            self.assertIsInstance(
                model_report["lm_behavior"]["final_prediction"],
                str,
            )
            self.assertIsInstance(
                model_report["lm_parent"]["final_prediction"],
                str,
            )

    def test_single_model_smoke_report(self):
        report = self._benchmark.run_benchmark(
            models=["fox"],
            phase_bins=4,
            train_cycles=1,
            eval_cycles=1,
            stretch=2,
            compression_stride=2,
            resolution=(16, 16),
            column_kwargs={"n_minicolumns": 128, "sparsity": 0.05, "seed": 42},
        )

        self.assertEqual(report["models"], ["fox"])
        self.assertTrue(report["real_assets_only"])
        self.assertEqual(
            report["boundary_active_threshold"],
            self._benchmark.BOUNDARY_PRESSURE_ACTIVE_THRESHOLD,
        )
        self.assertEqual(report["benchmark_seed"], 42)
        self.assertEqual(report["core_temporal_state_mode"], "trace_bank_d_t")
        self.assertEqual(
            report["input_geometry_mode"],
            "inferred_relative_hidden_state",
        )
        self.assertTrue(report["trace_biased_inference"])
        self.assertTrue(report["child_lm_temporal_parity"])
        self.assertEqual(
            report["action_context_falsifier_mode"],
            "forced_action_replay_with_efference_ablation",
        )
        self.assertFalse(report["full_active_passive_test"])
        self.assertEqual(
            report["temporal_learning_mode"],
            "self_supervised_predictive_trace_state",
        )
        self.assertFalse(report["temporal_state_provider_used"])
        self.assertFalse(report["phase_labels_used_in_training"])
        self.assertEqual(
            report["motor_control_mode"],
            "lm_parent_goal_state_driven_with_exploratory_fallback",
        )
        self.assertEqual(
            report["goal_state_authority"],
            ["lm_parent"],
        )
        self.assertTrue(report["goal_state_driven_motor"])
        self.assertTrue(report["parent_goal_location_memory_enabled"])
        self.assertTrue(report["lateral_voting_enabled"])
        self.assertEqual(
            report["lateral_voting_mode"],
            "conditional_settle_gated_top_k",
        )
        self.assertEqual(report["vote_confident_threshold"], 2)
        self.assertEqual(report["vote_cooldown_steps"], 10000)
        self.assertTrue(report["action_sampler_fallback_enabled"])
        self.assertIn("move_backward", report["motor_action_space"])
        self.assertIn("move_left", report["motor_action_space"])
        self.assertIn("move_right", report["motor_action_space"])
        self.assertIn("move_up", report["motor_action_space"])
        self.assertIn("move_down", report["motor_action_space"])
        self.assertIn("trace_weight", report["temporal_trace_config"])
        self.assertEqual(report["column_kwargs"]["seed"], 42)
        self.assertFalse(report["fully_self_supervised_lms"])
        self.assertEqual(
            report["object_identity_training_mode"],
            "named_object_graph_training",
        )
        self.assertIn("lm_morphology", report["child_lm_track12_matrix"])
        self.assertIn("lm_behavior", report["child_lm_track12_matrix"])
        self.assertEqual(
            report["child_lm_track12_matrix"]["lm_morphology"],
            report["child_lm_track12_matrix"]["lm_behavior"],
        )
        self.assertTrue(
            report["child_lm_track12_matrix"]["lm_morphology"][
                "temporal_trace_enabled"
            ]
        )
        self.assertTrue(
            report["child_lm_track12_matrix"]["lm_morphology"][
                "action_predictive_enabled"
            ]
        )
        self.assertTrue(
            report["child_lm_track12_matrix"]["lm_morphology"][
                "inferred_state_enabled"
            ]
        )
        self.assertEqual(
            report["child_lm_track12_matrix"]["lm_morphology"]["vote_top_k"],
            3,
        )

        self.assertIn("fox", report["per_model"])
        fox_report = report["per_model"]["fox"]
        self.assertEqual(
            fox_report["matched"]["primary"]["final_prediction_lm_id"],
            "lm_behavior",
        )
        self.assertIn(
            "final_prediction_latent_id",
            fox_report["matched"]["primary"],
        )
        self.assertIn(
            "final_prediction_graph_id",
            fox_report["matched"]["primary"],
        )
        self.assertEqual(
            fox_report["matched"]["primary"]["final_prediction_latent_id"],
            fox_report["matched"]["primary"]["final_prediction_graph_id"],
        )
        self.assertIn("lm_object_predictions", fox_report["matched"])

        self.assertIn("matched", fox_report)
        self.assertIn("training", fox_report)
        self.assertIn("child_lm_primary", fox_report["matched"])
        self.assertIn("interference_debug", fox_report["matched"])
        self.assertIn("lm_trace_coverage", fox_report["matched"])
        self.assertIn("interference_debug_best_lm_id", fox_report["matched"])
        self.assertIn(
            "interference_debug_best_available",
            fox_report["matched"],
        )
        self.assertIn(
            "lm_morphology",
            fox_report["matched"]["child_lm_primary"],
        )
        self.assertIn(
            "lm_behavior",
            fox_report["matched"]["child_lm_primary"],
        )
        self.assertEqual(
            fox_report["matched"]["child_lm_primary"]["lm_morphology"][
                "trace_steps"
            ],
            fox_report["matched"]["child_lm_primary"]["lm_behavior"][
                "trace_steps"
            ],
        )
        self.assertIn("known_state_count", fox_report["training"])
        self.assertIn("state_object_totals", fox_report["training"])
        self.assertIn("trace_coverage", fox_report["training"])
        self.assertIn("matched_action_stats", fox_report)
        self.assertIn("active_replay", fox_report)
        self.assertIn("action_blind_replay", fox_report)
        self.assertIn("stretched", fox_report)
        self.assertIn("compressed", fox_report)
        self.assertIn("omission", fox_report)
        self.assertIn("perturbation", fox_report)

        self.assertIn("vs_matched", fox_report["active_replay"])
        self.assertIn("primary_deltas", fox_report["active_replay"]["vs_matched"])
        self.assertIn(
            "trace_disagreement",
            fox_report["action_blind_replay"]["vs_active_replay"],
        )

        omission_windowed = fox_report["omission"]["windowed_primary"]
        perturb_windowed = fox_report["perturbation"]["windowed_primary"]

        self.assertIn("per_window", omission_windowed)
        self.assertIn("per_window", perturb_windowed)
        self.assertGreaterEqual(len(omission_windowed["per_window"]), 1)
        self.assertGreaterEqual(len(perturb_windowed["per_window"]), 1)

        first_omission = omission_windowed["per_window"][0]
        self.assertIn("pre_condition", first_omission)
        self.assertIn("condition", first_omission)
        self.assertIn("recovery_condition", first_omission)
        self.assertIn("recovery_steps_to_confident", first_omission)
        self.assertIn("mean_boundary_pressure", first_omission["condition"])
        self.assertIn("boundary_active_fraction", first_omission["condition"])

        omission_primary = fox_report["omission"]["primary"]
        self.assertIn("temporal_boundary_pressure_mean", omission_primary)
        self.assertIn("temporal_boundary_active_fraction", omission_primary)
        self.assertIn(
            "base_target_win_fraction",
            fox_report["matched"]["interference_debug"],
        )
        self.assertIn(
            "steps_with_evidence_debug",
            fox_report["matched"]["interference_debug"],
        )
        self.assertIn(
            "skip_reason_counts",
            fox_report["matched"]["interference_debug"],
        )
        self.assertIn(
            "target_lost_after_self_supervised_fraction",
            fox_report["matched"]["interference_debug"],
        )

        self.assertIn("aggregate", report)
        self.assertIn("omission_surprise_delta_mean", report["aggregate"])
        self.assertIn("perturbation_surprise_delta_mean", report["aggregate"])
        self.assertIn("omission_boundary_pressure_delta_mean", report["aggregate"])
        self.assertIn("perturbation_boundary_pressure_delta_mean", report["aggregate"])
        self.assertIn("action_blind_surprise_delta_mean", report["aggregate"])
        self.assertIn(
            "action_blind_current_label_disagreement_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_action_context_nonzero_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_base_target_win_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_target_lost_after_self_supervised_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_joint_top_latent_multi_chart_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_joint_same_latent_chart_switch_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "matched_joint_top_context_ranked_chart_prior_fraction_mean",
            report["aggregate"],
        )
        self.assertIn(
            "morphology_matched_top1_accuracy_mean",
            report["aggregate"],
        )
        self.assertIn(
            "parent_matched_top1_accuracy_mean",
            report["aggregate"],
        )

    def test_single_model_fully_self_supervised_smoke_report(self):
        report = self._benchmark.run_benchmark(
            models=["fox"],
            phase_bins=4,
            train_cycles=1,
            eval_cycles=1,
            stretch=2,
            compression_stride=2,
            resolution=(16, 16),
            column_kwargs={"n_minicolumns": 128, "sparsity": 0.05, "seed": 42},
            fully_self_supervised_lms=True,
            object_decoder_tail_steps=3,
        )

        self.assertTrue(report["fully_self_supervised_lms"])
        self.assertEqual(
            report["object_identity_training_mode"],
            "self_supervised_auto_label_posthoc_decoded",
        )
        self.assertEqual(report["object_identity_decoder_tail_steps"], 3)
        self.assertNotIn("defer_sensor_auto_label", report["column_kwargs"])
        self.assertTrue(
            report["morphology_column_kwargs"]["defer_sensor_auto_label"]
        )
        self.assertTrue(
            report["behavior_column_kwargs"]["defer_sensor_auto_label"]
        )
        self.assertEqual(
            report["object_identity_disambiguation_mode"],
            "joint_child_signature_then_parent_fallback_on_ambiguous_graph_id",
        )
        self.assertEqual(
            report["object_identity_ambiguity_reporting_mode"],
            "ambiguous_graph_ids_remain_unresolved_and_are_excluded_from_accuracy",
        )
        self.assertIn("lm_behavior", report["object_identity_decoder"])
        self.assertIn(
            "latent_id_to_object",
            report["object_identity_decoder"]["lm_behavior"],
        )
        self.assertIn(
            "latent_id_support",
            report["object_identity_decoder"]["lm_behavior"],
        )
        self.assertIn("signature_to_object", report["joint_object_identity_decoder"])
        self.assertIn("fox", report["per_model"])

        fox_report = report["per_model"]["fox"]
        self.assertIn("object_identity_decoder", fox_report)
        self.assertIn(
            "final_prediction_latent_id",
            fox_report["matched"]["primary"],
        )
        self.assertIn(
            "final_prediction_graph_id",
            fox_report["matched"]["primary"],
        )
        self.assertEqual(
            fox_report["matched"]["primary"]["final_prediction_latent_id"],
            fox_report["matched"]["primary"]["final_prediction_graph_id"],
        )
        self.assertIn("lm_object_predictions", fox_report["matched"])
        for prediction in fox_report["matched"]["lm_object_predictions"].values():
            self.assertIn("final_prediction", prediction)
            self.assertIn("final_prediction_latent_id", prediction)
            self.assertIn("final_prediction_graph_id", prediction)
            self.assertEqual(
                prediction["final_prediction_latent_id"],
                prediction["final_prediction_graph_id"],
            )
            self.assertIn("final_prediction_correct", prediction)
            self.assertIn("final_prediction_ambiguous", prediction)
            self.assertIn("final_prediction_resolved", prediction)

    def test_pairwise_self_supervised_smoke_report_surfaces_vote_chart_priors(self):
        report = self._benchmark.run_benchmark(
            models=["fox", "robot"],
            phase_bins=4,
            train_cycles=1,
            eval_cycles=1,
            stretch=2,
            compression_stride=2,
            resolution=(16, 16),
            column_kwargs={"n_minicolumns": 128, "sparsity": 0.05, "seed": 42},
            fully_self_supervised_lms=True,
            object_decoder_tail_steps=3,
            lm_family="predictive_hypothesis_torch",
            detail_grid_shape=(2, 2),
            use_detail_aware_sensors=True,
        )

        self.assertGreater(
            report["per_model"]["fox"]["matched"]["interference_debug"][
                "joint_vote_ranked_chart_prior_active_fraction"
            ],
            0.0,
        )
        self.assertGreater(
            report["per_model"]["robot"]["matched"]["interference_debug"][
                "joint_vote_ranked_chart_prior_active_fraction"
            ],
            0.0,
        )

    def test_triad_self_supervised_smoke_report_surfaces_top_vote_chart_priors(self):
        report = self._benchmark.run_benchmark(
            models=["fox", "cesiumman", "robot"],
            phase_bins=4,
            train_cycles=1,
            eval_cycles=1,
            stretch=2,
            compression_stride=2,
            resolution=(16, 16),
            column_kwargs={"n_minicolumns": 128, "sparsity": 0.05, "seed": 42},
            fully_self_supervised_lms=True,
            object_decoder_tail_steps=3,
            lm_family="predictive_hypothesis_torch",
            detail_grid_shape=(2, 2),
            use_detail_aware_sensors=True,
        )

        self.assertGreater(
            report["aggregate"][
                "matched_joint_top_vote_ranked_chart_prior_fraction_mean"
            ],
            0.0,
        )
        self.assertTrue(
            any(
                model_report["matched"]["interference_debug"][
                    "joint_top_vote_ranked_chart_prior_fraction"
                ]
                > 0.0
                for model_report in report["per_model"].values()
            )
        )
