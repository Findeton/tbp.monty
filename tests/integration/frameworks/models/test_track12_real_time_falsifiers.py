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

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
            diagnostics["lm_parent"]["graph_id_to_object"]["auto_parent_fox"],
            "fox",
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
            report["primary"]["final_prediction_graph_id"],
            "auto_beh_fox",
        )
        self.assertEqual(report["primary"]["final_prediction"], "fox")
        self.assertTrue(report["primary"]["final_prediction_correct"])
        self.assertTrue(report["primary"]["final_prediction_decoded"])
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
            report["primary"]["final_prediction_signature_graph_ids"],
            {
                "lm_morphology": "auto_geom_fox",
                "lm_behavior": "auto_beh_shared",
            },
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

    def test_summarize_pairwise_benchmark_keeps_child_and_parent_matched_views(self):
        summary = self._benchmark._summarize_pairwise_benchmark(
            {
                "models": ["fox", "robot"],
                "aggregate": {
                    "matched_top1_accuracy_mean": 0.5,
                    "morphology_matched_top1_accuracy_mean": 1.0,
                    "behavior_matched_top1_accuracy_mean": 0.5,
                    "parent_matched_top1_accuracy_mean": 0.0,
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
            summary["per_model"]["fox"]["matched"]["lm_parent"][
                "final_prediction_graph_id"
            ],
            "auto_parent_robot",
        )

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
            "final_prediction_graph_id",
            fox_report["matched"]["primary"],
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
        self.assertIn("lm_behavior", report["object_identity_decoder"])
        self.assertIn("signature_to_object", report["joint_object_identity_decoder"])
        self.assertIn("fox", report["per_model"])

        fox_report = report["per_model"]["fox"]
        self.assertIn("object_identity_decoder", fox_report)
        self.assertIn(
            "final_prediction_graph_id",
            fox_report["matched"]["primary"],
        )
        self.assertIn("lm_object_predictions", fox_report["matched"])
        for prediction in fox_report["matched"]["lm_object_predictions"].values():
            self.assertIn("final_prediction", prediction)
            self.assertIn("final_prediction_graph_id", prediction)
            self.assertIn("final_prediction_correct", prediction)
