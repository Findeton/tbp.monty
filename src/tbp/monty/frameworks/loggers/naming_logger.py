# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Naming logger: outputs human-readable recognition results per episode.

Implements T3.1 (object naming) and T3.2 (category naming) by attaching to
the existing logging framework as a BaseMontyLogger. After each episode, it
reads the LM recognition results and maps them to words via LanguageBridge.

Output is written to a JSONL file (one line per episode) and optionally
printed to stdout.

Usage in Hydra config:
    See experiment/phase2_omniglot_scaled6_naming_eval_v2.yaml for an example.
    The NamingLogger is registered in the model's LOGGING_REGISTRY and
    activated when monty_log_level is set to include naming output.

Alternatively, use standalone:
    from tbp.monty.frameworks.loggers.naming_logger import NamingLogger
    logger = NamingLogger(handlers=[], bridge=bridge, print_output=True)
    logger.post_episode(logger_args, output_dir, model)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from typing_extensions import override

from tbp.monty.frameworks.loggers.exp_logger import BaseMontyLogger
from tbp.monty.frameworks.models.language_bridge import LanguageBridge

module_logger = logging.getLogger(__name__)


class NamingLogger(BaseMontyLogger):
    """Logger that outputs named recognition results per episode.

    After each evaluation episode, reads the most likely hypothesis from
    each LM and produces human-readable output using a LanguageBridge.

    Attributes:
        bridge: LanguageBridge instance for object/category name lookup.
        print_output: If True, print naming results to stdout.
        log: List of naming results (one dict per episode).
    """

    def __init__(
        self,
        handlers,
        object_names: Optional[Dict[str, str]] = None,
        category_names: Optional[Dict[str, str]] = None,
        category_taxonomy: Optional[Dict[str, str]] = None,
        print_output: bool = True,
    ):
        """Initialize NamingLogger.

        Args:
            handlers: List of output handlers (passed to BaseMontyLogger).
            object_names: Maps latent_id or graph_id -> human-readable object name.
            category_names: Maps category_id -> human-readable category name.
            category_taxonomy: Maps latent_id or graph_id -> category_id.
            print_output: Whether to print results to stdout.
        """
        super().__init__(handlers)
        self.bridge = LanguageBridge(
            object_names=object_names,
            category_names=category_names,
            category_taxonomy=category_taxonomy,
        )
        self.print_output = print_output
        self.log = []

    @override
    def post_episode(self, logger_args, output_dir, model):
        """Read LM results and produce named output."""
        episode_results = []

        for i, lm in enumerate(model.learning_modules):
            mlh = lm.get_current_mlh()
            latent_id = mlh.get("latent_id", mlh.get("graph_id", "unknown"))
            graph_id = mlh.get("graph_id", latent_id)
            evidence = mlh.get("evidence", 0.0)

            # Get per-graph evidence for category naming (T3.2)
            evidence_per_graph = {}
            if hasattr(lm, "get_evidence_for_each_graph"):
                try:
                    graph_ids, graph_evidences = lm.get_evidence_for_each_graph()
                    if len(graph_ids) > 0:
                        evidence_per_graph = {
                            gid: float(ev)
                            for gid, ev in zip(graph_ids, graph_evidences)
                        }
                except (IndexError, AttributeError):
                    pass

            # T3.1: Object naming
            obj_name = self.bridge.name_object(mlh)

            # T3.2: Category naming
            cat_name, cat_ev, cat_evidence = self.bridge.name_category(
                evidence_per_graph
            )

            # Full description
            description = self.bridge.describe_recognition(
                mlh, evidence, evidence_per_graph
            )

            result = {
                "lm_id": f"LM_{i}",
                "latent_id": latent_id,
                "graph_id": graph_id,
                "object_name": obj_name,
                "evidence": float(evidence),
                "category_name": cat_name,
                "category_evidence": float(cat_ev),
                "description": description,
            }
            episode_results.append(result)

            if self.print_output:
                print(f"  [LM_{i}] {description}")

        # Get target info if available
        target = logger_args.get("target", {})
        target_obj = target.get("object", "unknown") if target else "unknown"

        entry = {
            "target": target_obj,
            "target_name": self.bridge.name_object(target_obj),
            "lm_results": episode_results,
        }
        self.log.append(entry)

    @override
    def close(self, logger_args, output_dir, model):
        """Write naming results to JSONL file."""
        if not self.log:
            return

        outfile = Path(output_dir) / "naming_results.jsonl"
        with outfile.open("w") as f:
            for entry in self.log:
                f.write(json.dumps(entry, default=str) + "\n")

        module_logger.info(f"Naming results written to {outfile}")

        # Also write summary
        total = len(self.log)
        correct_name = sum(
            1
            for e in self.log
            if any(
                r["object_name"] == e["target_name"] for r in e["lm_results"]
            )
        )
        correct_cat = sum(
            1
            for e in self.log
            if any(
                r["category_name"]
                == self.bridge.category_names.get(
                    self.bridge.category_taxonomy.get(e["target"], ""), ""
                )
                for r in e["lm_results"]
            )
        )

        summary = {
            "total_episodes": total,
            "correct_object_name": correct_name,
            "correct_category_name": correct_cat,
            "object_name_accuracy": round(100 * correct_name / total, 1)
            if total
            else 0,
            "category_name_accuracy": round(100 * correct_cat / total, 1)
            if total
            else 0,
        }

        if self.print_output:
            print(f"\n=== Naming Summary ===")
            print(f"  Object naming:   {correct_name}/{total} "
                  f"({summary['object_name_accuracy']}%)")
            print(f"  Category naming: {correct_cat}/{total} "
                  f"({summary['category_name_accuracy']}%)")

        summary_file = Path(output_dir) / "naming_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)

        for handler in self.handlers:
            handler.close()
