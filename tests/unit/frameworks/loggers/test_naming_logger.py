# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

from tbp.monty.frameworks.loggers.naming_logger import NamingLogger


class _FakeLM:
    def __init__(self, mlh, evidence_ids=None, evidence_scores=None):
        self._mlh = dict(mlh)
        self._evidence_ids = list(evidence_ids or [])
        self._evidence_scores = list(evidence_scores or [])

    def get_current_mlh(self):
        return dict(self._mlh)

    def get_evidence_for_each_graph(self):
        return list(self._evidence_ids), list(self._evidence_scores)


class _FakeModel:
    def __init__(self, learning_modules):
        self.learning_modules = list(learning_modules)


class TestNamingLogger(unittest.TestCase):
    def test_post_episode_prefers_latent_id_and_keeps_graph_alias(self):
        logger = NamingLogger(
            handlers=[],
            object_names={"latent_object_0": "latent fox"},
            category_names={"animal": "animals"},
            category_taxonomy={"latent_object_0": "animal"},
            print_output=False,
        )
        model = _FakeModel(
            [
                _FakeLM(
                    {
                        "latent_id": "latent_object_0",
                        "graph_id": "legacy_fox_graph",
                        "evidence": 3.5,
                    },
                    evidence_ids=["latent_object_0"],
                    evidence_scores=[3.5],
                )
            ]
        )

        logger.post_episode(
            logger_args={"target": {"object": "latent_object_0"}},
            output_dir=".",
            model=model,
        )

        entry = logger.log[0]
        result = entry["lm_results"][0]
        self.assertEqual(entry["target_name"], "latent fox")
        self.assertEqual(result["latent_id"], "latent_object_0")
        self.assertEqual(result["graph_id"], "legacy_fox_graph")
        self.assertEqual(result["object_name"], "latent fox")
        self.assertEqual(result["category_name"], "animals")
        self.assertIn("latent fox", result["description"])


if __name__ == "__main__":
    unittest.main()