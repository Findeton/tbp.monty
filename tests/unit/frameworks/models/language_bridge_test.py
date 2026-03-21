# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the LanguageBridge (T3.1-T3.2)."""

import unittest

from tbp.monty.frameworks.models.language_bridge import LanguageBridge


class TestObjectNaming(unittest.TestCase):
    """T3.1: Object naming tests."""

    def setUp(self):
        self.bridge = LanguageBridge(
            object_names={"mug_1": "red mug", "bowl_3": "large bowl"},
            category_taxonomy={"mug_1": "cup", "bowl_3": "cup", "fork_2": "utensil"},
        )

    def test_name_known_object(self):
        self.assertEqual(self.bridge.name_object("mug_1"), "red mug")
        self.assertEqual(self.bridge.name_object("bowl_3"), "large bowl")

    def test_name_unknown_object_falls_back_to_id(self):
        self.assertEqual(self.bridge.name_object("mystery_99"), "mystery_99")

    def test_auto_generated_name_from_taxonomy(self):
        # fork_2 is in taxonomy but not in object_names -> auto-generated
        self.assertEqual(self.bridge.name_object("fork_2"), "fork 2")

    def test_empty_bridge(self):
        bridge = LanguageBridge()
        self.assertEqual(bridge.name_object("anything"), "anything")


class TestCategoryNaming(unittest.TestCase):
    """T3.2: Category naming tests."""

    def setUp(self):
        self.bridge = LanguageBridge(
            category_names={"cup": "cups", "utensil": "utensils"},
            category_taxonomy={
                "mug_1": "cup",
                "bowl_3": "cup",
                "fork_2": "utensil",
                "spoon_1": "utensil",
            },
        )

    def test_category_aggregation(self):
        evidence = {"mug_1": 5.0, "bowl_3": 3.0, "fork_2": 2.0, "spoon_1": 1.0}
        cat_name, cat_ev, cat_evidence = self.bridge.name_category(evidence)
        self.assertEqual(cat_name, "cups")  # cup: 5+3=8 > utensil: 2+1=3
        self.assertAlmostEqual(cat_ev, 8.0)
        self.assertAlmostEqual(cat_evidence["cup"], 8.0)
        self.assertAlmostEqual(cat_evidence["utensil"], 3.0)

    def test_negative_evidence_ignored(self):
        evidence = {"mug_1": -5.0, "fork_2": 2.0}
        cat_name, cat_ev, _ = self.bridge.name_category(evidence)
        self.assertEqual(cat_name, "utensils")  # only positive evidence counts

    def test_empty_evidence(self):
        cat_name, cat_ev, _ = self.bridge.name_category({})
        self.assertEqual(cat_name, "unknown")
        self.assertEqual(cat_ev, 0.0)

    def test_no_taxonomy(self):
        bridge = LanguageBridge()
        cat_name, cat_ev, _ = bridge.name_category({"mug_1": 5.0})
        self.assertEqual(cat_name, "unknown")

    def test_unknown_graph_ids_skipped(self):
        evidence = {"unknown_99": 10.0, "fork_2": 2.0}
        cat_name, _, _ = self.bridge.name_category(evidence)
        self.assertEqual(cat_name, "utensils")


class TestDescribeRecognition(unittest.TestCase):
    """Test the full description output."""

    def setUp(self):
        self.bridge = LanguageBridge(
            object_names={"mug_1": "red mug"},
            category_names={"cup": "cups"},
            category_taxonomy={"mug_1": "cup", "bowl_3": "cup", "fork_2": "utensil"},
        )

    def test_basic_description(self):
        desc = self.bridge.describe_recognition("mug_1", 5.0)
        self.assertIn("red mug", desc)
        self.assertIn("5.0", desc)

    def test_description_with_category(self):
        evidence = {"mug_1": 5.0, "bowl_3": 3.0}
        desc = self.bridge.describe_recognition("mug_1", 5.0, evidence)
        self.assertIn("red mug", desc)
        self.assertIn("cups", desc)

    def test_description_category_disagrees(self):
        # fork_2 is utensil, but evidence says cup category wins
        evidence = {"mug_1": 5.0, "bowl_3": 3.0, "fork_2": 1.0}
        desc = self.bridge.describe_recognition("fork_2", 1.0, evidence)
        self.assertIn("cups", desc)  # category winner
        self.assertIn("utensil", desc)  # instance category


class TestOmniglotNaming(unittest.TestCase):
    """Test with Omniglot-style naming."""

    def setUp(self):
        self.bridge = LanguageBridge(
            category_taxonomy={
                "Alphabet_of_the_Magi_1": "Alphabet_of_the_Magi",
                "Alphabet_of_the_Magi_2": "Alphabet_of_the_Magi",
                "Anglo-Saxon_Futhorc_1": "Anglo-Saxon_Futhorc",
                "Anglo-Saxon_Futhorc_2": "Anglo-Saxon_Futhorc",
            },
        )

    def test_auto_generated_names(self):
        name = self.bridge.name_object("Alphabet_of_the_Magi_1")
        self.assertEqual(name, "Alphabet of the Magi 1")

    def test_auto_generated_category_names(self):
        evidence = {
            "Alphabet_of_the_Magi_1": 5.0,
            "Alphabet_of_the_Magi_2": 3.0,
            "Anglo-Saxon_Futhorc_1": 2.0,
            "Anglo-Saxon_Futhorc_2": 1.0,
        }
        cat_name, _, _ = self.bridge.name_category(evidence)
        self.assertEqual(cat_name, "Alphabet of the Magi")


if __name__ == "__main__":
    unittest.main()
