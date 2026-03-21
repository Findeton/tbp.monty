# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Language bridge: maps Monty's internal representations to words.

T3.1: Object naming — lookup from graph IDs to human-readable words.
T3.2: Category naming — aggregate evidence by category and name the winner.

This is a grounding bridge, not a language model. It provides the minimal
mapping layer between Monty's object/category representations and natural
language tokens.

Usage:
    bridge = LanguageBridge(
        object_names={"mug_1": "mug", "bowl_3": "bowl"},
        category_names={"cup": "cup", "utensil": "utensil"},
        category_taxonomy={"mug_1": "cup", "bowl_3": "cup"},
    )

    # T3.1: Name the recognized object
    name = bridge.name_object("mug_1")  # -> "mug"

    # T3.2: Name the best category from evidence
    cat_name = bridge.name_category({"mug_1": 5.2, "bowl_3": 3.1, "fork_2": 0.8})
    # -> "cup" (because cup category has 5.2 + 3.1 = 8.3 total evidence)
"""

from collections import defaultdict
from typing import Dict, Optional, Tuple


class LanguageBridge:
    """Maps Monty object IDs and categories to human-readable words.

    Attributes:
        object_names: Maps graph_id -> human-readable object name.
        category_names: Maps category_id -> human-readable category name.
        category_taxonomy: Maps graph_id -> category_id.
    """

    def __init__(
        self,
        object_names: Optional[Dict[str, str]] = None,
        category_names: Optional[Dict[str, str]] = None,
        category_taxonomy: Optional[Dict[str, str]] = None,
    ):
        self.object_names = object_names or {}
        self.category_names = category_names or {}
        self.category_taxonomy = category_taxonomy or {}

        # Auto-generate missing object names from graph IDs
        # e.g. "Alphabet_of_the_Magi_3" -> "Alphabet of the Magi 3"
        for graph_id in self.category_taxonomy:
            if graph_id not in self.object_names:
                self.object_names[graph_id] = graph_id.replace("_", " ")

        # Auto-generate missing category names from category IDs
        for cat_id in set(self.category_taxonomy.values()):
            if cat_id not in self.category_names:
                self.category_names[cat_id] = cat_id.replace("_", " ")

    def name_object(self, graph_id: str) -> str:
        """T3.1: Return human-readable name for a recognized object.

        Falls back to the raw graph_id if no mapping exists.
        """
        return self.object_names.get(graph_id, graph_id)

    def name_category(
        self, evidence_per_graph: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """T3.2: Aggregate evidence by category and name the winning category.

        Args:
            evidence_per_graph: Maps graph_id -> max evidence for that graph.

        Returns:
            Tuple of (category_name, total_evidence, all_category_evidence).
            Returns ("unknown", 0.0, {}) if no taxonomy or no positive evidence.
        """
        if not self.category_taxonomy or not evidence_per_graph:
            return ("unknown", 0.0, {})

        cat_evidence = defaultdict(float)
        for graph_id, ev in evidence_per_graph.items():
            cat_id = self.category_taxonomy.get(graph_id)
            if cat_id is not None and ev > 0:
                cat_evidence[cat_id] += ev

        if not cat_evidence:
            return ("unknown", 0.0, {})

        best_cat = max(cat_evidence, key=cat_evidence.get)
        best_ev = cat_evidence[best_cat]
        cat_name = self.category_names.get(best_cat, best_cat)

        return (cat_name, best_ev, dict(cat_evidence))

    def describe_recognition(
        self,
        graph_id: str,
        evidence: float,
        evidence_per_graph: Optional[Dict[str, float]] = None,
    ) -> str:
        """Produce a human-readable sentence describing what was recognized.

        Args:
            graph_id: The winning graph ID from the LM.
            evidence: The evidence score for the winning graph.
            evidence_per_graph: Optional per-graph evidence for category naming.

        Returns:
            A natural language description of the recognition result.
        """
        obj_name = self.name_object(graph_id)
        parts = [f'Recognized "{obj_name}" (evidence: {evidence:.1f})']

        if evidence_per_graph and self.category_taxonomy:
            cat_name, cat_ev, cat_evidence = self.name_category(evidence_per_graph)
            if cat_name != "unknown":
                obj_cat = self.category_taxonomy.get(graph_id)
                obj_cat_name = self.category_names.get(obj_cat, obj_cat)
                if obj_cat_name != cat_name:
                    parts.append(
                        f'. Category evidence suggests "{cat_name}" '
                        f"(total: {cat_ev:.1f}) over "
                        f'instance category "{obj_cat_name}"'
                    )
                else:
                    parts.append(f'. Category: "{cat_name}" (total: {cat_ev:.1f})')

        return "".join(parts)
