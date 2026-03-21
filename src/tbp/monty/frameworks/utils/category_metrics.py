from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_category_taxonomy(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Category taxonomy must be a mapping.")

    object_to_category = payload.get("object_to_category", payload)
    if not isinstance(object_to_category, dict) or not object_to_category:
        raise ValueError("Category taxonomy must define a non-empty object_to_category mapping.")

    return {
        "object_to_category": object_to_category,
        "train_objects": list(payload.get("train_objects", [])),
        "holdout_objects": list(payload.get("holdout_objects", [])),
    }


def summarize_category_eval(
    eval_stats: pd.DataFrame,
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    if "primary_target_object" not in eval_stats.columns:
        raise ValueError("eval_stats must contain primary_target_object.")
    if "most_likely_object" not in eval_stats.columns:
        raise ValueError("eval_stats must contain most_likely_object.")

    object_to_category = taxonomy["object_to_category"]
    missing_targets = sorted(
        set(eval_stats["primary_target_object"].dropna()) - set(object_to_category)
    )
    if missing_targets:
        raise ValueError(
            f"Targets missing from taxonomy: {', '.join(map(str, missing_targets))}"
        )

    target_category = eval_stats["primary_target_object"].map(object_to_category)
    predicted_category = eval_stats["most_likely_object"].map(object_to_category)

    predicted_object = eval_stats["most_likely_object"]
    predicted_present = predicted_object.notna()
    exact_match = predicted_object.eq(eval_stats["primary_target_object"]) & predicted_present
    category_match = predicted_category.eq(target_category) & predicted_present
    within_category_confusion = category_match & ~exact_match
    cross_category_confusion = predicted_present & ~category_match
    no_prediction = ~predicted_present

    per_category: dict[str, dict[str, Any]] = {}
    for category in sorted(target_category.dropna().unique()):
        category_rows = target_category == category
        count = int(category_rows.sum())
        per_category[category] = {
            "rows": count,
            "exact_accuracy": float(exact_match[category_rows].mean() * 100.0),
            "category_accuracy": float(category_match[category_rows].mean() * 100.0),
            "within_category_confusion": int(within_category_confusion[category_rows].sum()),
            "cross_category_confusion": int(cross_category_confusion[category_rows].sum()),
            "no_prediction": int(no_prediction[category_rows].sum()),
        }

    summary = {
        "rows": int(len(eval_stats)),
        "exact_accuracy": float(exact_match.mean() * 100.0),
        "category_accuracy": float(category_match.mean() * 100.0),
        "within_category_confusion_rate": float(within_category_confusion.mean() * 100.0),
        "cross_category_confusion_rate": float(cross_category_confusion.mean() * 100.0),
        "no_prediction_rate": float(no_prediction.mean() * 100.0),
        "performance_counts": {
            str(key): int(value)
            for key, value in eval_stats["primary_performance"].value_counts(dropna=False).to_dict().items()
        }
        if "primary_performance" in eval_stats.columns
        else {},
        "per_category": per_category,
        "train_objects": taxonomy.get("train_objects", []),
        "holdout_objects": taxonomy.get("holdout_objects", []),
    }

    return summary