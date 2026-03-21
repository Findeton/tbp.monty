from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_compositional_taxonomy(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Compositional taxonomy must be a mapping.")

    object_to_base = payload.get("object_to_base")
    object_to_logo = payload.get("object_to_logo")
    if not isinstance(object_to_base, dict) or not object_to_base:
        raise ValueError("Compositional taxonomy must define a non-empty object_to_base mapping.")
    if not isinstance(object_to_logo, dict) or not object_to_logo:
        raise ValueError("Compositional taxonomy must define a non-empty object_to_logo mapping.")

    return {
        "object_to_base": object_to_base,
        "object_to_logo": object_to_logo,
        "train_objects": list(payload.get("train_objects", [])),
        "holdout_objects": list(payload.get("holdout_objects", [])),
    }


def summarize_compositional_eval(
    eval_stats: pd.DataFrame,
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    if "primary_target_object" not in eval_stats.columns:
        raise ValueError("eval_stats must contain primary_target_object.")
    if "most_likely_object" not in eval_stats.columns:
        raise ValueError("eval_stats must contain most_likely_object.")

    object_to_base = taxonomy["object_to_base"]
    object_to_logo = taxonomy["object_to_logo"]

    required_targets = set(eval_stats["primary_target_object"].dropna())
    missing_base_targets = sorted(required_targets - set(object_to_base))
    if missing_base_targets:
        raise ValueError(
            f"Targets missing from object_to_base mapping: {', '.join(map(str, missing_base_targets))}"
        )
    missing_logo_targets = sorted(required_targets - set(object_to_logo))
    if missing_logo_targets:
        raise ValueError(
            f"Targets missing from object_to_logo mapping: {', '.join(map(str, missing_logo_targets))}"
        )

    predicted_present = eval_stats["most_likely_object"].notna()
    target_base = eval_stats["primary_target_object"].map(object_to_base)
    target_logo = eval_stats["primary_target_object"].map(object_to_logo)
    predicted_base = eval_stats["most_likely_object"].map(object_to_base)
    predicted_logo = eval_stats["most_likely_object"].map(object_to_logo)

    exact_match = eval_stats["most_likely_object"].eq(eval_stats["primary_target_object"]) & predicted_present
    base_match = target_base.eq(predicted_base) & predicted_present
    logo_match = target_logo.eq(predicted_logo) & predicted_present
    both_parts_match = base_match & logo_match
    base_only_confusion = base_match & ~logo_match
    logo_only_confusion = logo_match & ~base_match
    cross_part_confusion = predicted_present & ~base_match & ~logo_match
    no_prediction = ~predicted_present

    per_base: dict[str, dict[str, Any]] = {}
    for base_name in sorted(target_base.dropna().unique()):
        base_rows = target_base == base_name
        per_base[base_name] = {
            "rows": int(base_rows.sum()),
            "exact_accuracy": float(exact_match[base_rows].mean() * 100.0),
            "base_accuracy": float(base_match[base_rows].mean() * 100.0),
            "logo_accuracy": float(logo_match[base_rows].mean() * 100.0),
            "both_parts_accuracy": float(both_parts_match[base_rows].mean() * 100.0),
            "base_only_confusion": int(base_only_confusion[base_rows].sum()),
            "logo_only_confusion": int(logo_only_confusion[base_rows].sum()),
            "cross_part_confusion": int(cross_part_confusion[base_rows].sum()),
            "no_prediction": int(no_prediction[base_rows].sum()),
        }

    return {
        "rows": int(len(eval_stats)),
        "exact_accuracy": float(exact_match.mean() * 100.0),
        "base_accuracy": float(base_match.mean() * 100.0),
        "logo_accuracy": float(logo_match.mean() * 100.0),
        "both_parts_accuracy": float(both_parts_match.mean() * 100.0),
        "base_only_confusion_rate": float(base_only_confusion.mean() * 100.0),
        "logo_only_confusion_rate": float(logo_only_confusion.mean() * 100.0),
        "cross_part_confusion_rate": float(cross_part_confusion.mean() * 100.0),
        "no_prediction_rate": float(no_prediction.mean() * 100.0),
        "performance_counts": {
            str(key): int(value)
            for key, value in eval_stats["primary_performance"].value_counts(dropna=False).to_dict().items()
        }
        if "primary_performance" in eval_stats.columns
        else {},
        "per_base": per_base,
        "train_objects": taxonomy.get("train_objects", []),
        "holdout_objects": taxonomy.get("holdout_objects", []),
    }