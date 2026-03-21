#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.utils.category_metrics import (  # noqa: E402
    load_category_taxonomy,
    summarize_category_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize category-level metrics from eval_stats.csv")
    parser.add_argument("--eval-stats", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    eval_stats = pd.read_csv(args.eval_stats)
    taxonomy = load_category_taxonomy(args.taxonomy)
    summary = summarize_category_eval(eval_stats, taxonomy)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.output is not None:
        args.output.write_text(payload)

    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())