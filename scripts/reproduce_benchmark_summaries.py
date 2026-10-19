#!/usr/bin/env python3
"""Recompute headline BinderRanker benchmark summaries from public TSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_row(
    rows: list[dict[str, str]], **conditions: str | float
) -> dict[str, str]:
    matches = []
    for row in rows:
        matched = True
        for key, expected in conditions.items():
            actual = row[key]
            if isinstance(expected, float):
                matched = matched and abs(float(actual) - expected) < 1e-12
            else:
                matched = matched and actual == expected
        if matched:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository / "data" / "benchmark",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    high_recall = args.data_root / "high_recall"
    pruning = read_tsv(high_recall / "macro_pruning_recall.tsv")
    fixed = read_tsv(high_recall / "macro_fixed_recall_operating_points.tsv")
    runtime = read_tsv(args.data_root / "runtime" / "macro_binderranker_af2_burden.tsv")

    recall_40 = {}
    recall_50 = {}
    for label in ("medium", "loose"):
        recall_40[label] = float(
            find_row(pruning, label_definition=label, pruned_fraction=0.4)[
                "macro_recall"
            ]
        )
        recall_50[label] = float(
            find_row(pruning, label_definition=label, pruned_fraction=0.5)[
                "macro_recall"
            ]
        )

    pruning_at_95 = {}
    pruning_at_99 = {}
    for label in ("medium", "loose"):
        pruning_at_95[label] = float(
            find_row(fixed, label_definition=label, target_recall=0.95)[
                "mean_pruned_fraction"
            ]
        )
        pruning_at_99[label] = float(
            find_row(fixed, label_definition=label, target_recall=0.99)[
                "mean_pruned_fraction"
            ]
        )

    report = {
        "schema_version": "1",
        "interpretation": "AF2-derived computational surrogate endpoints, not wet-lab success rates",
        "held_out_target_count": 5,
        "candidate_count": 25000,
        "macro_recall_at_40_percent_pruning": recall_40,
        "macro_recall_at_50_percent_pruning": recall_50,
        "mean_pruned_fraction_at_95_percent_within_target_recall": pruning_at_95,
        "mean_pruned_fraction_at_99_percent_within_target_recall": pruning_at_99,
        "runtime_rows": runtime,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
