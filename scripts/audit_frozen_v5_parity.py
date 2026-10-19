#!/usr/bin/env python3
"""Audit migrated frozen-v5 scoring against five complete historical pools."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from binderranker.geometry.contact_field import FIELDNAMES
from binderranker.scoring.frozen_v5 import score_pool


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_frozen_score(rows: list[dict[str, str]], reference_dir: Path):
    v4 = load_module("binderranker_parity_frozen_v4", reference_dir / "rank_candidates_v4_layered.py")
    v5 = load_module("binderranker_parity_frozen_v5", reference_dir / "rank_candidates_v5_proto.py")
    output = [dict(row) for row in rows]
    v4.add_derived_metrics(output)
    morphology = v4.infer_target_morphology(output, alpha=1.25, min_mode_weight=0.05)
    v4.compute_scores(output, morphology, q_low=0.05, q_high=0.95)
    thresholds = v4.compute_filter_thresholds(output)
    v4.apply_layered_filters(output, thresholds)
    v5.add_v5_scores(
        rows=output,
        morph=morphology,
        v4=v4,
        beta_local=0.0,
        local_lambda=1.0,
        gain_plane=1.0,
        gain_compact=1.0,
        morph_alpha=1.25,
        min_mode_weight=0.05,
        q_low=0.05,
        q_high=0.95,
        extent_variant="coverage_cb_micro",
        micro_weight=0.0,
        extent_weight=0.90,
        safety_weight=0.0,
        roughness_weight=0.0,
    )
    v4.add_ranks(
        output,
        [
            "final_score_v4",
            "morphology_adaptive_score",
            "rank_consensus_score",
            "score_line",
            "score_plane",
            "score_compact",
            "score_roughness",
            "score_microfit",
            "score_safety",
        ],
    )
    v5.add_v5_ranks(output, v4)
    return output


def parse_number(value: Any) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compare(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    fields: list[str],
    *,
    detail_limit: int = 100,
) -> dict[str, Any]:
    actual_by_id = {str(row["pdb_name"]): row for row in actual}
    field_stats: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    missing_rows = []
    numerical_cells = 0
    exact_numerical_differences = 0
    tolerance_exceedances = 0
    text_cells = 0
    text_mismatches = 0
    rank_mismatches = 0
    max_absolute = 0.0
    max_relative = 0.0
    for expected_row in expected:
        candidate = str(expected_row["pdb_name"])
        actual_row = actual_by_id.get(candidate)
        if actual_row is None:
            missing_rows.append(candidate)
            continue
        for field in fields:
            expected_number = parse_number(expected_row.get(field, ""))
            actual_number = parse_number(actual_row.get(field, ""))
            if expected_number is not None and actual_number is not None:
                numerical_cells += 1
                delta = actual_number - expected_number
                absolute = abs(delta)
                relative = absolute / max(abs(expected_number), 1e-300)
                stats = field_stats.setdefault(
                    field,
                    {
                        "numerical_cells": 0,
                        "exact_difference_count": 0,
                        "tolerance_exceedance_count": 0,
                        "max_absolute_difference": 0.0,
                        "max_relative_difference": 0.0,
                    },
                )
                stats["numerical_cells"] += 1
                stats["max_absolute_difference"] = max(
                    stats["max_absolute_difference"], absolute
                )
                stats["max_relative_difference"] = max(
                    stats["max_relative_difference"], relative
                )
                max_absolute = max(max_absolute, absolute)
                max_relative = max(max_relative, relative)
                if absolute != 0.0:
                    exact_numerical_differences += 1
                    stats["exact_difference_count"] += 1
                    if len(details) < detail_limit:
                        details.append(
                            {
                                "candidate_id": candidate,
                                "field": field,
                                "expected": expected_number,
                                "actual": actual_number,
                                "delta": delta,
                                "absolute_difference": absolute,
                                "relative_difference": relative,
                            }
                        )
                if absolute > 1e-12 and relative > 1e-12:
                    tolerance_exceedances += 1
                    stats["tolerance_exceedance_count"] += 1
                is_rank_field = field.startswith("rank_") and field != "rank_consensus_score"
                if is_rank_field and absolute != 0.0:
                    rank_mismatches += 1
            else:
                text_cells += 1
                expected_text = str(expected_row.get(field, ""))
                actual_value = actual_row.get(field, "")
                actual_text = "" if actual_value is None else str(actual_value)
                if expected_text != actual_text:
                    text_mismatches += 1
                    if field.startswith("rank_") and field != "rank_consensus_score":
                        rank_mismatches += 1
                    if len(details) < detail_limit:
                        details.append(
                            {
                                "candidate_id": candidate,
                                "field": field,
                                "expected": expected_text,
                                "actual": actual_text,
                            }
                        )
    extra_rows = sorted(set(actual_by_id) - {str(row["pdb_name"]) for row in expected})
    return {
        "numerical_cell_count": numerical_cells,
        "text_cell_count": text_cells,
        "exact_numerical_difference_count": exact_numerical_differences,
        "tolerance_exceedance_count": tolerance_exceedances,
        "text_mismatch_count": text_mismatches,
        "rank_mismatch_count": rank_mismatches,
        "missing_row_count": len(missing_rows),
        "extra_row_count": len(extra_rows),
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "field_statistics": field_stats,
        "difference_examples": details,
    }


def run(evidence_root: Path, reference_dir: Path) -> tuple[dict[str, Any], int]:
    scored_root = (
        evidence_root
        / "04_blind_provenance"
        / "core10_heldout_v5_blind_scores_v1"
    )
    files = sorted(scored_root.glob("*/v5_alpha.scored.tsv"))
    if len(files) != 5:
        raise ValueError(f"expected five held-out scored tables, found {len(files)}")
    targets = []
    total_rows = 0
    total_cells = 0
    historical_exact_differences = 0
    max_historical_absolute = 0.0
    max_historical_relative = 0.0
    for path in files:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            historical = list(csv.DictReader(handle, delimiter="\t"))
            headers = list(historical[0]) if historical else []
        if len(historical) != 5000:
            raise ValueError(f"{path.parent.name}: expected 5000 rows")
        comparison_fields = [field for field in headers if field not in FIELDNAMES]
        migrated, _ = score_pool(historical)
        direct = direct_frozen_score(historical, reference_dir)
        historical_comparison = compare(historical, migrated, comparison_fields)
        migration_comparison = compare(direct, migrated, comparison_fields)
        targets.append(
            {
                "target_id": path.parent.name,
                "row_count": len(historical),
                "comparison_field_count": len(comparison_fields),
                "historical_vs_migrated": historical_comparison,
                "direct_frozen_vs_migrated": migration_comparison,
            }
        )
        total_rows += len(historical)
        total_cells += historical_comparison["numerical_cell_count"]
        historical_exact_differences += historical_comparison[
            "exact_numerical_difference_count"
        ]
        max_historical_absolute = max(
            max_historical_absolute,
            historical_comparison["max_absolute_difference"],
        )
        max_historical_relative = max(
            max_historical_relative,
            historical_comparison["max_relative_difference"],
        )
    direct_failures = sum(
        target["direct_frozen_vs_migrated"]["exact_numerical_difference_count"]
        + target["direct_frozen_vs_migrated"]["text_mismatch_count"]
        + target["direct_frozen_vs_migrated"]["missing_row_count"]
        + target["direct_frozen_vs_migrated"]["extra_row_count"]
        for target in targets
    )
    material_historical_failures = sum(
        target["historical_vs_migrated"]["tolerance_exceedance_count"]
        + target["historical_vs_migrated"]["text_mismatch_count"]
        + target["historical_vs_migrated"]["rank_mismatch_count"]
        + target["historical_vs_migrated"]["missing_row_count"]
        + target["historical_vs_migrated"]["extra_row_count"]
        for target in targets
    )
    report = {
        "schema_version": "1",
        "status": "pass" if direct_failures == 0 and material_historical_failures == 0 else "fail",
        "normalization_scope": "five independent complete 5000-row target pools",
        "target_count": len(targets),
        "candidate_count": total_rows,
        "historical_numerical_cell_count": total_cells,
        "historical_exact_numerical_difference_count": historical_exact_differences,
        "historical_tolerance_exceedance_count": sum(
            target["historical_vs_migrated"]["tolerance_exceedance_count"]
            for target in targets
        ),
        "historical_rank_mismatch_count": sum(
            target["historical_vs_migrated"]["rank_mismatch_count"]
            for target in targets
        ),
        "historical_text_mismatch_count": sum(
            target["historical_vs_migrated"]["text_mismatch_count"]
            for target in targets
        ),
        "historical_max_absolute_difference": max_historical_absolute,
        "historical_max_relative_difference": max_historical_relative,
        "direct_frozen_vs_migrated_exact_difference_count": direct_failures,
        "targets": targets,
    }
    return report, 0 if report["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reference" / "frozen_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, exit_code = run(args.evidence_root.resolve(), args.reference_dir.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "target_count",
                        "candidate_count",
                        "historical_numerical_cell_count",
                        "historical_exact_numerical_difference_count",
                        "historical_tolerance_exceedance_count",
                        "historical_rank_mismatch_count",
                        "historical_text_mismatch_count",
                        "historical_max_absolute_difference",
                        "historical_max_relative_difference",
                        "direct_frozen_vs_migrated_exact_difference_count",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(serialized, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
