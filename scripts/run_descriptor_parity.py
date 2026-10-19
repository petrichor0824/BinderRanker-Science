#!/usr/bin/env python3
"""Compare real PDB extraction against historical scored-table descriptors.

The input manifest is a TSV with columns:
target_id, candidate_id, pdb_path, scored_tsv. Relative paths are resolved from
the manifest directory. No evidence file is copied into the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from binderranker.geometry.contact_field import FIELDNAMES, extract_structure
from binderranker.provenance import sha256_file, write_json_atomic
from binderranker.validation import compare_targets, validate_target_pool


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def relative_difference(expected: float, actual: float) -> float:
    denominator = max(abs(expected), 1e-300)
    return abs(actual - expected) / denominator


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest_path = args.manifest.resolve()
    manifest_rows = read_tsv(manifest_path)
    required = {"target_id", "candidate_id", "pdb_path", "scored_tsv"}
    if not manifest_rows:
        raise ValueError("parity manifest has no rows")
    missing_manifest_columns = required - set(manifest_rows[0])
    if missing_manifest_columns:
        raise ValueError(
            "parity manifest missing columns: " + ", ".join(sorted(missing_manifest_columns))
        )

    base = manifest_path.parent
    scored_cache: dict[Path, dict[str, dict[str, str]]] = {}
    structures_by_target: dict[str, list[Any]] = defaultdict(list)
    field_stats: dict[str, dict[str, Any]] = {}
    differences: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    numerical_cells = 0
    missing_fields = 0
    missing_values = 0
    extraction_errors = 0

    descriptor_fields = [
        name for name in FIELDNAMES if name not in {"pdb_name", "error"}
    ]
    for record in manifest_rows:
        target_id = record["target_id"]
        candidate_id = record["candidate_id"]
        pdb_path = resolve(base, record["pdb_path"])
        scored_path = resolve(base, record["scored_tsv"])
        if scored_path not in scored_cache:
            indexed: dict[str, dict[str, str]] = {}
            for row in read_tsv(scored_path):
                key = str(row.get(args.id_column, ""))
                if not key:
                    raise ValueError(f"{scored_path}: empty {args.id_column}")
                if key in indexed:
                    raise ValueError(
                        f"{scored_path}: duplicate {args.id_column}={key}"
                    )
                indexed[key] = row
            scored_cache[scored_path] = indexed
        expected_row = scored_cache[scored_path].get(candidate_id)
        if expected_row is None:
            raise ValueError(
                f"{scored_path}: candidate not found: {candidate_id}"
            )

        candidate_report: dict[str, Any] = {
            "target_id": target_id,
            "candidate_id": candidate_id,
            "pdb_path": str(pdb_path),
            "pdb_sha256": sha256_file(pdb_path),
            "scored_tsv": str(scored_path),
            "status": "ok",
        }
        try:
            structure, actual_row = extract_structure(
                pdb_path, binder_chain=args.binder_chain, strict=args.strict
            )
            structures_by_target[target_id].append(structure)
            candidate_report.update(
                {
                    "binder_len": len(structure.binder),
                    "target_len": len(structure.target),
                    "binder_chains": sorted(
                        {residue.chain for residue in structure.binder}
                    ),
                    "target_chains": sorted(
                        {residue.chain for residue in structure.target}
                    ),
                    "pdb_diagnostics": structure.diagnostics.as_dict(),
                }
            )
        except Exception as exc:
            extraction_errors += 1
            candidate_report["status"] = "extraction_error"
            candidate_report["error"] = str(exc)
            candidate_results.append(candidate_report)
            continue

        for field in descriptor_fields:
            if field not in expected_row:
                missing_fields += 1
                differences.append(
                    {
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "field": field,
                        "kind": "missing_expected_field",
                    }
                )
                continue
            expected_text = str(expected_row[field]).strip()
            if expected_text == "":
                missing_values += 1
                differences.append(
                    {
                        "target_id": target_id,
                        "candidate_id": candidate_id,
                        "field": field,
                        "kind": "missing_expected_value",
                    }
                )
                continue
            expected = float(expected_text)
            actual = float(actual_row[field])
            absolute = abs(actual - expected)
            relative = relative_difference(expected, actual)
            numerical_cells += 1
            stats = field_stats.setdefault(
                field,
                {
                    "count": 0,
                    "max_absolute_difference": 0.0,
                    "max_relative_difference": 0.0,
                    "flagged_count": 0,
                    "different_count": 0,
                },
            )
            stats["count"] += 1
            stats["max_absolute_difference"] = max(
                stats["max_absolute_difference"], absolute
            )
            stats["max_relative_difference"] = max(
                stats["max_relative_difference"], relative
            )
            if absolute != 0.0:
                stats["different_count"] += 1
                item = {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "field": field,
                    "kind": "numeric_difference",
                    "expected": expected,
                    "actual": actual,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                    "flagged": absolute > args.absolute_tolerance
                    and relative > args.relative_tolerance,
                }
                differences.append(item)
                if item["flagged"]:
                    stats["flagged_count"] += 1
        candidate_results.append(candidate_report)

    target_checks: dict[str, Any] = {}
    target_validation_errors: list[str] = []
    for target_id, structures in sorted(structures_by_target.items()):
        try:
            target_checks[target_id] = validate_target_pool(
                structures, rmsd_tolerance=args.target_rmsd_tolerance
            )
        except Exception as exc:
            target_validation_errors.append(f"{target_id}: {exc}")

    cross_target_checks: list[dict[str, Any]] = []
    cross_target_acceptance_errors: list[str] = []
    target_representatives = {
        target_id: structures[0]
        for target_id, structures in structures_by_target.items()
        if structures
    }
    for (left_id, left), (right_id, right) in combinations(
        sorted(target_representatives.items()), 2
    ):
        comparison = compare_targets(
            left, right, rmsd_tolerance=args.target_rmsd_tolerance
        )
        item = {
            "left_target_id": left_id,
            "right_target_id": right_id,
            "rejected_as_mixed": not comparison.accepted,
            "sequence_match": comparison.sequence_match,
            "ca_rmsd": comparison.ca_rmsd,
            "reason": comparison.reason,
        }
        cross_target_checks.append(item)
        if comparison.accepted:
            cross_target_acceptance_errors.append(
                f"{left_id} vs {right_id}: distinct labeled targets were accepted as one pool"
            )

    flagged = [item for item in differences if item.get("flagged")]
    report = {
        "schema_version": "1",
        "status": "pass"
        if not flagged
        and not extraction_errors
        and not missing_fields
        and not missing_values
        and not target_validation_errors
        and not cross_target_acceptance_errors
        else "fail",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_count": len(manifest_rows),
        "target_count": len(structures_by_target),
        "numerical_cell_count": numerical_cells,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "flagged_numerical_cell_count": len(flagged),
        "different_numerical_cell_count": sum(
            item.get("kind") == "numeric_difference" for item in differences
        ),
        "missing_field_count": missing_fields,
        "missing_value_count": missing_values,
        "extraction_error_count": extraction_errors,
        "target_validation_errors": target_validation_errors,
        "cross_target_identity_checks": cross_target_checks,
        "cross_target_acceptance_errors": cross_target_acceptance_errors,
        "field_statistics": dict(sorted(field_stats.items())),
        "target_identity_checks": target_checks,
        "candidates": candidate_results,
        "differences": differences,
    }
    return report, 0 if report["status"] == "pass" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="pdb_name")
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--target-rmsd-tolerance", type=float, default=1.0)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report, exit_code = run(args)
    write_json_atomic(args.output, report, overwrite=args.overwrite)
    print(json.dumps({key: report[key] for key in [
        "status", "candidate_count", "target_count", "numerical_cell_count",
        "flagged_numerical_cell_count", "missing_field_count",
        "missing_value_count", "extraction_error_count"
    ]}, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
