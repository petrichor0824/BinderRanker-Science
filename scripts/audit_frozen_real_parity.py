#!/usr/bin/env python3
"""Audit direct frozen extraction against history and the migrated package.

This is deliberately separate from ``run_descriptor_parity.py``. It imports the
byte-protected historical ``contact_field_v3.py`` and runs that source directly
on every PDB in the parity manifest. The report distinguishes:

1. historical scored TSV vs direct frozen extraction in the current runtime;
2. direct frozen extraction vs the migrated package implementation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from binderranker.config import FROZEN_GEOMETRY_V1
from binderranker.geometry.contact_field import FIELDNAMES, extract_structure
from binderranker.provenance import sha256_file, write_json_atomic


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_frozen(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage2a_frozen_contact_field_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_args(binder_chain: str) -> argparse.Namespace:
    values = asdict(FROZEN_GEOMETRY_V1)
    values.pop("profile")
    values["binder_chain"] = binder_chain
    return argparse.Namespace(**values)


class Comparison:
    def __init__(self, label: str, absolute_tolerance: float, relative_tolerance: float):
        self.label = label
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.numerical_cell_count = 0
        self.different_numerical_cell_count = 0
        self.flagged_numerical_cell_count = 0
        self.max_absolute_difference = 0.0
        self.max_relative_difference = 0.0
        self.field_statistics: dict[str, dict[str, Any]] = {}
        self.differences: list[dict[str, Any]] = []

    def add(
        self,
        *,
        target_id: str,
        candidate_id: str,
        field: str,
        expected: float,
        actual: float,
    ) -> None:
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1e-300)
        flagged = (
            absolute > self.absolute_tolerance
            and relative > self.relative_tolerance
        )
        self.numerical_cell_count += 1
        self.max_absolute_difference = max(self.max_absolute_difference, absolute)
        self.max_relative_difference = max(self.max_relative_difference, relative)
        stats = self.field_statistics.setdefault(
            field,
            {
                "count": 0,
                "different_count": 0,
                "flagged_count": 0,
                "max_absolute_difference": 0.0,
                "max_relative_difference": 0.0,
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
            self.different_numerical_cell_count += 1
            stats["different_count"] += 1
            self.differences.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                    "flagged": flagged,
                }
            )
        if flagged:
            self.flagged_numerical_cell_count += 1
            stats["flagged_count"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": "pass" if self.flagged_numerical_cell_count == 0 else "fail",
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "numerical_cell_count": self.numerical_cell_count,
            "different_numerical_cell_count": self.different_numerical_cell_count,
            "flagged_numerical_cell_count": self.flagged_numerical_cell_count,
            "max_absolute_difference": self.max_absolute_difference,
            "max_relative_difference": self.max_relative_difference,
            "field_statistics": dict(sorted(self.field_statistics.items())),
            "differences": self.differences,
        }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    manifest_path = args.manifest.resolve()
    manifest_rows = read_tsv(manifest_path)
    if not manifest_rows:
        raise ValueError("parity manifest has no rows")
    required = {"target_id", "candidate_id", "pdb_path", "scored_tsv"}
    missing_columns = required - set(manifest_rows[0])
    if missing_columns:
        raise ValueError(
            "parity manifest missing columns: " + ", ".join(sorted(missing_columns))
        )

    frozen_source = args.frozen_source.resolve()
    frozen = load_frozen(frozen_source)
    if list(frozen.FIELDNAMES) != list(FIELDNAMES):
        raise ValueError("frozen and package extractor field orders differ")
    descriptor_fields = [
        field for field in FIELDNAMES if field not in {"pdb_name", "error"}
    ]
    historical_vs_frozen = Comparison(
        "historical_scored_tsv_vs_direct_frozen",
        args.absolute_tolerance,
        args.relative_tolerance,
    )
    frozen_vs_package = Comparison(
        "direct_frozen_vs_migrated_package",
        args.absolute_tolerance,
        args.relative_tolerance,
    )
    scored_cache: dict[Path, dict[str, dict[str, str]]] = {}
    errors: list[dict[str, str]] = []
    base = manifest_path.parent

    for record in manifest_rows:
        target_id = record["target_id"]
        candidate_id = record["candidate_id"]
        pdb_path = resolve(base, record["pdb_path"])
        scored_path = resolve(base, record["scored_tsv"])
        if scored_path not in scored_cache:
            indexed: dict[str, dict[str, str]] = {}
            for row in read_tsv(scored_path):
                key = str(row.get(args.id_column, ""))
                if not key or key in indexed:
                    raise ValueError(
                        f"{scored_path}: empty or duplicate {args.id_column}={key!r}"
                    )
                indexed[key] = row
            scored_cache[scored_path] = indexed
        historical = scored_cache[scored_path].get(candidate_id)
        if historical is None:
            raise ValueError(f"{scored_path}: candidate not found: {candidate_id}")
        try:
            direct = frozen.compute_metrics(
                pdb_path, frozen_args(args.binder_chain), [], []
            )
            _, migrated = extract_structure(
                pdb_path, binder_chain=args.binder_chain, strict=False
            )
            for field in descriptor_fields:
                historical_vs_frozen.add(
                    target_id=target_id,
                    candidate_id=candidate_id,
                    field=field,
                    expected=float(historical[field]),
                    actual=float(direct[field]),
                )
                frozen_vs_package.add(
                    target_id=target_id,
                    candidate_id=candidate_id,
                    field=field,
                    expected=float(direct[field]),
                    actual=float(migrated[field]),
                )
        except Exception as exc:
            errors.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "error": str(exc),
                }
            )

    comparisons = {
        "historical_vs_direct_frozen": historical_vs_frozen.as_dict(),
        "direct_frozen_vs_migrated_package": frozen_vs_package.as_dict(),
    }
    passed = not errors and all(
        comparison["status"] == "pass" for comparison in comparisons.values()
    )
    report = {
        "schema_version": "1",
        "status": "pass" if passed else "fail",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "frozen_source": str(frozen_source),
        "frozen_source_sha256": sha256_file(frozen_source),
        "candidate_count": len(manifest_rows),
        "descriptor_field_count": len(descriptor_fields),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "extraction_error_count": len(errors),
        "extraction_errors": errors,
        "comparisons": comparisons,
    }
    return report, 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="pdb_name")
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument("--relative-tolerance", type=float, default=1e-8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report, exit_code = run(args)
    write_json_atomic(args.output, report, overwrite=args.overwrite)
    compact = {
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "descriptor_field_count": report["descriptor_field_count"],
        "extraction_error_count": report["extraction_error_count"],
        "comparisons": {
            key: {
                name: value[name]
                for name in (
                    "status",
                    "numerical_cell_count",
                    "different_numerical_cell_count",
                    "flagged_numerical_cell_count",
                    "max_absolute_difference",
                    "max_relative_difference",
                )
            }
            for key, value in report["comparisons"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
