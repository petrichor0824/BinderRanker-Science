#!/usr/bin/env python3
"""Audit whether historical scored TSVs reconstruct extractor input schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from binderranker.geometry.contact_field import FIELDNAMES
from binderranker.provenance import sha256_file, write_json_atomic


def audit(path: Path, expected_rows: int, id_column: str) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        columns = reader.fieldnames or []
    required = list(FIELDNAMES)
    missing = [column for column in required if column not in columns]
    duplicate_ids = 0
    seen: set[str] = set()
    empty_ids = 0
    error_rows = 0
    numeric_parse_errors: list[dict[str, Any]] = []
    numeric_columns = [
        column for column in required if column not in {"pdb_name", "error"}
    ]
    for index, row in enumerate(rows, 2):
        candidate_id = str(row.get(id_column, "")).strip()
        if not candidate_id:
            empty_ids += 1
        elif candidate_id in seen:
            duplicate_ids += 1
        seen.add(candidate_id)
        if str(row.get("error", "")).strip():
            error_rows += 1
        for column in numeric_columns:
            if column not in row or str(row[column]).strip() == "":
                numeric_parse_errors.append(
                    {"line": index, "candidate_id": candidate_id, "column": column, "value": row.get(column, "")}
                )
                continue
            try:
                float(row[column])
            except ValueError:
                numeric_parse_errors.append(
                    {"line": index, "candidate_id": candidate_id, "column": column, "value": row[column]}
                )
    reconstructable = (
        not missing
        and len(rows) == expected_rows
        and empty_ids == 0
        and duplicate_ids == 0
        and not numeric_parse_errors
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "column_count": len(columns),
        "required_extractor_column_count": len(required),
        "missing_extractor_columns": missing,
        "extra_columns": [column for column in columns if column not in required],
        "empty_id_count": empty_ids,
        "duplicate_id_count": duplicate_ids,
        "error_row_count": error_rows,
        "numeric_parse_error_count": len(numeric_parse_errors),
        "numeric_parse_errors": numeric_parse_errors,
        "extractor_input_schema_reconstructable": reconstructable,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-tsv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=5000)
    parser.add_argument("--id-column", default="pdb_name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    files = [audit(path, args.expected_rows, args.id_column) for path in args.scored_tsv]
    report = {
        "schema_version": "1",
        "file_count": len(files),
        "all_reconstructable": all(
            item["extractor_input_schema_reconstructable"] for item in files
        ),
        "normalization_contract": {
            "scope": "complete_per_target_pool",
            "expected_rows_per_target": args.expected_rows,
            "fragment_normalization_forbidden": True,
            "representative_subset_scoring_forbidden": True,
            "rule": (
                "For each target, preserve the complete original candidate pool and "
                "reproduce frozen per-target normalization over that entire pool. Do "
                "not normalize table fragments or the representative-PDB subset."
            ),
        },
        "files": files,
    }
    write_json_atomic(args.output, report, overwrite=args.overwrite)
    print(json.dumps({
        "file_count": report["file_count"],
        "all_reconstructable": report["all_reconstructable"],
    }, indent=2, sort_keys=True))
    return 0 if report["all_reconstructable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
