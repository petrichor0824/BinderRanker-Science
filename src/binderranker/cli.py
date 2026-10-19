"""Command-line interface for Stage 2A BinderRanker functionality."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from binderranker import __version__
from binderranker.config import FROZEN_GEOMETRY_V1
from binderranker.exceptions import BinderRankerError
from binderranker.geometry.contact_field import FIELDNAMES, extract_structure
from binderranker.io.pdb import ParsedPDB, parse_structure
from binderranker.io.tables import (
    read_delimited,
    write_csv_atomic,
    write_delimited_atomic,
)
from binderranker.provenance import (
    build_provenance,
    runtime_identity,
    sha256_file,
    write_json_atomic,
)
from binderranker.scoring import FROZEN_V5, ranked_rows, score_pool
from binderranker.scoring.frozen_v5 import is_valid_row, ordered_output_fields
from binderranker.selection import select_top
from binderranker.validation import validate_target_pool


def collect_pdbs(input_paths: list[str] | None, input_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    for value in input_paths or []:
        paths.append(Path(value))
    if input_dir:
        paths.extend(sorted(Path(input_dir).glob("*.pdb")))
    if not paths:
        raise BinderRankerError("no PDB files found")
    resolved: set[Path] = set()
    stems: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise BinderRankerError(f"PDB input does not exist: {path}")
        canonical = path.resolve()
        if canonical in resolved:
            raise BinderRankerError(f"duplicate PDB input: {path}")
        if path.stem in stems:
            raise BinderRankerError(
                f"duplicate candidate stem is not allowed: {path.stem}"
            )
        resolved.add(canonical)
        stems.add(path.stem)
    return paths


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", action="append", default=[], help="PDB path; repeatable")
    parser.add_argument("--input-dir", help="non-recursive directory containing *.pdb")
    parser.add_argument("--binder-chain", default="A")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="reject ambiguous/incomplete records without changing legacy parsing",
    )
    parser.add_argument(
        "--target-rmsd-tolerance",
        type=float,
        default=1.0,
        help="maximum Kabsch-aligned target C-alpha RMSD within one pool (angstrom)",
    )


def command_version(args: argparse.Namespace) -> int:
    identity = runtime_identity()
    identity["scientific_profile"] = FROZEN_V5.profile
    identity["scientific_profiles"] = {
        "geometry": FROZEN_GEOMETRY_V1.profile,
        "scoring": FROZEN_V5.profile,
    }
    if args.json:
        print(json.dumps(identity, indent=2, sort_keys=True))
    else:
        print(f"BinderRanker {__version__}")
        print(f"scientific profile: {FROZEN_V5.profile}")
        print(f"geometry profile: {FROZEN_GEOMETRY_V1.profile}")
        print(f"Python: {identity['python_version']}")
        print(f"NumPy: {identity['numpy_version']}")
        for name, digest in identity["frozen_source_sha256"].items():
            print(f"{name}: {digest}")
    return 0


def _parse_pool(args: argparse.Namespace) -> tuple[list[Path], list[ParsedPDB], dict[str, Any]]:
    paths = collect_pdbs(args.input, args.input_dir)
    structures = [
        parse_structure(path, binder_chain=args.binder_chain, strict=args.strict)
        for path in paths
    ]
    target_validation = validate_target_pool(
        structures, rmsd_tolerance=args.target_rmsd_tolerance
    )
    return paths, structures, target_validation


def command_inspect(args: argparse.Namespace) -> int:
    paths, structures, target_validation = _parse_pool(args)
    report = {
        "schema_version": "1",
        "binder_chain": args.binder_chain,
        "strict_validation": args.strict,
        "candidate_count": len(structures),
        "candidates": [
            {
                "name": structure.path.name,
                "binder_len": len(structure.binder),
                "target_len": len(structure.target),
                "binder_chains": sorted({residue.chain for residue in structure.binder}),
                "target_chains": sorted({residue.chain for residue in structure.target}),
                "diagnostics": structure.diagnostics.as_dict(),
            }
            for structure in structures
        ],
        "target_pool_validation": target_validation,
    }
    if args.output:
        write_json_atomic(Path(args.output), report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_extract(args: argparse.Namespace) -> int:
    paths = collect_pdbs(args.input, args.input_dir)
    structures: list[ParsedPDB] = []
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            structure, row = extract_structure(
                path,
                binder_chain=args.binder_chain,
                strict=args.strict,
                config=FROZEN_GEOMETRY_V1,
                desired_regions=args.desired_regions,
                undesired_regions=args.undesired_regions,
            )
            structures.append(structure)
            rows.append(row)
        except Exception as exc:
            if args.on_invalid == "error":
                raise
            errors.append(f"{path.name}: {exc}")
            rows.append({"pdb_name": path.stem, "error": str(exc)})
    if not structures:
        raise BinderRankerError("no valid structures remain after parsing")
    target_validation = validate_target_pool(
        structures, rmsd_tolerance=args.target_rmsd_tolerance
    )
    output = Path(args.output)
    provenance_path = (
        Path(args.provenance)
        if args.provenance
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    if provenance_path.exists() and not args.overwrite:
        raise BinderRankerError(
            f"refusing to overwrite existing provenance: {provenance_path}"
        )
    write_csv_atomic(output, rows, FIELDNAMES, overwrite=args.overwrite)
    provenance = build_provenance(
        command="extract",
        inputs=paths,
        config=FROZEN_GEOMETRY_V1,
        binder_chain=args.binder_chain,
        strict=args.strict,
        target_validation=target_validation,
        valid_count=len(structures),
        error_count=len(errors),
    )
    provenance["row_errors"] = errors
    write_json_atomic(provenance_path, provenance, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "provenance": str(provenance_path),
                "rows": len(rows),
                "valid": len(structures),
                "errors": len(errors),
                "target_sequence_signature": target_validation[
                    "target_sequence_signature"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_selection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top-k", type=int, help="select this many candidates after ranking")
    group.add_argument(
        "--top-fraction",
        type=float,
        help="select this fraction of the pool after ranking (0 to 1)",
    )
    parser.add_argument(
        "--selection-output",
        help="selected TSV/CSV; defaults to <output>.selected when a budget is given",
    )


def _write_scored_pool(
    *,
    scored_rows: list[dict[str, Any]],
    output: Path,
    overwrite: bool,
    top_k: int | None,
    top_fraction: float | None,
    selection_output: str | None,
) -> tuple[list[dict[str, Any]], Path | None]:
    ranked = ranked_rows(scored_rows)
    invalid = [dict(row) for row in scored_rows if not is_valid_row(row)]
    output_rows = ranked + invalid
    fields = ordered_output_fields(output_rows)
    write_delimited_atomic(output, output_rows, fields, overwrite=overwrite)
    if top_k is None and top_fraction is None:
        if selection_output:
            raise ValueError("--selection-output requires --top-k or --top-fraction")
        return ranked, None
    selected = select_top(ranked, top_k=top_k, top_fraction=top_fraction)
    if selection_output:
        selected_path = Path(selection_output)
    else:
        selected_path = output.with_name(
            f"{output.stem}.selected{output.suffix or '.tsv'}"
        )
    write_delimited_atomic(selected_path, selected, fields, overwrite=overwrite)
    return ranked, selected_path


def command_score(args: argparse.Namespace) -> int:
    source = Path(args.input)
    if not source.is_file():
        raise BinderRankerError(f"descriptor input does not exist: {source}")
    rows = read_delimited(source)
    scored, scoring_metadata = score_pool(rows)
    output = Path(args.output)
    provenance_path = (
        Path(args.provenance)
        if args.provenance
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    if provenance_path.exists() and not args.overwrite:
        raise BinderRankerError(
            f"refusing to overwrite existing provenance: {provenance_path}"
        )
    ranked, selected_path = _write_scored_pool(
        scored_rows=scored,
        output=output,
        overwrite=args.overwrite,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        selection_output=args.selection_output,
    )
    provenance = {
        "schema_version": "1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "score",
        **runtime_identity(),
        "scientific_profile": FROZEN_V5.profile,
        "input": {
            "name": source.name,
            "resolved_path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "scoring": scoring_metadata,
        "output_semantics": {
            "rank_scope": "complete_valid_target_pool",
            "filters_change_primary_rank": False,
            "score_is_binding_probability": False,
        },
    }
    write_json_atomic(provenance_path, provenance, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "provenance": str(provenance_path),
                "pool_rows": len(scored),
                "ranked_rows": len(ranked),
                "selected_rows": 0 if selected_path is None else len(
                    select_top(
                        ranked,
                        top_k=args.top_k,
                        top_fraction=args.top_fraction,
                    )
                ),
                "selection_output": None if selected_path is None else str(selected_path),
                "scientific_profile": FROZEN_V5.profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_rank(args: argparse.Namespace) -> int:
    paths = collect_pdbs(args.input, args.input_dir)
    structures: list[ParsedPDB] = []
    descriptors: list[dict[str, Any]] = []
    for path in paths:
        structure, row = extract_structure(
            path,
            binder_chain=args.binder_chain,
            strict=args.strict,
            config=FROZEN_GEOMETRY_V1,
            desired_regions=args.desired_regions,
            undesired_regions=args.undesired_regions,
        )
        structures.append(structure)
        descriptors.append(row)
    target_validation = validate_target_pool(
        structures, rmsd_tolerance=args.target_rmsd_tolerance
    )
    scored, scoring_metadata = score_pool(descriptors)
    output = Path(args.output)
    provenance_path = (
        Path(args.provenance)
        if args.provenance
        else output.with_suffix(output.suffix + ".provenance.json")
    )
    if provenance_path.exists() and not args.overwrite:
        raise BinderRankerError(
            f"refusing to overwrite existing provenance: {provenance_path}"
        )
    ranked, selected_path = _write_scored_pool(
        scored_rows=scored,
        output=output,
        overwrite=args.overwrite,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        selection_output=args.selection_output,
    )
    if args.descriptors_output:
        write_csv_atomic(
            Path(args.descriptors_output),
            descriptors,
            FIELDNAMES,
            overwrite=args.overwrite,
        )
    provenance = build_provenance(
        command="rank",
        inputs=paths,
        config=FROZEN_GEOMETRY_V1,
        binder_chain=args.binder_chain,
        strict=args.strict,
        target_validation=target_validation,
        valid_count=len(descriptors),
        error_count=0,
    )
    provenance["scoring"] = scoring_metadata
    provenance["output_semantics"] = {
        "rank_scope": "complete_valid_target_pool",
        "filters_change_primary_rank": False,
        "score_is_binding_probability": False,
    }
    write_json_atomic(provenance_path, provenance, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(output),
                "provenance": str(provenance_path),
                "ranked_rows": len(ranked),
                "selection_output": None if selected_path is None else str(selected_path),
                "target_sequence_signature": target_validation[
                    "target_sequence_signature"
                ],
                "scientific_profile": FROZEN_V5.profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="binderranker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="show package and scientific-profile versions")
    version_parser.add_argument("--json", action="store_true")
    version_parser.set_defaults(func=command_version)

    inspect_parser = subparsers.add_parser("inspect", help="inspect PDB parsing and target-pool identity")
    _add_inputs(inspect_parser)
    inspect_parser.add_argument("--output", help="optional JSON output path")
    inspect_parser.add_argument("--overwrite", action="store_true")
    inspect_parser.set_defaults(func=command_inspect)

    extract_parser = subparsers.add_parser("extract", help="extract frozen-v1 geometric descriptors")
    _add_inputs(extract_parser)
    extract_parser.add_argument("--output", required=True, help="descriptor CSV")
    extract_parser.add_argument("--provenance", help="provenance JSON; defaults beside CSV")
    extract_parser.add_argument("--desired-regions", default="")
    extract_parser.add_argument("--undesired-regions", default="")
    extract_parser.add_argument("--on-invalid", choices=["error", "record"], default="error")
    extract_parser.add_argument("--overwrite", action="store_true")
    extract_parser.set_defaults(func=command_extract)

    score_parser = subparsers.add_parser(
        "score", help="score and rank one complete target-specific descriptor pool"
    )
    score_parser.add_argument("--input", required=True, help="descriptor CSV or TSV")
    score_parser.add_argument("--output", required=True, help="ranked CSV or TSV")
    score_parser.add_argument("--provenance", help="provenance JSON; defaults beside output")
    _add_selection(score_parser)
    score_parser.add_argument("--overwrite", action="store_true")
    score_parser.set_defaults(func=command_score)

    rank_parser = subparsers.add_parser(
        "rank", help="extract, score, and rank one target-specific PDB pool"
    )
    _add_inputs(rank_parser)
    rank_parser.add_argument("--output", required=True, help="ranked CSV or TSV")
    rank_parser.add_argument("--provenance", help="provenance JSON; defaults beside output")
    rank_parser.add_argument("--descriptors-output", help="optional raw descriptor CSV")
    rank_parser.add_argument("--desired-regions", default="")
    rank_parser.add_argument("--undesired-regions", default="")
    _add_selection(rank_parser)
    rank_parser.add_argument("--overwrite", action="store_true")
    rank_parser.set_defaults(func=command_rank)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (BinderRankerError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
