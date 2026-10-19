#!/usr/bin/env python3
"""Build the repository data checksum and provenance manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(relative: str) -> tuple[str, str]:
    if relative.startswith("source_data/"):
        return "figure_source_data", "final_v1.2/source_data"
    if relative.startswith("benchmark/frozen_scores/"):
        return "complete_frozen_v5_score_pool", "evidence/04_blind_provenance/core10_heldout_v5_blind_scores_v1"
    if relative.startswith("benchmark/af2_labels/"):
        return "heldout_af2_labels_and_runtime", "evidence/01_main_results/core10_af2_ground_truth_tables/heldout_merged_v1"
    if relative.startswith("benchmark/high_recall/"):
        return "high_recall_pruning_summary", "evidence/01_main_results/stage1_high_recall_pruning_v1"
    if relative.startswith("benchmark/baselines/"):
        return "baseline_or_length_confound_summary", "evidence/02_baselines/stage1_literature_baselines_v1"
    if relative.startswith("benchmark/runtime/"):
        return "runtime_summary", "evidence/03_runtime/runtime_audit_v1"
    if relative.startswith("benchmark/") and relative.endswith("parity.json"):
        return "full_pool_scoring_parity", "local frozen-source parity run"
    if relative.startswith("benchmark/") and relative.endswith("headline_summary.json"):
        return "recomputed_headline_summary", "repository benchmark TSVs"
    if relative.startswith("representative_structures/") and relative.endswith(".pdb"):
        return "representative_normalized_structure", "evidence/09_representative_structures"
    if relative == "representative_structures/representative_structures_metadata.tsv":
        return "representative_structure_metadata", "evidence/09_representative_structures"
    return "documentation", "release-candidate repository"


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repository / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = (args.output or data_root / "MANIFEST.tsv").resolve()
    files = sorted(
        (
            path
            for path in data_root.rglob("*")
            if path.is_file() and path.resolve() != output
        ),
        key=lambda path: path.relative_to(data_root).as_posix(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "relative_path",
                "bytes",
                "sha256",
                "role",
                "source",
                "redistribution_status",
            ]
        )
        for path in files:
            relative = path.relative_to(data_root).as_posix()
            role, source = describe(relative)
            writer.writerow(
                [
                    relative,
                    path.stat().st_size,
                    sha256_file(path),
                    role,
                    source,
                    "owner_confirmation_required",
                ]
            )
    print(f"wrote {len(files)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
