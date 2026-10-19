"""External Stage 2A golden tests.

Set ``BINDERRANKER_EVIDENCE_ROOT`` to the unpacked submission evidence pack.
The tests never modify or copy evidence files. They are skipped when the local,
untracked evidence pack is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from binderranker.io.pdb import parse_structure


EVIDENCE_ENV = "BINDERRANKER_EVIDENCE_ROOT"


def load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        "real_evidence_" + name.removesuffix(".py"), path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        return sum(1 for _ in reader)


@unittest.skipUnless(os.environ.get(EVIDENCE_ENV), f"set {EVIDENCE_ENV}")
class RealEvidenceGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = Path(os.environ[EVIDENCE_ENV]).resolve()
        if not cls.evidence.is_dir():
            raise AssertionError(f"evidence root is not a directory: {cls.evidence}")
        cls.project = Path(__file__).resolve().parents[1]

    def build_parity_manifest(self, path: Path) -> None:
        metadata_path = (
            self.evidence
            / "09_representative_structures"
            / "representative_structures_metadata.tsv"
        )
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(21, len(rows))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["target_id", "candidate_id", "pdb_path", "scored_tsv"],
                delimiter="\t",
            )
            writer.writeheader()
            for row in rows:
                target = f"{row['target_id']}_{row['pdb_id']}"
                writer.writerow(
                    {
                        "target_id": target,
                        "candidate_id": f"design_{row['design_index']}_normalized",
                        "pdb_path": str(
                            self.evidence
                            / "09_representative_structures"
                            / (
                                f"{target}__{row['category']}__"
                                f"design_{row['design_index']}.pdb"
                            )
                        ),
                        "scored_tsv": str(
                            self.evidence
                            / "04_blind_provenance"
                            / "core10_heldout_v5_blind_scores_v1"
                            / target
                            / "v5_alpha.scored.tsv"
                        ),
                    }
                )

    def test_evidence_manifest_has_no_missing_or_changed_files(self) -> None:
        manifest = self.evidence / "00_manifest" / "SHA256SUMS.txt"
        entries: list[tuple[str, str]] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                expected, relative = line.split(None, 1)
                relative = relative.removeprefix("*").removeprefix("./")
                entries.append((expected.lower(), relative))
        self.assertEqual(286, len(entries))
        self.assertEqual(286, len({relative for _, relative in entries}))
        for expected, relative in entries:
            path = self.evidence / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(expected, sha256_file(path), relative)

    def test_ten_core_tables_match_recorded_hashes_and_schema(self) -> None:
        schema = load_script("audit_scored_schema.py")
        manifest_path = (
            self.evidence
            / "01_main_results"
            / "core10_af2_ground_truth_tables"
            / "heldout_merged_v1"
            / "heldout_merged_manifest.tsv"
        )
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(5, len(records))
        for row in records:
            target = f"{row['target_id']}_{row['pdb_id']}"
            scored = (
                self.evidence
                / "04_blind_provenance"
                / "core10_heldout_v5_blind_scores_v1"
                / target
                / "v5_alpha.scored.tsv"
            )
            af2 = (
                self.evidence
                / "01_main_results"
                / "core10_af2_ground_truth_tables"
                / "heldout_merged_v1"
                / f"{target}_af2_scores_5000.tsv"
            )
            self.assertEqual(row["blind_score_sha256"], sha256_file(scored), target)
            self.assertEqual(row["merged_af2_sha256"], sha256_file(af2), target)
            self.assertEqual(5000, data_row_count(af2), target)
            scored_audit = schema.audit(scored, expected_rows=5000, id_column="pdb_name")
            self.assertTrue(
                scored_audit["extractor_input_schema_reconstructable"], target
            )

    def test_21_pdb_descriptor_parity_and_target_identity(self) -> None:
        parity = load_script("run_descriptor_parity.py")
        frozen_audit = load_script("audit_frozen_real_parity.py")
        pdbs = sorted((self.evidence / "09_representative_structures").glob("*.pdb"))
        self.assertEqual(21, len(pdbs))
        for pdb in pdbs:
            parsed = parse_structure(pdb, strict=True)
            self.assertTrue(parsed.diagnostics.strict_valid, pdb.name)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "real_parity_manifest.tsv"
            self.build_parity_manifest(manifest)
            parity_args = argparse.Namespace(
                manifest=manifest,
                id_column="pdb_name",
                binder_chain="A",
                strict=False,
                target_rmsd_tolerance=1.0,
                absolute_tolerance=1e-8,
                relative_tolerance=1e-8,
            )
            parity_report, parity_exit = parity.run(parity_args)
            self.assertEqual(0, parity_exit)
            self.assertEqual("pass", parity_report["status"])
            self.assertEqual(21, parity_report["candidate_count"])
            self.assertEqual(5, parity_report["target_count"])
            self.assertEqual(1428, parity_report["numerical_cell_count"])
            self.assertEqual(0, parity_report["flagged_numerical_cell_count"])
            self.assertEqual(0, parity_report["missing_field_count"])
            self.assertEqual(0, parity_report["missing_value_count"])
            self.assertEqual(0, parity_report["extraction_error_count"])
            self.assertEqual(10, len(parity_report["cross_target_identity_checks"]))
            self.assertTrue(
                all(
                    item["rejected_as_mixed"]
                    for item in parity_report["cross_target_identity_checks"]
                )
            )

            frozen_args = argparse.Namespace(
                manifest=manifest,
                frozen_source=self.project
                / "reference"
                / "frozen_v1"
                / "contact_field_v3.py",
                id_column="pdb_name",
                binder_chain="A",
                absolute_tolerance=1e-8,
                relative_tolerance=1e-8,
            )
            frozen_report, frozen_exit = frozen_audit.run(frozen_args)
            self.assertEqual(0, frozen_exit)
            self.assertEqual("pass", frozen_report["status"])
            direct = frozen_report["comparisons"]["historical_vs_direct_frozen"]
            migrated = frozen_report["comparisons"][
                "direct_frozen_vs_migrated_package"
            ]
            self.assertEqual(1428, direct["numerical_cell_count"])
            self.assertEqual(0, direct["flagged_numerical_cell_count"])
            self.assertEqual(1428, migrated["numerical_cell_count"])
            self.assertEqual(0.0, migrated["max_absolute_difference"])

    def test_five_complete_pools_frozen_v5_scoring_and_ranking_parity(self) -> None:
        parity = load_script("audit_frozen_v5_parity.py")
        report, exit_code = parity.run(
            self.evidence,
            self.project / "reference" / "frozen_v1",
        )
        self.assertEqual(0, exit_code)
        self.assertEqual("pass", report["status"])
        self.assertEqual(5, report["target_count"])
        self.assertEqual(25000, report["candidate_count"])
        self.assertEqual(0, report["historical_tolerance_exceedance_count"])
        self.assertEqual(0, report["historical_rank_mismatch_count"])
        self.assertEqual(0, report["historical_text_mismatch_count"])
        self.assertEqual(
            0, report["direct_frozen_vs_migrated_exact_difference_count"]
        )


if __name__ == "__main__":
    unittest.main()
