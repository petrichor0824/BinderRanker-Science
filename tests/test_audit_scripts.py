from __future__ import annotations

import argparse
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from binderranker.geometry.contact_field import FIELDNAMES, extract_structure

from tests.helpers import write_complex


def load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditScriptTests(unittest.TestCase):
    def test_parity_and_scored_schema_harnesses(self) -> None:
        parity = load_script("run_descriptor_parity.py")
        schema = load_script("audit_scored_schema.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdbs = [
                write_complex(root / "candidate_1.pdb"),
                write_complex(root / "candidate_2.pdb", target_offset=100, target_noise=0.01),
            ]
            scored = root / "scored.tsv"
            rows = [extract_structure(path)[1] for path in pdbs]
            with scored.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            manifest = root / "manifest.tsv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["target_id", "candidate_id", "pdb_path", "scored_tsv"],
                    delimiter="\t",
                )
                writer.writeheader()
                for path in pdbs:
                    writer.writerow(
                        {
                            "target_id": "synthetic_target",
                            "candidate_id": path.stem,
                            "pdb_path": path.name,
                            "scored_tsv": scored.name,
                        }
                    )
            args = argparse.Namespace(
                manifest=manifest,
                id_column="pdb_name",
                binder_chain="A",
                strict=True,
                target_rmsd_tolerance=0.1,
                absolute_tolerance=1e-12,
                relative_tolerance=1e-12,
            )
            report, exit_code = parity.run(args)
            self.assertEqual(0, exit_code)
            self.assertEqual("pass", report["status"])
            self.assertGreater(report["numerical_cell_count"], 100)
            self.assertEqual(0, report["flagged_numerical_cell_count"])
            self.assertEqual([], report["cross_target_identity_checks"])
            schema_report = schema.audit(scored, expected_rows=2, id_column="pdb_name")
            self.assertTrue(schema_report["extractor_input_schema_reconstructable"])

    def test_schema_cli_records_complete_pool_normalization_contract(self) -> None:
        schema = load_script("audit_scored_schema.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scored = root / "scored.tsv"
            row = extract_structure(write_complex(root / "candidate.pdb"))[1]
            with scored.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            output = root / "audit.json"
            exit_code = schema.main(
                [
                    "--scored-tsv",
                    str(scored),
                    "--output",
                    str(output),
                    "--expected-rows",
                    "1",
                ]
            )
            self.assertEqual(0, exit_code)
            import json

            report = json.loads(output.read_text(encoding="utf-8"))
            contract = report["normalization_contract"]
            self.assertEqual("complete_per_target_pool", contract["scope"])
            self.assertTrue(contract["fragment_normalization_forbidden"])
            self.assertTrue(contract["representative_subset_scoring_forbidden"])


if __name__ == "__main__":
    unittest.main()
