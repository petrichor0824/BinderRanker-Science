from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from binderranker.cli import main
from binderranker.geometry.contact_field import FIELDNAMES

from tests.helpers import write_complex


class CLITests(unittest.TestCase):
    def test_version_json(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["version", "--json"])
        self.assertEqual(0, code)
        value = json.loads(output.getvalue())
        self.assertEqual("frozen-v5-alpha", value["scientific_profile"])
        self.assertEqual("frozen-v1", value["scientific_profiles"]["geometry"])
        self.assertEqual(4, len(value["frozen_source_sha256"]))

    def test_inspect_and_extract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = write_complex(root / "candidate_0001.pdb")
            inspect_output = io.StringIO()
            with contextlib.redirect_stdout(inspect_output):
                inspect_code = main(["inspect", "--input", str(pdb), "--strict"])
            self.assertEqual(0, inspect_code)
            inspection = json.loads(inspect_output.getvalue())
            self.assertEqual(1, inspection["candidate_count"])
            self.assertEqual(1, inspection["target_pool_validation"]["accepted_count"])

            csv_path = root / "descriptors.csv"
            extract_output = io.StringIO()
            with contextlib.redirect_stdout(extract_output):
                extract_code = main(
                    [
                        "extract",
                        "--input",
                        str(pdb),
                        "--strict",
                        "--output",
                        str(csv_path),
                    ]
                )
            self.assertEqual(0, extract_code)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(csv_path.with_suffix(".csv.provenance.json").is_file())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(FIELDNAMES, reader.fieldnames)
                rows = list(reader)
            self.assertEqual(1, len(rows))
            self.assertEqual("candidate_0001", rows[0]["pdb_name"])

    def test_rank_and_score_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_complex(root / "candidate_0001.pdb")
            second = write_complex(root / "candidate_0002.pdb")
            ranked_path = root / "ranked.tsv"
            descriptor_path = root / "descriptors.csv"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(
                    [
                        "rank",
                        "--input",
                        str(first),
                        "--input",
                        str(second),
                        "--strict",
                        "--output",
                        str(ranked_path),
                        "--descriptors-output",
                        str(descriptor_path),
                        "--top-k",
                        "1",
                    ]
                )
            self.assertEqual(0, code)
            with ranked_path.open(newline="", encoding="utf-8") as handle:
                ranked = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(["1", "2"], [row["rank_final_score_v5"] for row in ranked])
            self.assertEqual("candidate_0001", ranked[0]["pdb_name"])
            selected_path = root / "ranked.selected.tsv"
            with selected_path.open(newline="", encoding="utf-8") as handle:
                selected = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(1, len(selected))

            rescored_path = root / "rescored.tsv"
            rescored_output = io.StringIO()
            with contextlib.redirect_stdout(rescored_output):
                score_code = main(
                    [
                        "score",
                        "--input",
                        str(descriptor_path),
                        "--output",
                        str(rescored_path),
                    ]
                )
            self.assertEqual(0, score_code)
            with rescored_path.open(newline="", encoding="utf-8") as handle:
                rescored = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(
                [row["pdb_name"] for row in ranked],
                [row["pdb_name"] for row in rescored],
            )
            self.assertEqual(
                [row["final_score_v5"] for row in ranked],
                [row["final_score_v5"] for row in rescored],
            )


if __name__ == "__main__":
    unittest.main()
