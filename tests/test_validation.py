from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from binderranker.exceptions import PDBValidationError, TargetPoolValidationError
from binderranker.io.pdb import parse_structure
from binderranker.validation import validate_target_pool

from tests.helpers import write_complex


class TargetValidationTests(unittest.TestCase):
    def test_residue_renumbering_and_small_coordinate_change_are_accepted(self) -> None:
        angle = np.deg2rad(12.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = parse_structure(write_complex(root / "reference.pdb"))
            shifted = parse_structure(
                write_complex(
                    root / "shifted.pdb",
                    target_offset=100,
                    target_rotation=rotation,
                    target_translation=np.array([11.0, -4.0, 3.0]),
                    target_noise=0.02,
                )
            )
            report = validate_target_pool([reference, shifted], rmsd_tolerance=0.1)
        self.assertEqual(2, report["accepted_count"])
        self.assertEqual(1, report["residue_id_mismatch_count"])
        self.assertLess(report["max_ca_rmsd"], 0.1)

    def test_different_target_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = parse_structure(write_complex(root / "reference.pdb"))
            different = parse_structure(
                write_complex(root / "different.pdb", changed_target_resname=True)
            )
            with self.assertRaises(TargetPoolValidationError):
                validate_target_pool([reference, different], rmsd_tolerance=1.0)

    def test_strict_validation_rejects_altloc_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdb = write_complex(
                Path(directory) / "altloc.pdb", altloc_first_atom=True
            )
            legacy = parse_structure(pdb, strict=False)
            self.assertGreater(legacy.diagnostics.altloc_records, 0)
            with self.assertRaises(PDBValidationError):
                parse_structure(pdb, strict=True)


if __name__ == "__main__":
    unittest.main()

