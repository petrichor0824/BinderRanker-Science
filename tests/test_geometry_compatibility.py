from __future__ import annotations

import argparse
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from binderranker.geometry.contact_field import FIELDNAMES, extract_structure

from tests.helpers import write_complex


def load_frozen_extractor():
    root = Path(__file__).resolve().parents[1]
    path = root / "reference" / "frozen_v1" / "contact_field_v3.py"
    spec = importlib.util.spec_from_file_location("frozen_contact_field_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_args() -> argparse.Namespace:
    return argparse.Namespace(
        binder_chain="A",
        shell_max=10.0,
        ideal_dist=7.5,
        density_sigma=2.0,
        clash_cutoff=3.5,
        graph_cutoff=10.0,
        contact_threshold=0.1,
        tangent_floor=0.45,
        front_dot=0.25,
        back_dot=-0.25,
        dot_sigma=0.20,
        back_penalty=0.25,
        cb_far_delta=0.8,
        cb_delta_sigma=0.5,
        cb_near_cutoff=10.0,
        cb_support_cutoff=12.0,
        cb_sigma=1.5,
        micro_radius=8.0,
        micro_dist_sigma=3.0,
        max_microdomains=80,
        jump_cutoff=15.0,
    )


class GeometryCompatibilityTests(unittest.TestCase):
    def test_fieldnames_are_identical(self) -> None:
        frozen = load_frozen_extractor()
        self.assertEqual(frozen.FIELDNAMES, FIELDNAMES)

    def test_all_synthetic_descriptor_fields_match_frozen(self) -> None:
        frozen = load_frozen_extractor()
        with tempfile.TemporaryDirectory() as directory:
            pdb = write_complex(Path(directory) / "candidate_0001.pdb")
            expected = frozen.compute_metrics(pdb, frozen_args(), [], [])
            _, actual = extract_structure(pdb)
        self.assertEqual(set(expected), set(actual) - {"error"})
        compared = 0
        for field, expected_value in expected.items():
            actual_value = actual[field]
            if isinstance(expected_value, str):
                self.assertEqual(expected_value, actual_value, field)
            elif isinstance(expected_value, int):
                self.assertEqual(expected_value, actual_value, field)
                compared += 1
            else:
                self.assertTrue(math.isfinite(float(expected_value)), field)
                self.assertAlmostEqual(
                    float(expected_value), float(actual_value), delta=1e-13, msg=field
                )
                compared += 1
        self.assertGreaterEqual(compared, 60)


if __name__ == "__main__":
    unittest.main()

