from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


class FrozenSourceHashTests(unittest.TestCase):
    def test_frozen_sources_match_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        directory = root / "reference" / "frozen_v1"
        entries = {}
        for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            entries[name] = digest
        self.assertEqual(4, len(entries))
        for name, expected in entries.items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            self.assertEqual(expected, actual, name)


if __name__ == "__main__":
    unittest.main()
