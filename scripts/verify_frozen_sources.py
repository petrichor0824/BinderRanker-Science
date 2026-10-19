#!/usr/bin/env python3
"""Verify the immutable historical BinderRanker source snapshot."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed manifest line {line_number}: {raw!r}")
        entries[parts[1].strip()] = parts[0].lower()
    return entries


def verify(directory: Path) -> list[str]:
    manifest = directory / "SHA256SUMS.txt"
    failures: list[str] = []
    for name, expected in parse_manifest(manifest).items():
        source = directory / name
        if not source.is_file():
            failures.append(f"MISSING {name}")
            continue
        actual = sha256_file(source)
        if actual != expected:
            failures.append(f"MISMATCH {name}: expected={expected} actual={actual}")
    return failures


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=repository / "reference" / "frozen_v1",
        help="directory containing SHA256SUMS.txt and frozen sources",
    )
    args = parser.parse_args(argv)
    failures = verify(args.directory)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"PASS verified {len(parse_manifest(args.directory / 'SHA256SUMS.txt'))} frozen sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

