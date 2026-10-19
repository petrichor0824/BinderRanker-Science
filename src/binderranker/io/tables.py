"""Stable tabular output helpers."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from binderranker.exceptions import OutputExistsError


def write_csv_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
    *,
    overwrite: bool = False,
) -> None:
    """Write a CSV atomically, refusing overwrite unless explicitly enabled."""

    output = Path(path)
    if output.exists() and not overwrite:
        raise OutputExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_delimited(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV or TSV table, inferring the delimiter from its suffix."""

    source = Path(path)
    delimiter = "\t" if source.suffix.lower() in {".tsv", ".tab"} else ","
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"table has no header: {source}")
        return [dict(row) for row in reader]


def write_delimited_atomic(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write CSV/TSV with stable blank serialization for NaN/Inf."""

    import math

    output = Path(path)
    if output.exists() and not overwrite:
        raise OutputExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    delimiter = "\t" if output.suffix.lower() in {".tsv", ".tab"} else ","
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter=delimiter,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                clean = {}
                for field in fieldnames:
                    value = row.get(field, "")
                    if isinstance(value, float) and (
                        math.isnan(value) or math.isinf(value)
                    ):
                        value = ""
                    clean[field] = value
                writer.writerow(clean)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
