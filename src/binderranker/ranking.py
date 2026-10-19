"""Deterministic generic ranking utility; no scientific score is computed here."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def rank_rows(
    rows: Iterable[dict[str, Any]],
    *,
    score_column: str,
    id_column: str = "pdb_name",
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    if len({str(row[id_column]) for row in copied}) != len(copied):
        raise ValueError(f"duplicate {id_column} values are not rankable deterministically")
    ranked = sorted(
        copied,
        key=lambda row: (-float(row[score_column]), str(row[id_column])),
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked

