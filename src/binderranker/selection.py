"""Budget-based selection from an already validated ranking."""

from __future__ import annotations

import math
from typing import Any


def select_top(
    ranked_rows: list[dict[str, Any]],
    *,
    top_k: int | None = None,
    top_fraction: float | None = None,
) -> list[dict[str, Any]]:
    if (top_k is None) == (top_fraction is None):
        raise ValueError("provide exactly one of top_k or top_fraction")
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        count = min(top_k, len(ranked_rows))
    else:
        assert top_fraction is not None
        if not 0.0 <= top_fraction <= 1.0:
            raise ValueError("top_fraction must be between 0 and 1")
        count = min(len(ranked_rows), math.ceil(len(ranked_rows) * top_fraction))
    return [dict(row) for row in ranked_rows[:count]]

