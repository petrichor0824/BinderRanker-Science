"""Frozen-v4-compatible scalar normalization helpers.

Full coherent-interface score assembly is intentionally deferred to Stage 2B.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

EPS = 1e-12


def quantile(values: Iterable[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    q = max(0.0, min(1.0, q))
    if len(finite) == 1:
        return finite[0]
    position = q * (len(finite) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def robust_norm_value(
    value: float, lower: float, upper: float, *, higher_better: bool = True
) -> float:
    if (
        not math.isfinite(value)
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or abs(upper - lower) < EPS
    ):
        return 0.5
    normalized = max(0.0, min(1.0, (value - lower) / (upper - lower)))
    return normalized if higher_better else 1.0 - normalized

