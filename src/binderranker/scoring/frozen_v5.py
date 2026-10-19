"""Frozen BinderRanker v5 scoring for one complete target-specific pool.

This is a dependency-light migration of the numerical path in the immutable
``rank_candidates_v4_layered.py`` and ``rank_candidates_v5_proto.py`` sources,
using the parameters fixed by ``rank_candidates_v5_alpha_frozen.py``.  Output
scores are relative to the supplied pool; rows from different targets must
never be normalized together.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from binderranker.geometry.contact_field import FIELDNAMES

EPS = 1e-12


@dataclass(frozen=True)
class FrozenV5Config:
    profile: str = "frozen-v5-alpha"
    morph_alpha: float = 1.25
    min_mode_weight: float = 0.05
    q_low: float = 0.05
    q_high: float = 0.95
    beta_local: float = 0.0
    local_lambda: float = 1.0
    gain_plane: float = 1.0
    gain_compact: float = 1.0
    extent_variant: str = "coverage_cb_micro"
    micro_weight: float = 0.0
    extent_weight: float = 0.90
    safety_weight: float = 0.0
    roughness_weight: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_V5 = FrozenV5Config()


PREFERRED_OUTPUT_FIELDS = [
    "pdb_name",
    "filter_level",
    "filter_broad_pass",
    "filter_broad_reasons",
    "filter_medium_pass",
    "filter_medium_reasons",
    "filter_strict_pass",
    "filter_strict_reasons",
    "final_score_v5",
    "rank_final_score_v5",
    "rank_final_v5_in_broad",
    "rank_final_v5_in_medium",
    "rank_final_v5_in_strict",
    "morphology_blended_score_v5",
    "morphology_local_score_v5",
    "v5_blend_weight_line",
    "v5_blend_weight_plane",
    "v5_blend_weight_compact",
    "v5_local_weight_line",
    "v5_local_weight_plane",
    "v5_local_weight_compact",
    "extent_score_v5",
    "extent_coverage_v5",
    "extent_coverage_cb_v5",
    "extent_coverage_micro_v5",
    "extent_coverage_continuity_v5",
    "extent_coverage_cb_micro_v5",
    "v5_norm_coverage",
    "v5_norm_cb_support",
    "v5_norm_microfit_raw",
    "v5_norm_continuity",
    "final_score_v4",
    "rank_final_score_v4",
    "morphology_adaptive_score",
    "score_line",
    "score_plane",
    "score_compact",
    "score_microfit",
    "score_safety",
    "score_roughness",
    "target_effective_coverage",
    "target_contact_span_norm",
    "target_contact_neff_norm",
    "cb_supported_weight_ratio",
    "micro_fit_score_weighted",
    "contact_map_continuity_score",
    "target_linearity",
    "target_planarity",
    "target_compactness",
    "contacted_target_linearity",
    "contacted_target_planarity",
    "contacted_target_compactness",
    "error",
]


def safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except Exception:
        return default


def is_valid_row(row: dict[str, Any]) -> bool:
    error = str(row.get("error", "")).strip()
    return error == "" or error.lower() in {"nan", "none", "null"}


def finite_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    values = []
    for row in rows:
        value = safe_float(row.get(column))
        if math.isfinite(value):
            values.append(value)
    return values


def quantile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return math.nan
    q = max(0.0, min(1.0, q))
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return max(0.0, min(1.0, value))


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
    normalized = clip01((value - lower) / (upper - lower))
    return normalized if higher_better else 1.0 - normalized


def _normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in raw.values())
    if total <= EPS:
        return {key: 0.0 for key in raw}
    return {key: max(0.0, value) / total for key, value in raw.items()}


def _add_derived_metrics(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        raw8 = safe_float(row.get("raw_mean_shell_8"), 0.0)
        raw12 = safe_float(row.get("raw_mean_shell_12"), 0.0)
        if raw12 > EPS:
            row["shell_sensitivity_12_vs_8"] = (raw12 - raw8) / (raw12 + EPS)
            row["tight_contact_fraction_8_over_12"] = raw8 / (raw12 + EPS)
        else:
            row["shell_sensitivity_12_vs_8"] = 1.0
            row["tight_contact_fraction_8_over_12"] = 0.0

        effective = safe_float(row.get("effective_weight_sum"), 0.0)
        coverage = safe_float(row.get("target_effective_coverage"), 0.0)
        target_len = safe_float(row.get("target_len"), math.nan)
        neff_norm = safe_float(row.get("target_contact_neff_norm"), math.nan)
        if math.isfinite(target_len) and math.isfinite(neff_norm):
            row["contact_density_target_neff"] = effective / max(
                neff_norm * target_len, EPS
            )
        else:
            row["contact_density_target_neff"] = math.nan
        if math.isfinite(target_len):
            row["contact_density_target_covered"] = effective / max(
                coverage * target_len, EPS
            )
        else:
            row["contact_density_target_covered"] = math.nan

        center_distance = safe_float(
            row.get("micro_center_distance_weighted"), math.nan
        )
        if math.isfinite(center_distance):
            row["micro_center_distance_score"] = math.exp(
                -((center_distance - 7.5) ** 2) / (2.0 * 3.0 * 3.0)
            )
        else:
            row["micro_center_distance_score"] = math.nan
        row["contacted_linearity_score"] = safe_float(
            row.get("contacted_target_linearity"), math.nan
        )
        row["contacted_planarity_score"] = safe_float(
            row.get("contacted_target_planarity"), math.nan
        )
        row["contacted_compactness_score"] = safe_float(
            row.get("contacted_target_compactness"), math.nan
        )


def _infer_target_morphology(
    rows: list[dict[str, Any]], config: FrozenV5Config
) -> dict[str, float]:
    valid = [row for row in rows if is_valid_row(row)]
    linearity = quantile(finite_values(valid, "target_linearity"), 0.5)
    planarity = quantile(finite_values(valid, "target_planarity"), 0.5)
    compactness = quantile(finite_values(valid, "target_compactness"), 0.5)
    if not math.isfinite(linearity):
        linearity = 0.0
    if not math.isfinite(planarity):
        planarity = 0.0
    if not math.isfinite(compactness):
        compactness = 0.0
    raw = {
        "line": max(linearity, 0.0) ** config.morph_alpha,
        "plane": max(planarity, 0.0) ** config.morph_alpha,
        "compact": max(compactness, 0.0) ** config.morph_alpha,
    }
    base = _normalize_weights(raw)
    floor = max(0.0, min(config.min_mode_weight, 0.30))
    if 3.0 * floor >= 1.0:
        floor = 0.0
    mode = _normalize_weights(
        {key: floor + (1.0 - 3.0 * floor) * value for key, value in base.items()}
    )
    return {
        "target_linearity_median": linearity,
        "target_planarity_median": planarity,
        "target_compactness_median": compactness,
        "morph_alpha": config.morph_alpha,
        "min_mode_weight": config.min_mode_weight,
        "mode_weight_line": mode["line"],
        "mode_weight_plane": mode["plane"],
        "mode_weight_compact": mode["compact"],
    }


def _component_weights() -> dict[str, dict[str, tuple[str, float, bool]]]:
    return {
        "score_line": {
            "effective_contact_strength": ("effective_weight_sum", 0.17, True),
            "target_coverage": ("target_effective_coverage", 0.19, True),
            "target_span": ("target_contact_span_norm", 0.19, True),
            "target_neff": ("target_contact_neff_norm", 0.08, True),
            "contact_map_continuity": ("contact_map_continuity_score", 0.09, True),
            "binder_active_smooth": ("binder_field_active_smoothness", 0.08, True),
            "contacted_target_linearity": ("contacted_linearity_score", 0.06, True),
            "cb_closer": ("cb_closer_weight_ratio", 0.05, True),
            "front_facing": ("front_facing_weight_ratio", 0.04, True),
            "tight_contact_fraction": ("tight_contact_fraction_8_over_12", 0.04, True),
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.07, False),
            "low_contact_jump": ("contact_map_jump_fraction", 0.04, False),
            "low_shell_sensitivity": ("shell_sensitivity_12_vs_8", 0.03, False),
        },
        "score_plane": {
            "micro_fit": ("micro_fit_score_weighted", 0.20, True),
            "micro_plane_alignment": ("micro_plane_alignment_weighted", 0.14, True),
            "micro_planarity": ("micro_planarity_mean", 0.10, True),
            "micro_center_distance": ("micro_center_distance_score", 0.08, True),
            "target_coverage": ("target_effective_coverage", 0.13, True),
            "effective_contact_strength": ("effective_weight_sum", 0.12, True),
            "contacted_target_planarity": ("contacted_planarity_score", 0.08, True),
            "contact_map_continuity": ("contact_map_continuity_score", 0.05, True),
            "cb_closer": ("cb_closer_weight_ratio", 0.04, True),
            "front_facing": ("front_facing_weight_ratio", 0.03, True),
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.07, False),
            "low_contact_jump": ("contact_map_jump_fraction", 0.03, False),
        },
        "score_compact": {
            "effective_contact_strength": ("effective_weight_sum", 0.18, True),
            "contact_density_neff": ("contact_density_target_neff", 0.15, True),
            "target_coverage": ("target_effective_coverage", 0.10, True),
            "micro_fit": ("micro_fit_score_weighted", 0.10, True),
            "contacted_target_compactness": ("contacted_compactness_score", 0.07, True),
            "cb_closer": ("cb_closer_weight_ratio", 0.08, True),
            "front_facing": ("front_facing_weight_ratio", 0.06, True),
            "binder_core_smooth": ("binder_field_core_smoothness", 0.07, True),
            "binder_active_smooth": ("binder_field_active_smoothness", 0.04, True),
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.12, False),
            "low_contact_jump": ("contact_map_jump_fraction", 0.04, False),
            "low_shell_sensitivity": ("shell_sensitivity_12_vs_8", 0.04, False),
        },
        "score_roughness": {
            "binder_active_smooth": ("binder_field_active_smoothness", 0.32, True),
            "binder_core_smooth": ("binder_field_core_smoothness", 0.24, True),
            "target_active_smooth": ("target_field_active_smoothness", 0.12, True),
            "target_core_smooth": ("target_field_core_smoothness", 0.08, True),
            "contact_map_continuity": ("contact_map_continuity_score", 0.14, True),
            "low_contact_jump": ("contact_map_jump_fraction", 0.05, False),
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.05, False),
        },
        "score_microfit": {
            "micro_fit": ("micro_fit_score_weighted", 0.38, True),
            "micro_plane_alignment": ("micro_plane_alignment_weighted", 0.22, True),
            "micro_planarity": ("micro_planarity_mean", 0.14, True),
            "micro_center_distance": ("micro_center_distance_score", 0.10, True),
            "effective_contact_strength": ("effective_weight_sum", 0.05, True),
            "target_coverage": ("target_effective_coverage", 0.05, True),
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.04, False),
            "low_contact_jump": ("contact_map_jump_fraction", 0.02, False),
        },
        "score_safety": {
            "low_backfacing_cb_far": ("backfacing_cb_far_weight_ratio", 0.24, False),
            "cb_closer": ("cb_closer_weight_ratio", 0.16, True),
            "low_contact_jump": ("contact_map_jump_fraction", 0.16, False),
            "binder_active_smooth": ("binder_field_active_smoothness", 0.14, True),
            "tight_contact_fraction": ("tight_contact_fraction_8_over_12", 0.10, True),
            "low_shell_sensitivity": ("shell_sensitivity_12_vs_8", 0.10, False),
            "front_facing": ("front_facing_weight_ratio", 0.06, True),
            "low_clash_pairs": ("clash_pairs", 0.04, False),
        },
    }


def _add_v4_scores(
    rows: list[dict[str, Any]], morph: dict[str, float], config: FrozenV5Config
) -> dict[tuple[str, bool], tuple[float, float]]:
    components = _component_weights()
    specs: dict[tuple[str, bool], None] = {}
    for component in components.values():
        for column, _, higher in component.values():
            specs[(column, higher)] = None
    valid = [row for row in rows if is_valid_row(row)]
    ranges = {
        (column, higher): (
            quantile(finite_values(valid, column), config.q_low),
            quantile(finite_values(valid, column), config.q_high),
        )
        for column, higher in specs
    }
    normalized: dict[tuple[int, str, bool], float] = {}
    for index, row in enumerate(rows):
        for (column, higher), (lower, upper) in ranges.items():
            normalized[(index, column, higher)] = robust_norm_value(
                safe_float(row.get(column)), lower, upper, higher_better=higher
            )
    for index, row in enumerate(rows):
        if not is_valid_row(row):
            for name in list(components) + [
                "morphology_adaptive_score",
                "final_score_v4",
                "rank_consensus_score",
            ]:
                row[name] = math.nan
            continue
        for name, component in components.items():
            total_weight = sum(max(0.0, weight) for _, weight, _ in component.values())
            score = 0.0
            for column, weight, higher in component.values():
                score += (max(0.0, weight) / total_weight) * normalized.get(
                    (index, column, higher), 0.5
                )
            row[name] = clip01(score)
        adaptive = (
            morph["mode_weight_line"] * safe_float(row.get("score_line"), 0.0)
            + morph["mode_weight_plane"] * safe_float(row.get("score_plane"), 0.0)
            + morph["mode_weight_compact"] * safe_float(row.get("score_compact"), 0.0)
        )
        row["morphology_adaptive_score"] = clip01(adaptive)
        row["final_score_v4"] = clip01(
            0.85 * safe_float(row.get("morphology_adaptive_score"), 0.0)
            + 0.10 * safe_float(row.get("score_safety"), 0.0)
            + 0.05 * safe_float(row.get("score_roughness"), 0.0)
        )
    _add_rank_consensus(rows, morph)
    return ranges


def _add_rank_consensus(rows: list[dict[str, Any]], morph: dict[str, float]) -> None:
    valid = [row for row in rows if is_valid_row(row)]
    score_columns = [
        ("score_line", morph["mode_weight_line"]),
        ("score_plane", morph["mode_weight_plane"]),
        ("score_compact", morph["mode_weight_compact"]),
        ("score_roughness", 0.10),
        ("score_microfit", 0.05),
        ("score_safety", 0.10),
    ]
    total = sum(weight for _, weight in score_columns)
    score_columns = [(column, weight / total) for column, weight in score_columns]
    percentiles: dict[str, dict[str, float]] = {}
    for column, _ in score_columns:
        ordered = sorted(valid, key=lambda row: safe_float(row.get(column), -1e99))
        denominator = max(len(ordered) - 1, 1)
        percentiles[column] = {
            str(row.get("pdb_name", "")): index / denominator
            for index, row in enumerate(ordered)
        }
    for row in rows:
        if not is_valid_row(row):
            row["rank_consensus_score"] = math.nan
            continue
        name = str(row.get("pdb_name", ""))
        score = 0.0
        for column, weight in score_columns:
            score += weight * percentiles[column].get(name, 0.0)
        row["rank_consensus_score"] = clip01(score)


def _filter_definition() -> dict[str, dict[str, tuple[str, str, float]]]:
    return {
        "broad": {
            "effective_weight_sum": ("effective_weight_sum", "min", 0.10),
            "target_effective_coverage": ("target_effective_coverage", "min", 0.10),
            "target_contact_span_norm": ("target_contact_span_norm", "min", 0.10),
            "backfacing_cb_far_weight_ratio": ("backfacing_cb_far_weight_ratio", "max", 0.95),
            "binder_field_active_roughness": ("binder_field_active_roughness", "max", 0.95),
            "contact_map_jump_fraction": ("contact_map_jump_fraction", "max", 0.95),
            "shell_sensitivity_12_vs_8": ("shell_sensitivity_12_vs_8", "max", 0.98),
            "clash_pairs": ("clash_pairs", "max", 0.99),
        },
        "medium": {
            "effective_weight_sum": ("effective_weight_sum", "min", 0.40),
            "target_effective_coverage": ("target_effective_coverage", "min", 0.40),
            "target_contact_span_norm": ("target_contact_span_norm", "min", 0.40),
            "backfacing_cb_far_weight_ratio": ("backfacing_cb_far_weight_ratio", "max", 0.85),
            "binder_field_active_roughness": ("binder_field_active_roughness", "max", 0.85),
            "contact_map_jump_fraction": ("contact_map_jump_fraction", "max", 0.85),
            "shell_sensitivity_12_vs_8": ("shell_sensitivity_12_vs_8", "max", 0.90),
            "cb_closer_weight_ratio": ("cb_closer_weight_ratio", "min", 0.20),
            "contact_map_continuity_score": ("contact_map_continuity_score", "min", 0.20),
            "clash_pairs": ("clash_pairs", "max", 0.99),
        },
        "strict": {
            "effective_weight_sum": ("effective_weight_sum", "min", 0.60),
            "target_effective_coverage": ("target_effective_coverage", "min", 0.60),
            "target_contact_span_norm": ("target_contact_span_norm", "min", 0.55),
            "backfacing_cb_far_weight_ratio": ("backfacing_cb_far_weight_ratio", "max", 0.75),
            "binder_field_active_roughness": ("binder_field_active_roughness", "max", 0.75),
            "contact_map_jump_fraction": ("contact_map_jump_fraction", "max", 0.75),
            "shell_sensitivity_12_vs_8": ("shell_sensitivity_12_vs_8", "max", 0.80),
            "cb_closer_weight_ratio": ("cb_closer_weight_ratio", "min", 0.35),
            "contact_map_continuity_score": ("contact_map_continuity_score", "min", 0.35),
            "score_safety": ("score_safety", "min", 0.35),
            "clash_pairs": ("clash_pairs", "max", 0.99),
        },
    }


def _add_filters(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    valid = [row for row in rows if is_valid_row(row)]
    thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    for layer, rules in _filter_definition().items():
        thresholds[layer] = {}
        for rule_name, (column, mode, q) in rules.items():
            threshold = quantile(finite_values(valid, column), q)
            if column == "clash_pairs":
                threshold = max(0.0, threshold)
            thresholds[layer][rule_name] = {
                "column": column,
                "mode": mode,
                "quantile": q,
                "threshold": threshold,
            }
    for row in rows:
        if not is_valid_row(row):
            for layer in ("broad", "medium", "strict"):
                row[f"filter_{layer}_pass"] = "NO"
                row[f"filter_{layer}_reasons"] = "error_not_empty"
            row["filter_level"] = "FAIL"
            continue
        for layer in ("broad", "medium", "strict"):
            reasons = []
            for spec in thresholds[layer].values():
                value = safe_float(row.get(spec["column"]))
                threshold = spec["threshold"]
                if not math.isfinite(value) or not math.isfinite(threshold):
                    continue
                if spec["mode"] == "min" and value < threshold:
                    reasons.append(f"low_{spec['column']}")
                elif spec["mode"] == "max" and value > threshold:
                    reasons.append(f"high_{spec['column']}")
            row[f"filter_{layer}_pass"] = "YES" if not reasons else "NO"
            row[f"filter_{layer}_reasons"] = ";".join(reasons)
        if row["filter_strict_pass"] == "YES":
            row["filter_level"] = "STRICT"
        elif row["filter_medium_pass"] == "YES":
            row["filter_level"] = "MEDIUM"
        elif row["filter_broad_pass"] == "YES":
            row["filter_level"] = "BROAD"
        else:
            row["filter_level"] = "FAIL"
    return thresholds


def _normalize3(line: float, plane: float, compact: float) -> tuple[float, float, float]:
    values = [max(0.0, line), max(0.0, plane), max(0.0, compact)]
    total = sum(values)
    if total <= EPS:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def _floor_mode_weights(
    weights: tuple[float, float, float], floor: float
) -> tuple[float, float, float]:
    floor = max(0.0, min(float(floor), 0.30))
    if 3.0 * floor >= 1.0:
        floor = 0.0
    return _normalize3(*(floor + (1.0 - 3.0 * floor) * weight for weight in weights))


def _local_mode_weights(
    row: dict[str, Any], config: FrozenV5Config
) -> tuple[float, float, float]:
    linearity = safe_float(row.get("contacted_linearity_score"), math.nan)
    planarity = safe_float(row.get("contacted_planarity_score"), math.nan)
    compactness = safe_float(row.get("contacted_compactness_score"), math.nan)
    linearity = linearity if math.isfinite(linearity) else 0.0
    planarity = planarity if math.isfinite(planarity) else 0.0
    compactness = compactness if math.isfinite(compactness) else 0.0
    base = _normalize3(
        max(linearity, 0.0) ** config.morph_alpha,
        max(planarity * config.gain_plane, 0.0) ** config.morph_alpha,
        max(compactness * config.gain_compact, 0.0) ** config.morph_alpha,
    )
    lam = max(0.0, min(1.0, config.local_lambda))
    neutral = 1.0 / 3.0
    shrunk = tuple(neutral + lam * (weight - neutral) for weight in base)
    return _floor_mode_weights(shrunk, config.min_mode_weight)  # type: ignore[arg-type]


def _geometric_mean(values: list[float]) -> float:
    clipped = [max(0.0, min(1.0, value)) if math.isfinite(value) else math.nan for value in values]
    if not clipped or any(not math.isfinite(value) for value in clipped):
        return math.nan
    product = 1.0
    for value in clipped:
        product *= max(0.0, value)
    return product ** (1.0 / len(clipped))


def _normalized_column(
    rows: list[dict[str, Any]], column: str, config: FrozenV5Config
) -> tuple[list[float], tuple[float, float]]:
    valid_values = [
        safe_float(row.get(column))
        for row in rows
        if is_valid_row(row) and math.isfinite(safe_float(row.get(column)))
    ]
    lower = quantile(valid_values, config.q_low)
    upper = quantile(valid_values, config.q_high)
    normalized = []
    for row in rows:
        value = safe_float(row.get(column))
        if not is_valid_row(row) or not math.isfinite(value):
            normalized.append(math.nan)
        else:
            normalized.append(robust_norm_value(value, lower, upper))
    return normalized, (lower, upper)


def _add_v5_scores(
    rows: list[dict[str, Any]], morph: dict[str, float], config: FrozenV5Config
) -> dict[str, tuple[float, float]]:
    coverage, coverage_range = _normalized_column(rows, "target_effective_coverage", config)
    cb_support, cb_range = _normalized_column(rows, "cb_supported_weight_ratio", config)
    microfit, micro_range = _normalized_column(rows, "micro_fit_score_weighted", config)
    continuity, continuity_range = _normalized_column(rows, "contact_map_continuity_score", config)
    global_weights = (
        morph["mode_weight_line"],
        morph["mode_weight_plane"],
        morph["mode_weight_compact"],
    )
    beta = max(0.0, min(1.0, config.beta_local))
    morphology_weight = max(
        0.0,
        1.0
        - config.micro_weight
        - config.extent_weight
        - config.safety_weight
        - config.roughness_weight,
    )
    variants = {
        "coverage": "extent_coverage_v5",
        "coverage_cb": "extent_coverage_cb_v5",
        "coverage_micro": "extent_coverage_micro_v5",
        "coverage_continuity": "extent_coverage_continuity_v5",
        "coverage_cb_micro": "extent_coverage_cb_micro_v5",
    }
    for index, row in enumerate(rows):
        if not is_valid_row(row):
            for column in PREFERRED_OUTPUT_FIELDS[13:31]:
                row[column] = math.nan
            continue
        local = _local_mode_weights(row, config)
        blended = _normalize3(
            (1.0 - beta) * global_weights[0] + beta * local[0],
            (1.0 - beta) * global_weights[1] + beta * local[1],
            (1.0 - beta) * global_weights[2] + beta * local[2],
        )
        for label, value in zip(("line", "plane", "compact"), local):
            row[f"v5_local_weight_{label}"] = value
        for label, value in zip(("line", "plane", "compact"), blended):
            row[f"v5_blend_weight_{label}"] = value
        component_scores = (
            safe_float(row.get("score_line"), 0.0),
            safe_float(row.get("score_plane"), 0.0),
            safe_float(row.get("score_compact"), 0.0),
        )
        row["morphology_local_score_v5"] = clip01(
            local[0] * component_scores[0]
            + local[1] * component_scores[1]
            + local[2] * component_scores[2]
        )
        row["morphology_blended_score_v5"] = clip01(
            blended[0] * component_scores[0]
            + blended[1] * component_scores[1]
            + blended[2] * component_scores[2]
        )
        row["v5_norm_coverage"] = coverage[index]
        row["v5_norm_cb_support"] = cb_support[index]
        row["v5_norm_microfit_raw"] = microfit[index]
        row["v5_norm_continuity"] = continuity[index]
        row["extent_coverage_v5"] = coverage[index]
        row["extent_coverage_cb_v5"] = _geometric_mean(
            [coverage[index], cb_support[index]]
        )
        row["extent_coverage_micro_v5"] = _geometric_mean(
            [coverage[index], microfit[index]]
        )
        row["extent_coverage_continuity_v5"] = _geometric_mean(
            [coverage[index], continuity[index]]
        )
        row["extent_coverage_cb_micro_v5"] = _geometric_mean(
            [coverage[index], cb_support[index], microfit[index]]
        )
        row["extent_score_v5"] = safe_float(
            row.get(variants[config.extent_variant]), 0.0
        )
        row["final_score_v5"] = clip01(
            morphology_weight * safe_float(row.get("morphology_blended_score_v5"), 0.0)
            + config.micro_weight * safe_float(row.get("score_microfit"), 0.0)
            + config.extent_weight * safe_float(row.get("extent_score_v5"), 0.0)
            + config.safety_weight * safe_float(row.get("score_safety"), 0.0)
            + config.roughness_weight * safe_float(row.get("score_roughness"), 0.0)
        )
    return {
        "target_effective_coverage": coverage_range,
        "cb_supported_weight_ratio": cb_range,
        "micro_fit_score_weighted": micro_range,
        "contact_map_continuity_score": continuity_range,
    }


def _add_ranks(rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if is_valid_row(row)]
    for column in (
        "final_score_v4",
        "morphology_adaptive_score",
        "rank_consensus_score",
        "score_line",
        "score_plane",
        "score_compact",
        "score_roughness",
        "score_microfit",
        "score_safety",
    ):
        ordered = sorted(valid, key=lambda row: safe_float(row.get(column), -1e99), reverse=True)
        for rank, row in enumerate(ordered, 1):
            row[f"rank_{column}"] = rank
        for row in rows:
            if not is_valid_row(row):
                row[f"rank_{column}"] = ""
    for layer in ("broad", "medium", "strict"):
        passed = [row for row in valid if row.get(f"filter_{layer}_pass") == "YES"]
        ordered = sorted(
            passed, key=lambda row: safe_float(row.get("final_score_v4"), -1e99), reverse=True
        )
        for rank, row in enumerate(ordered, 1):
            row[f"rank_final_in_{layer}"] = rank
        for row in rows:
            if row.get(f"filter_{layer}_pass") != "YES":
                row[f"rank_final_in_{layer}"] = ""

    for column in (
        "final_score_v5",
        "morphology_local_score_v5",
        "morphology_blended_score_v5",
        "extent_score_v5",
    ):
        ordered = sorted(
            valid,
            key=lambda row: (
                -safe_float(row.get(column), -1e99),
                str(row.get("pdb_name", "")),
            ),
        )
        for rank, row in enumerate(ordered, 1):
            row[f"rank_{column}"] = rank
    for layer in ("broad", "medium", "strict"):
        passed = [row for row in valid if row.get(f"filter_{layer}_pass") == "YES"]
        ordered = sorted(
            passed,
            key=lambda row: (
                -safe_float(row.get("final_score_v5"), -1e99),
                str(row.get("pdb_name", "")),
            ),
        )
        for rank, row in enumerate(ordered, 1):
            row[f"rank_final_v5_in_{layer}"] = rank
        for row in rows:
            if row.get(f"filter_{layer}_pass") != "YES":
                row[f"rank_final_v5_in_{layer}"] = ""


def ordered_output_fields(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    present: list[str] = []
    for row in rows:
        for field in row:
            if field not in present:
                present.append(field)
    return [field for field in PREFERRED_OUTPUT_FIELDS if field in present] + [
        field for field in present if field not in PREFERRED_OUTPUT_FIELDS
    ]


def score_pool(
    input_rows: list[dict[str, Any]], config: FrozenV5Config = FROZEN_V5
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score one complete target-specific descriptor pool using frozen v5."""

    if not input_rows:
        raise ValueError("descriptor pool is empty")
    rows = [dict(row) for row in input_rows]
    required = set(FIELDNAMES)
    for index, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"descriptor row {index} is missing frozen extractor fields: "
                + ", ".join(missing)
            )
    names = [str(row.get("pdb_name", "")).strip() for row in rows]
    if any(not name for name in names):
        raise ValueError("every descriptor row must have a non-empty pdb_name")
    if len(set(names)) != len(names):
        raise ValueError("duplicate pdb_name values are not allowed")
    if config.extent_variant not in {
        "coverage",
        "coverage_cb",
        "coverage_micro",
        "coverage_continuity",
        "coverage_cb_micro",
    }:
        raise ValueError(f"unsupported frozen extent variant: {config.extent_variant}")
    fixed = (
        config.micro_weight
        + config.extent_weight
        + config.safety_weight
        + config.roughness_weight
    )
    if min(
        config.micro_weight,
        config.extent_weight,
        config.safety_weight,
        config.roughness_weight,
    ) < -EPS or fixed > 1.0 + EPS:
        raise ValueError("invalid frozen v5 outer weights")

    _add_derived_metrics(rows)
    morphology = _infer_target_morphology(rows, config)
    v4_ranges = _add_v4_scores(rows, morphology, config)
    thresholds = _add_filters(rows)
    v5_ranges = _add_v5_scores(rows, morphology, config)
    _add_ranks(rows)
    metadata = {
        "scientific_profile": config.profile,
        "pool_size": len(rows),
        "valid_count": sum(is_valid_row(row) for row in rows),
        "invalid_count": sum(not is_valid_row(row) for row in rows),
        "config": config.as_dict(),
        "target_morphology": morphology,
        "v4_normalization_ranges": {
            f"{column}|{'higher' if higher else 'lower'}": [lower, upper]
            for (column, higher), (lower, upper) in v4_ranges.items()
        },
        "v5_normalization_ranges": {
            column: [lower, upper] for column, (lower, upper) in v5_ranges.items()
        },
        "filter_thresholds": thresholds,
    }
    return rows, metadata


def ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the valid pool in its frozen continuous-rank order."""

    return sorted(
        (dict(row) for row in rows if is_valid_row(row)),
        key=lambda row: (
            int(row["rank_final_score_v5"]),
            str(row.get("pdb_name", "")),
        ),
    )
