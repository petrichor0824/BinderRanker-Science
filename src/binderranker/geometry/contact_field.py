"""Frozen-v1-compatible contact-field descriptor implementation.

Operation order, strict inequalities, fallback behavior, and constants are kept
aligned with reference/frozen_v1/contact_field_v3.py.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from binderranker.config import FROZEN_GEOMETRY_V1, GeometryConfig
from binderranker.io.pdb import ParsedPDB, Residue, parse_structure

EPS = 1e-8


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def pseudo_cb_coord(res: Residue) -> Optional[np.ndarray]:
    if res.has("CB"):
        return res.atoms["CB"]
    if not (res.has("N") and res.has("CA") and res.has("C")):
        return None
    n, ca, c = res.atoms["N"], res.atoms["CA"], res.atoms["C"]
    a = n - ca
    b = c - ca
    cvec = np.cross(a, b)
    cb = ca + (-0.58273431 * a + 0.56802827 * b - 0.54067466 * cvec)
    return cb


def pseudo_cb_direction(res: Residue) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cb = pseudo_cb_coord(res)
    if cb is None:
        return None, None
    u = cb - res.ca
    norm = np.linalg.norm(u)
    if norm < EPS:
        return cb, None
    return cb, u / norm


def residue_arrays(residues: list[Residue]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ca = np.array([r.ca for r in residues], dtype=float)
    cb_list: list[np.ndarray] = []
    dir_list: list[np.ndarray] = []
    for res in residues:
        cb, u = pseudo_cb_direction(res)
        if cb is None or u is None:
            cb_list.append(res.ca.copy())
            dir_list.append(np.array([np.nan, np.nan, np.nan], dtype=float))
        else:
            cb_list.append(cb)
            dir_list.append(u)
    return ca, np.array(cb_list, dtype=float), np.array(dir_list, dtype=float)


def has_valid_dir(dirs: np.ndarray) -> np.ndarray:
    return ~np.isnan(dirs).any(axis=1)


def effective_number(x: np.ndarray) -> float:
    total = float(np.sum(x))
    if total <= EPS:
        return 0.0
    return (total * total) / (float(np.sum(x * x)) + EPS)


def entropy_norm(x: np.ndarray) -> float:
    total = float(np.sum(x))
    if total <= EPS:
        return 0.0
    probabilities = x / total
    probabilities = probabilities[probabilities > EPS]
    if len(probabilities) <= 1:
        return 0.0
    return -float(np.sum(probabilities * np.log(probabilities))) / math.log(len(x))


def max_pairwise_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    return float(np.max(pairwise_dist(coords, coords)))


def pca_shape_features(coords: np.ndarray) -> tuple[float, float, float]:
    if coords.shape[0] < 3:
        return 0.0, 0.0, 0.0
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(coords.shape[0] - 1, 1)
    values, _ = np.linalg.eigh(covariance)
    values = np.sort(values)[::-1]
    l1, l2, l3 = values[0], values[1], values[2]
    if l1 <= EPS:
        return 0.0, 0.0, 0.0
    linearity = float((l1 - l2) / (l1 + EPS))
    planarity = float((l2 - l3) / (l1 + EPS))
    compactness = float(l3 / (l1 + EPS))
    return max(0.0, linearity), max(0.0, planarity), max(0.0, compactness)


def pca_plane_normal(coords: np.ndarray) -> tuple[Optional[np.ndarray], float]:
    if coords.shape[0] < 3:
        return None, 0.0
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(coords.shape[0] - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    normal = vectors[:, -1]
    normal = normal / (np.linalg.norm(normal) + EPS)
    planarity_confidence = (
        0.0 if values[1] <= EPS else 1.0 - float(values[-1] / (values[1] + EPS))
    )
    return normal, max(0.0, min(1.0, planarity_confidence))


def parse_region_specs(spec: str) -> list[str]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    return [item.strip() for item in spec.split(",") if item.strip()]


def region_mask_for_target(target: list[Residue], specs: list[str]) -> np.ndarray:
    mask = np.zeros(len(target), dtype=bool)
    for specification in specs:
        if ":" in specification:
            chain, residue_range = specification.split(":", 1)
            chain = chain.strip()
            if "-" in residue_range:
                a, b = residue_range.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(residue_range)
            for index, residue in enumerate(target):
                if residue.chain == chain and start <= residue.resid <= end:
                    mask[index] = True
        else:
            if "-" in specification:
                a, b = specification.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(specification)
            start_index = max(1, start)
            end_index = min(len(target), end)
            if start_index <= end_index:
                mask[start_index - 1 : end_index] = True
    return mask


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum <= EPS:
        return 0.0
    return float(np.sum(values * weights) / (weight_sum + EPS))


def build_spatial_edges(
    coords: np.ndarray,
    cutoff: float,
    dirs: Optional[np.ndarray] = None,
    face_aware: bool = False,
    face_floor: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(coords)
    if count < 2:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    distances = pairwise_dist(coords, coords)
    mask = np.triu((distances < cutoff) & (distances > EPS), k=1)
    edge_i, edge_j = np.where(mask)
    if len(edge_i) == 0:
        return edge_i, edge_j, np.array([], dtype=float)
    distance = distances[edge_i, edge_j]
    spatial_weight = np.exp(-(distance**2) / (2.0 * (cutoff / 2.0) ** 2))
    if face_aware and dirs is not None:
        valid = has_valid_dir(dirs)
        both_valid = valid[edge_i] & valid[edge_j]
        face_weight = np.ones(len(edge_i), dtype=float)
        if np.any(both_valid):
            dots = np.sum(dirs[edge_i[both_valid]] * dirs[edge_j[both_valid]], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            face_weight[both_valid] = face_floor + (1.0 - face_floor) * np.clip(
                dots, 0.0, 1.0
            )
        edge_weight = spatial_weight * face_weight
    else:
        edge_weight = spatial_weight
    return edge_i, edge_j, edge_weight


def graph_field_roughness(
    coords: np.ndarray,
    field: np.ndarray,
    cutoff: float,
    dirs: Optional[np.ndarray] = None,
    face_aware: bool = False,
    contact_threshold: float = 0.1,
) -> dict[str, float]:
    edge_i, edge_j, edge_weight = build_spatial_edges(
        coords=coords, cutoff=cutoff, dirs=dirs, face_aware=face_aware
    )
    if len(edge_i) == 0:
        return {
            "active_roughness": 0.0,
            "active_smoothness": 0.0,
            "active_edges": 0,
            "core_roughness": 0.0,
            "core_smoothness": 0.0,
            "core_edges": 0,
        }
    field_i, field_j = field[edge_i], field[edge_j]
    active_mask = (field_i > contact_threshold) | (field_j > contact_threshold)
    core_mask = (field_i > contact_threshold) & (field_j > contact_threshold)

    def calculate(mask: np.ndarray) -> tuple[float, float, int]:
        if not np.any(mask):
            return 0.0, 0.0, 0
        difference = np.abs(field_i[mask] - field_j[mask])
        average = 0.5 * (field_i[mask] + field_j[mask]) + EPS
        weights = edge_weight[mask]
        roughness = float(
            np.sum(weights * difference) / (np.sum(weights * average) + EPS)
        )
        return roughness, math.exp(-roughness), int(np.sum(mask))

    active_rough, active_smooth, active_edges = calculate(active_mask)
    core_rough, core_smooth, core_edges = calculate(core_mask)
    return {
        "active_roughness": active_rough,
        "active_smoothness": active_smooth,
        "active_edges": active_edges,
        "core_roughness": core_rough,
        "core_smoothness": core_smooth,
        "core_edges": core_edges,
    }


def compute_contact_weights(
    binder: list[Residue], target: list[Residue], config: GeometryConfig
) -> dict[str, np.ndarray]:
    b_ca, b_cb, b_dirs = residue_arrays(binder)
    t_ca, _t_cb, t_dirs = residue_arrays(target)
    distance = pairwise_dist(b_ca, t_ca)
    cb_distance = pairwise_dist(b_cb, t_ca)
    raw_mask = distance < config.shell_max
    distance_weight = (
        np.exp(
            -((distance - config.ideal_dist) ** 2)
            / (2.0 * config.density_sigma * config.density_sigma)
        )
        * raw_mask
    )
    distance_weight[distance < config.clash_cutoff] *= 0.25

    vector = t_ca[None, :, :] - b_ca[:, None, :]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    vector_unit = vector / (vector_norm + EPS)
    valid_binder_direction = has_valid_dir(b_dirs)
    dot = np.sum(b_dirs[:, None, :] * vector_unit, axis=-1)
    dot[~valid_binder_direction, :] = 0.0
    cb_delta = cb_distance - distance

    front_score = sigmoid((dot - config.front_dot) / max(config.dot_sigma, EPS))
    orientation_weight = config.tangent_floor + (1.0 - config.tangent_floor) * front_score
    back_orientation_score = sigmoid(
        (config.back_dot - dot) / max(config.dot_sigma, EPS)
    )
    cb_far_score = sigmoid(
        (cb_delta - config.cb_far_delta) / max(config.cb_delta_sigma, EPS)
    )
    back_suspicion = back_orientation_score * cb_far_score * raw_mask
    orientation_weight = orientation_weight * (
        1.0 - (1.0 - config.back_penalty) * back_suspicion
    )
    orientation_weight[~valid_binder_direction, :] = 1.0

    cb_support_score = sigmoid(
        (config.cb_support_cutoff - cb_distance) / max(config.cb_sigma, EPS)
    )
    cb_support_weight = 0.65 + 0.35 * cb_support_score
    effective_weight = distance_weight * orientation_weight * cb_support_weight

    return {
        "b_ca": b_ca,
        "t_ca": t_ca,
        "b_dirs": b_dirs,
        "t_dirs": t_dirs,
        "D": distance,
        "cb_D": cb_distance,
        "cb_delta": cb_delta,
        "raw_mask": raw_mask,
        "dist_w": distance_weight,
        "W": effective_weight,
        "dot": dot,
        "front_mask": (dot >= config.front_dot) & raw_mask,
        "tangential_mask": (dot > config.back_dot) & (dot < config.front_dot) & raw_mask,
        "backfacing_mask": (dot <= config.back_dot) & raw_mask,
        "backfacing_cb_far_mask": (dot <= config.back_dot)
        & (cb_delta > config.cb_far_delta)
        & raw_mask,
        "cb_near_mask": (cb_distance < config.cb_near_cutoff) & raw_mask,
        "cb_supported_mask": (cb_distance < config.cb_support_cutoff) & raw_mask,
        "cb_closer_mask": (cb_delta < 0.0) & raw_mask,
    }


def weighted_ratio(base_weight: np.ndarray, mask: np.ndarray, denominator: float) -> float:
    if denominator <= EPS:
        return 0.0
    return float(np.sum(base_weight[mask]) / (denominator + EPS))


def empty_microdomain_result() -> dict[str, float]:
    return {
        "microdomain_count": 0,
        "micro_plane_alignment_weighted": 0.0,
        "micro_center_distance_weighted": 0.0,
        "micro_planarity_mean": 0.0,
        "micro_fit_score_weighted": 0.0,
    }


def compute_microdomain_plane_metrics(
    effective_weight: np.ndarray,
    binder_ca: np.ndarray,
    target_ca: np.ndarray,
    target_field: np.ndarray,
    config: GeometryConfig,
) -> dict[str, float]:
    if len(target_ca) < 3 or len(binder_ca) < 3:
        return empty_microdomain_result()
    target_internal_distance = pairwise_dist(target_ca, target_ca)
    contacted_indices = np.where(target_field > config.contact_threshold)[0]
    if len(contacted_indices) == 0:
        return empty_microdomain_result()
    if len(contacted_indices) > config.max_microdomains:
        order = np.argsort(target_field[contacted_indices])[::-1]
        contacted_indices = contacted_indices[order[: config.max_microdomains]]

    aligns: list[float] = []
    distances: list[float] = []
    planarities: list[float] = []
    fits: list[float] = []
    weights: list[float] = []
    for target_index in contacted_indices:
        target_micro_indices = np.where(
            target_internal_distance[target_index] < config.micro_radius
        )[0]
        if len(target_micro_indices) < 3:
            continue
        local_binder_weight = np.sum(effective_weight[:, target_micro_indices], axis=1)
        binder_micro_indices = np.where(
            local_binder_weight > config.contact_threshold
        )[0]
        if len(binder_micro_indices) < 3:
            top_k = min(8, len(binder_ca))
            top_indices = np.argsort(local_binder_weight)[::-1][:top_k]
            binder_micro_indices = top_indices[local_binder_weight[top_indices] > EPS]
        if len(binder_micro_indices) < 3:
            continue
        target_coords = target_ca[target_micro_indices]
        binder_coords = binder_ca[binder_micro_indices]
        target_normal, target_planarity = pca_plane_normal(target_coords)
        binder_normal, binder_planarity = pca_plane_normal(binder_coords)
        if target_normal is None or binder_normal is None:
            continue
        alignment = abs(float(np.dot(target_normal, binder_normal)))
        center_distance = float(
            np.linalg.norm(np.mean(target_coords, axis=0) - np.mean(binder_coords, axis=0))
        )
        planarity = target_planarity * binder_planarity
        distance_score = math.exp(
            -((center_distance - config.ideal_dist) ** 2)
            / (2.0 * config.micro_dist_sigma * config.micro_dist_sigma)
        )
        local_contact_weight = float(np.sum(local_binder_weight[binder_micro_indices]))
        fit_score = alignment * distance_score * planarity
        aligns.append(alignment)
        distances.append(center_distance)
        planarities.append(planarity)
        fits.append(fit_score)
        weights.append(local_contact_weight * max(planarity, 0.05))
    if len(weights) == 0 or np.sum(weights) <= EPS:
        return empty_microdomain_result()
    aligns_array, distances_array, planarities_array, fits_array, weights_array = map(
        np.array, (aligns, distances, planarities, fits, weights)
    )
    weight_sum = float(np.sum(weights_array)) + EPS
    return {
        "microdomain_count": int(len(weights_array)),
        "micro_plane_alignment_weighted": float(
            np.sum(aligns_array * weights_array) / weight_sum
        ),
        "micro_center_distance_weighted": float(
            np.sum(distances_array * weights_array) / weight_sum
        ),
        "micro_planarity_mean": float(np.mean(planarities_array)),
        "micro_fit_score_weighted": float(
            np.sum(fits_array * weights_array) / weight_sum
        ),
    }


def empty_continuity_result() -> dict[str, float]:
    return {
        "contact_map_edges": 0,
        "contact_map_continuity_mean_dist": 0.0,
        "contact_map_continuity_score": 0.0,
        "contact_map_distortion_mean": 0.0,
        "contact_map_jump_fraction": 0.0,
    }


def compute_contact_map_continuity(
    effective_weight: np.ndarray,
    binder_ca: np.ndarray,
    target_ca: np.ndarray,
    binder_field: np.ndarray,
    binder_dirs: np.ndarray,
    config: GeometryConfig,
) -> dict[str, float]:
    mapped_target = np.zeros_like(binder_ca)
    for index in range(len(binder_ca)):
        if binder_field[index] > EPS:
            mapped_target[index] = (
                np.sum(effective_weight[index, :, None] * target_ca, axis=0)
                / (binder_field[index] + EPS)
            )
    edge_i, edge_j, edge_weight = build_spatial_edges(
        binder_ca, config.graph_cutoff, dirs=binder_dirs, face_aware=True
    )
    if len(edge_i) == 0:
        return empty_continuity_result()
    active = (binder_field[edge_i] > config.contact_threshold) & (
        binder_field[edge_j] > config.contact_threshold
    )
    if not np.any(active):
        return empty_continuity_result()
    selected_i, selected_j, selected_weight = (
        edge_i[active],
        edge_j[active],
        edge_weight[active],
    )
    mapped_distance = np.linalg.norm(
        mapped_target[selected_i] - mapped_target[selected_j], axis=1
    )
    binder_distance = np.linalg.norm(
        binder_ca[selected_i] - binder_ca[selected_j], axis=1
    )
    mean_mapped_distance = weighted_mean(mapped_distance, selected_weight)
    mean_distortion = weighted_mean(
        np.abs(mapped_distance - binder_distance), selected_weight
    )
    jump_fraction = float(
        np.sum(selected_weight[mapped_distance > config.jump_cutoff])
        / (np.sum(selected_weight) + EPS)
    )
    return {
        "contact_map_edges": int(len(selected_i)),
        "contact_map_continuity_mean_dist": mean_mapped_distance,
        "contact_map_continuity_score": math.exp(-mean_mapped_distance / 10.0),
        "contact_map_distortion_mean": mean_distortion,
        "contact_map_jump_fraction": jump_fraction,
    }


def region_metrics(
    target_field: np.ndarray,
    mask: np.ndarray,
    total_weight: float,
    contact_threshold: float,
) -> dict[str, float]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "count": 0,
            "effective_coverage": 0.0,
            "weight_fraction": 0.0,
            "mean_field": 0.0,
        }
    region_field = target_field[mask]
    return {
        "count": count,
        "effective_coverage": float(np.mean(region_field > contact_threshold)),
        "weight_fraction": float(np.sum(region_field) / (total_weight + EPS)),
        "mean_field": float(np.mean(region_field)),
    }


def compute_metrics(
    structure: ParsedPDB,
    config: GeometryConfig = FROZEN_GEOMETRY_V1,
    desired_specs: Optional[list[str]] = None,
    undesired_specs: Optional[list[str]] = None,
) -> dict[str, float | int | str]:
    binder, target = structure.binder, structure.target
    desired_specs = desired_specs or []
    undesired_specs = undesired_specs or []
    binder_count, target_count = len(binder), len(target)
    data = compute_contact_weights(binder, target, config)

    binder_ca = data["b_ca"]
    target_ca = data["t_ca"]
    binder_dirs = data["b_dirs"]
    target_dirs = data["t_dirs"]
    distance = data["D"]
    cb_distance = data["cb_D"]
    cb_delta = data["cb_delta"]
    raw_mask = data["raw_mask"]
    distance_weight = data["dist_w"]
    effective_weight = data["W"]

    raw_weight_sum = float(np.sum(distance_weight))
    effective_weight_sum = float(np.sum(effective_weight))
    effective_to_raw_ratio = effective_weight_sum / (raw_weight_sum + EPS)

    binder_field = np.sum(effective_weight, axis=1)
    target_field = np.sum(effective_weight, axis=0)
    contacted_binder = binder_ca[binder_field > config.contact_threshold]
    contacted_target = target_ca[target_field > config.contact_threshold]

    binder_roughness = graph_field_roughness(
        binder_ca,
        binder_field,
        config.graph_cutoff,
        dirs=binder_dirs,
        face_aware=True,
        contact_threshold=config.contact_threshold,
    )
    target_roughness = graph_field_roughness(
        target_ca,
        target_field,
        config.graph_cutoff,
        dirs=target_dirs,
        face_aware=False,
        contact_threshold=config.contact_threshold,
    )
    continuity = compute_contact_map_continuity(
        effective_weight,
        binder_ca,
        target_ca,
        binder_field,
        binder_dirs,
        config,
    )

    target_linearity, target_planarity, target_compactness = pca_shape_features(target_ca)
    if len(contacted_target) >= 3:
        contacted_linearity, contacted_planarity, contacted_compactness = pca_shape_features(
            contacted_target
        )
    else:
        contacted_linearity, contacted_planarity, contacted_compactness = 0.0, 0.0, 0.0

    microdomain = compute_microdomain_plane_metrics(
        effective_weight, binder_ca, target_ca, target_field, config
    )
    desired_mask = region_mask_for_target(target, desired_specs)
    undesired_mask = region_mask_for_target(target, undesired_specs)
    total_target_weight = float(np.sum(target_field))
    desired = region_metrics(
        target_field, desired_mask, total_target_weight, config.contact_threshold
    )
    undesired = region_metrics(
        target_field, undesired_mask, total_target_weight, config.contact_threshold
    )

    cb_delta_weighted_mean = (
        float(np.sum(distance_weight * cb_delta) / (raw_weight_sum + EPS))
        if raw_weight_sum > EPS
        else 0.0
    )
    cb_distance_weighted_mean = (
        float(np.sum(distance_weight * cb_distance) / (raw_weight_sum + EPS))
        if raw_weight_sum > EPS
        else 0.0
    )

    return {
        "pdb_name": structure.path.stem,
        "binder_len": binder_count,
        "target_len": target_count,
        "raw_mean_shell_count": float(np.mean(np.sum(raw_mask, axis=1))),
        "raw_mean_shell_6": float(np.mean(np.sum(distance < 6.0, axis=1))),
        "raw_mean_shell_7": float(np.mean(np.sum(distance < 7.0, axis=1))),
        "raw_mean_shell_8": float(np.mean(np.sum(distance < 8.0, axis=1))),
        "raw_mean_shell_10": float(np.mean(np.sum(distance < 10.0, axis=1))),
        "raw_mean_shell_12": float(np.mean(np.sum(distance < 12.0, axis=1))),
        "raw_weight_sum": raw_weight_sum,
        "effective_weight_sum": effective_weight_sum,
        "effective_to_raw_ratio": effective_to_raw_ratio,
        "orientation_discount_ratio": max(0.0, min(1.0, 1.0 - effective_to_raw_ratio)),
        "front_facing_weight_ratio": weighted_ratio(
            distance_weight, data["front_mask"], raw_weight_sum
        ),
        "tangential_weight_ratio": weighted_ratio(
            distance_weight, data["tangential_mask"], raw_weight_sum
        ),
        "backfacing_weight_ratio": weighted_ratio(
            distance_weight, data["backfacing_mask"], raw_weight_sum
        ),
        "backfacing_cb_far_weight_ratio": weighted_ratio(
            distance_weight, data["backfacing_cb_far_mask"], raw_weight_sum
        ),
        "cb_near_weight_ratio": weighted_ratio(
            distance_weight, data["cb_near_mask"], raw_weight_sum
        ),
        "cb_supported_weight_ratio": weighted_ratio(
            distance_weight, data["cb_supported_mask"], raw_weight_sum
        ),
        "cb_closer_weight_ratio": weighted_ratio(
            distance_weight, data["cb_closer_mask"], raw_weight_sum
        ),
        "cb_delta_weighted_mean": cb_delta_weighted_mean,
        "cb_dist_weighted_mean": cb_distance_weighted_mean,
        "clash_pairs": int(np.sum(distance < config.clash_cutoff)),
        "min_distance_global": float(np.min(distance)),
        "binder_effective_hit_ratio": float(
            np.mean(binder_field > config.contact_threshold)
        ),
        "target_effective_coverage": float(
            np.mean(target_field > config.contact_threshold)
        ),
        "binder_contact_neff_norm": effective_number(binder_field) / binder_count,
        "target_contact_neff_norm": effective_number(target_field) / target_count,
        "binder_contact_entropy": entropy_norm(binder_field),
        "target_contact_entropy": entropy_norm(target_field),
        "binder_contact_span_norm": max_pairwise_distance(contacted_binder)
        / (max_pairwise_distance(binder_ca) + EPS),
        "target_contact_span": max_pairwise_distance(contacted_target),
        "target_contact_span_norm": max_pairwise_distance(contacted_target)
        / (max_pairwise_distance(target_ca) + EPS),
        "binder_field_active_roughness": binder_roughness["active_roughness"],
        "binder_field_active_smoothness": binder_roughness["active_smoothness"],
        "binder_field_active_edges": binder_roughness["active_edges"],
        "binder_field_core_roughness": binder_roughness["core_roughness"],
        "binder_field_core_smoothness": binder_roughness["core_smoothness"],
        "binder_field_core_edges": binder_roughness["core_edges"],
        "target_field_active_roughness": target_roughness["active_roughness"],
        "target_field_active_smoothness": target_roughness["active_smoothness"],
        "target_field_active_edges": target_roughness["active_edges"],
        "target_field_core_roughness": target_roughness["core_roughness"],
        "target_field_core_smoothness": target_roughness["core_smoothness"],
        "target_field_core_edges": target_roughness["core_edges"],
        **continuity,
        "target_linearity": target_linearity,
        "target_planarity": target_planarity,
        "target_compactness": target_compactness,
        "contacted_target_linearity": contacted_linearity,
        "contacted_target_planarity": contacted_planarity,
        "contacted_target_compactness": contacted_compactness,
        **microdomain,
        "desired_region_count": desired["count"],
        "desired_effective_coverage": desired["effective_coverage"],
        "desired_weight_fraction": desired["weight_fraction"],
        "desired_mean_field": desired["mean_field"],
        "undesired_region_count": undesired["count"],
        "undesired_effective_coverage": undesired["effective_coverage"],
        "undesired_weight_fraction": undesired["weight_fraction"],
        "undesired_mean_field": undesired["mean_field"],
        "error": "",
    }


def extract_structure(
    pdb_path: Path,
    *,
    binder_chain: str = "A",
    strict: bool = False,
    config: GeometryConfig = FROZEN_GEOMETRY_V1,
    desired_regions: str = "",
    undesired_regions: str = "",
) -> tuple[ParsedPDB, dict[str, float | int | str]]:
    structure = parse_structure(pdb_path, binder_chain=binder_chain, strict=strict)
    row = compute_metrics(
        structure,
        config=config,
        desired_specs=parse_region_specs(desired_regions),
        undesired_specs=parse_region_specs(undesired_regions),
    )
    return structure, row


FIELDNAMES = [
    "pdb_name", "binder_len", "target_len",
    "raw_mean_shell_count", "raw_mean_shell_6", "raw_mean_shell_7", "raw_mean_shell_8", "raw_mean_shell_10", "raw_mean_shell_12",
    "raw_weight_sum", "effective_weight_sum", "effective_to_raw_ratio", "orientation_discount_ratio",
    "front_facing_weight_ratio", "tangential_weight_ratio", "backfacing_weight_ratio", "backfacing_cb_far_weight_ratio",
    "cb_near_weight_ratio", "cb_supported_weight_ratio", "cb_closer_weight_ratio", "cb_delta_weighted_mean", "cb_dist_weighted_mean",
    "clash_pairs", "min_distance_global",
    "binder_effective_hit_ratio", "target_effective_coverage",
    "binder_contact_neff_norm", "target_contact_neff_norm", "binder_contact_entropy", "target_contact_entropy",
    "binder_contact_span_norm", "target_contact_span", "target_contact_span_norm",
    "binder_field_active_roughness", "binder_field_active_smoothness", "binder_field_active_edges",
    "binder_field_core_roughness", "binder_field_core_smoothness", "binder_field_core_edges",
    "target_field_active_roughness", "target_field_active_smoothness", "target_field_active_edges",
    "target_field_core_roughness", "target_field_core_smoothness", "target_field_core_edges",
    "contact_map_edges", "contact_map_continuity_mean_dist", "contact_map_continuity_score", "contact_map_distortion_mean", "contact_map_jump_fraction",
    "target_linearity", "target_planarity", "target_compactness",
    "contacted_target_linearity", "contacted_target_planarity", "contacted_target_compactness",
    "microdomain_count", "micro_plane_alignment_weighted", "micro_center_distance_weighted", "micro_planarity_mean", "micro_fit_score_weighted",
    "desired_region_count", "desired_effective_coverage", "desired_weight_fraction", "desired_mean_field",
    "undesired_region_count", "undesired_effective_coverage", "undesired_weight_fraction", "undesired_mean_field",
    "error",
]

