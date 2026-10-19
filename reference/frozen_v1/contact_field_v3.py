#!/usr/bin/env python3
"""contact_field_v3.py

Orientation-aware, morphology-aware, and roughness-aware contact-field descriptors
for RFdiffusion binder backbones.

This script is designed as a feature extractor, not a final scoring model.
It reads RFdiffusion-designed binder-target PDB files, treats binder_chain as binder
and all non-binder chains as target, then outputs a CSV of geometry/contact descriptors.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

EPS = 1e-8


class Residue:
    def __init__(self, chain: str, resid: int, resname: str, icode: str = ""):
        self.chain = chain if chain else "_"
        self.resid = resid
        self.resname = resname
        self.icode = icode.strip()
        self.atoms: Dict[str, np.ndarray] = {}

    def has(self, atom: str) -> bool:
        return atom in self.atoms

    @property
    def ca(self) -> np.ndarray:
        return self.atoms["CA"]

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.resid}{self.icode}"


def parse_pdb_backbone(pdb_path: Path, binder_chain: str = "A") -> Tuple[List[Residue], List[Residue]]:
    residues: Dict[Tuple[str, int, str, str], Residue] = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C", "O", "CB"}:
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "_"
            icode = line[26].strip()
            try:
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resid, icode, resname)
            if key not in residues:
                residues[key] = Residue(chain, resid, resname, icode)
            residues[key].atoms[atom] = np.array([x, y, z], dtype=float)

    binder, target = [], []
    for r in residues.values():
        if not r.has("CA"):
            continue
        if r.chain == binder_chain:
            binder.append(r)
        else:
            target.append(r)

    binder.sort(key=lambda r: (r.chain, r.resid, r.icode))
    target.sort(key=lambda r: (r.chain, r.resid, r.icode))

    if not binder:
        raise ValueError(f"No binder residues found for binder_chain={binder_chain}")
    if not target:
        raise ValueError("No target residues found. Target = all non-binder chains.")
    return binder, target


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


def pseudo_cb_direction(res: Residue) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    cb = pseudo_cb_coord(res)
    if cb is None:
        return None, None
    u = cb - res.ca
    norm = np.linalg.norm(u)
    if norm < EPS:
        return cb, None
    return cb, u / norm


def residue_arrays(residues: List[Residue]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ca = np.array([r.ca for r in residues], dtype=float)
    cb_list, dir_list = [], []
    for r in residues:
        cb, u = pseudo_cb_direction(r)
        if cb is None or u is None:
            cb_list.append(r.ca.copy())
            dir_list.append(np.array([np.nan, np.nan, np.nan], dtype=float))
        else:
            cb_list.append(cb)
            dir_list.append(u)
    return ca, np.array(cb_list, dtype=float), np.array(dir_list, dtype=float)


def has_valid_dir(dirs: np.ndarray) -> np.ndarray:
    return ~np.isnan(dirs).any(axis=1)


def effective_number(x: np.ndarray) -> float:
    s = float(np.sum(x))
    if s <= EPS:
        return 0.0
    return (s * s) / (float(np.sum(x * x)) + EPS)


def entropy_norm(x: np.ndarray) -> float:
    s = float(np.sum(x))
    if s <= EPS:
        return 0.0
    p = x / s
    p = p[p > EPS]
    if len(p) <= 1:
        return 0.0
    return -float(np.sum(p * np.log(p))) / math.log(len(x))


def max_pairwise_distance(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    return float(np.max(pairwise_dist(coords, coords)))


def pca_shape_features(coords: np.ndarray) -> Tuple[float, float, float]:
    if coords.shape[0] < 3:
        return 0.0, 0.0, 0.0
    x = coords - np.mean(coords, axis=0, keepdims=True)
    cov = x.T @ x / max(coords.shape[0] - 1, 1)
    vals, _ = np.linalg.eigh(cov)
    vals = np.sort(vals)[::-1]
    l1, l2, l3 = vals[0], vals[1], vals[2]
    if l1 <= EPS:
        return 0.0, 0.0, 0.0
    linearity = float((l1 - l2) / (l1 + EPS))
    planarity = float((l2 - l3) / (l1 + EPS))
    compactness = float(l3 / (l1 + EPS))
    return max(0.0, linearity), max(0.0, planarity), max(0.0, compactness)


def pca_plane_normal(coords: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    if coords.shape[0] < 3:
        return None, 0.0
    x = coords - np.mean(coords, axis=0, keepdims=True)
    cov = x.T @ x / max(coords.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    normal = vecs[:, -1]
    normal = normal / (np.linalg.norm(normal) + EPS)
    planarity_conf = 0.0 if vals[1] <= EPS else 1.0 - float(vals[-1] / (vals[1] + EPS))
    return normal, max(0.0, min(1.0, planarity_conf))


def parse_region_specs(spec: str) -> List[str]:
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    return [x.strip() for x in spec.split(",") if x.strip()]


def region_mask_for_target(target: List[Residue], specs: List[str]) -> np.ndarray:
    mask = np.zeros(len(target), dtype=bool)
    for sp in specs:
        if ":" in sp:
            chain, rr = sp.split(":", 1)
            chain = chain.strip()
            if "-" in rr:
                a, b = rr.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(rr)
            for i, r in enumerate(target):
                if r.chain == chain and start <= r.resid <= end:
                    mask[i] = True
        else:
            if "-" in sp:
                a, b = sp.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(sp)
            start_idx = max(1, start)
            end_idx = min(len(target), end)
            if start_idx <= end_idx:
                mask[start_idx - 1:end_idx] = True
    return mask


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = float(np.sum(weights))
    if w <= EPS:
        return 0.0
    return float(np.sum(values * weights) / (w + EPS))


def build_spatial_edges(coords: np.ndarray, cutoff: float, dirs: Optional[np.ndarray] = None,
                        face_aware: bool = False, face_floor: float = 0.35) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(coords)
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)
    D = pairwise_dist(coords, coords)
    mask = np.triu((D < cutoff) & (D > EPS), k=1)
    edge_i, edge_j = np.where(mask)
    if len(edge_i) == 0:
        return edge_i, edge_j, np.array([], dtype=float)
    d = D[edge_i, edge_j]
    spatial_w = np.exp(-(d ** 2) / (2.0 * (cutoff / 2.0) ** 2))
    if face_aware and dirs is not None:
        valid = has_valid_dir(dirs)
        both_valid = valid[edge_i] & valid[edge_j]
        face_w = np.ones(len(edge_i), dtype=float)
        if np.any(both_valid):
            dots = np.sum(dirs[edge_i[both_valid]] * dirs[edge_j[both_valid]], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            face_w[both_valid] = face_floor + (1.0 - face_floor) * np.clip(dots, 0.0, 1.0)
        edge_w = spatial_w * face_w
    else:
        edge_w = spatial_w
    return edge_i, edge_j, edge_w


def graph_field_roughness(coords: np.ndarray, field: np.ndarray, cutoff: float,
                          dirs: Optional[np.ndarray] = None, face_aware: bool = False,
                          contact_threshold: float = 0.1) -> Dict[str, float]:
    edge_i, edge_j, edge_w = build_spatial_edges(coords=coords, cutoff=cutoff, dirs=dirs, face_aware=face_aware)
    if len(edge_i) == 0:
        return {"active_roughness": 0.0, "active_smoothness": 0.0, "active_edges": 0,
                "core_roughness": 0.0, "core_smoothness": 0.0, "core_edges": 0}
    fi, fj = field[edge_i], field[edge_j]
    active_mask = (fi > contact_threshold) | (fj > contact_threshold)
    core_mask = (fi > contact_threshold) & (fj > contact_threshold)

    def calc(mask: np.ndarray) -> Tuple[float, float, int]:
        if not np.any(mask):
            return 0.0, 0.0, 0
        diff = np.abs(fi[mask] - fj[mask])
        avg = 0.5 * (fi[mask] + fj[mask]) + EPS
        w = edge_w[mask]
        rough = float(np.sum(w * diff) / (np.sum(w * avg) + EPS))
        return rough, math.exp(-rough), int(np.sum(mask))

    active_rough, active_smooth, active_edges = calc(active_mask)
    core_rough, core_smooth, core_edges = calc(core_mask)
    return {"active_roughness": active_rough, "active_smoothness": active_smooth, "active_edges": active_edges,
            "core_roughness": core_rough, "core_smoothness": core_smooth, "core_edges": core_edges}


def compute_contact_weights(binder: List[Residue], target: List[Residue], shell_max: float, ideal_dist: float,
                            density_sigma: float, clash_cutoff: float, tangent_floor: float, front_dot: float,
                            back_dot: float, dot_sigma: float, back_penalty: float, cb_far_delta: float,
                            cb_delta_sigma: float, cb_near_cutoff: float, cb_support_cutoff: float,
                            cb_sigma: float) -> Dict[str, np.ndarray]:
    b_ca, b_cb, b_dirs = residue_arrays(binder)
    t_ca, t_cb, t_dirs = residue_arrays(target)
    D = pairwise_dist(b_ca, t_ca)
    cb_D = pairwise_dist(b_cb, t_ca)
    raw_mask = D < shell_max
    dist_w = np.exp(-((D - ideal_dist) ** 2) / (2.0 * density_sigma * density_sigma)) * raw_mask
    dist_w[D < clash_cutoff] *= 0.25

    v = t_ca[None, :, :] - b_ca[:, None, :]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_unit = v / (v_norm + EPS)
    valid_b_dir = has_valid_dir(b_dirs)
    dot = np.sum(b_dirs[:, None, :] * v_unit, axis=-1)
    dot[~valid_b_dir, :] = 0.0
    cb_delta = cb_D - D

    front_score = sigmoid((dot - front_dot) / max(dot_sigma, EPS))
    orient_w = tangent_floor + (1.0 - tangent_floor) * front_score
    back_orientation_score = sigmoid((back_dot - dot) / max(dot_sigma, EPS))
    cb_far_score = sigmoid((cb_delta - cb_far_delta) / max(cb_delta_sigma, EPS))
    back_suspicion = back_orientation_score * cb_far_score * raw_mask
    orient_w = orient_w * (1.0 - (1.0 - back_penalty) * back_suspicion)
    orient_w[~valid_b_dir, :] = 1.0

    cb_support_score = sigmoid((cb_support_cutoff - cb_D) / max(cb_sigma, EPS))
    cb_support_w = 0.65 + 0.35 * cb_support_score
    W = dist_w * orient_w * cb_support_w

    return {
        "b_ca": b_ca, "t_ca": t_ca, "b_dirs": b_dirs, "t_dirs": t_dirs,
        "D": D, "cb_D": cb_D, "cb_delta": cb_delta, "raw_mask": raw_mask,
        "dist_w": dist_w, "W": W, "dot": dot,
        "front_mask": (dot >= front_dot) & raw_mask,
        "tangential_mask": (dot > back_dot) & (dot < front_dot) & raw_mask,
        "backfacing_mask": (dot <= back_dot) & raw_mask,
        "backfacing_cb_far_mask": (dot <= back_dot) & (cb_delta > cb_far_delta) & raw_mask,
        "cb_near_mask": (cb_D < cb_near_cutoff) & raw_mask,
        "cb_supported_mask": (cb_D < cb_support_cutoff) & raw_mask,
        "cb_closer_mask": (cb_delta < 0.0) & raw_mask,
    }


def weighted_ratio(base_weight: np.ndarray, mask: np.ndarray, denom: float) -> float:
    if denom <= EPS:
        return 0.0
    return float(np.sum(base_weight[mask]) / (denom + EPS))


def empty_microdomain_result() -> Dict[str, float]:
    return {"microdomain_count": 0, "micro_plane_alignment_weighted": 0.0,
            "micro_center_distance_weighted": 0.0, "micro_planarity_mean": 0.0,
            "micro_fit_score_weighted": 0.0}


def compute_microdomain_plane_metrics(W: np.ndarray, b_ca: np.ndarray, t_ca: np.ndarray, t_field: np.ndarray,
                                      contact_threshold: float, micro_radius: float, ideal_dist: float,
                                      micro_dist_sigma: float, max_microdomains: int) -> Dict[str, float]:
    if len(t_ca) < 3 or len(b_ca) < 3:
        return empty_microdomain_result()
    t_internal_D = pairwise_dist(t_ca, t_ca)
    contacted_indices = np.where(t_field > contact_threshold)[0]
    if len(contacted_indices) == 0:
        return empty_microdomain_result()
    if len(contacted_indices) > max_microdomains:
        order = np.argsort(t_field[contacted_indices])[::-1]
        contacted_indices = contacted_indices[order[:max_microdomains]]

    aligns, dists, plans, fits, weights = [], [], [], [], []
    for j in contacted_indices:
        target_micro_idx = np.where(t_internal_D[j] < micro_radius)[0]
        if len(target_micro_idx) < 3:
            continue
        local_b_weight = np.sum(W[:, target_micro_idx], axis=1)
        binder_micro_idx = np.where(local_b_weight > contact_threshold)[0]
        if len(binder_micro_idx) < 3:
            topk = min(8, len(b_ca))
            top_idx = np.argsort(local_b_weight)[::-1][:topk]
            binder_micro_idx = top_idx[local_b_weight[top_idx] > EPS]
        if len(binder_micro_idx) < 3:
            continue
        t_coords, b_coords = t_ca[target_micro_idx], b_ca[binder_micro_idx]
        nT, pT = pca_plane_normal(t_coords)
        nB, pB = pca_plane_normal(b_coords)
        if nT is None or nB is None:
            continue
        alignment = abs(float(np.dot(nT, nB)))
        center_dist = float(np.linalg.norm(np.mean(t_coords, axis=0) - np.mean(b_coords, axis=0)))
        planarity = pT * pB
        distance_score = math.exp(-((center_dist - ideal_dist) ** 2) / (2.0 * micro_dist_sigma * micro_dist_sigma))
        local_contact_weight = float(np.sum(local_b_weight[binder_micro_idx]))
        fit_score = alignment * distance_score * planarity
        aligns.append(alignment); dists.append(center_dist); plans.append(planarity); fits.append(fit_score)
        weights.append(local_contact_weight * max(planarity, 0.05))
    if len(weights) == 0 or np.sum(weights) <= EPS:
        return empty_microdomain_result()
    aligns, dists, plans, fits, weights = map(np.array, (aligns, dists, plans, fits, weights))
    wsum = float(np.sum(weights)) + EPS
    return {"microdomain_count": int(len(weights)),
            "micro_plane_alignment_weighted": float(np.sum(aligns * weights) / wsum),
            "micro_center_distance_weighted": float(np.sum(dists * weights) / wsum),
            "micro_planarity_mean": float(np.mean(plans)),
            "micro_fit_score_weighted": float(np.sum(fits * weights) / wsum)}


def empty_continuity_result() -> Dict[str, float]:
    return {"contact_map_edges": 0, "contact_map_continuity_mean_dist": 0.0,
            "contact_map_continuity_score": 0.0, "contact_map_distortion_mean": 0.0,
            "contact_map_jump_fraction": 0.0}


def compute_contact_map_continuity(W: np.ndarray, b_ca: np.ndarray, t_ca: np.ndarray, b_field: np.ndarray,
                                   b_dirs: np.ndarray, graph_cutoff: float, contact_threshold: float,
                                   jump_cutoff: float) -> Dict[str, float]:
    mapped_target = np.zeros_like(b_ca)
    for i in range(len(b_ca)):
        if b_field[i] > EPS:
            mapped_target[i] = np.sum(W[i, :, None] * t_ca, axis=0) / (b_field[i] + EPS)
    edge_i, edge_j, edge_w = build_spatial_edges(b_ca, graph_cutoff, dirs=b_dirs, face_aware=True)
    if len(edge_i) == 0:
        return empty_continuity_result()
    active = (b_field[edge_i] > contact_threshold) & (b_field[edge_j] > contact_threshold)
    if not np.any(active):
        return empty_continuity_result()
    ei, ej, ew = edge_i[active], edge_j[active], edge_w[active]
    mapped_dist = np.linalg.norm(mapped_target[ei] - mapped_target[ej], axis=1)
    binder_dist = np.linalg.norm(b_ca[ei] - b_ca[ej], axis=1)
    mean_mapped_dist = weighted_mean(mapped_dist, ew)
    mean_distortion = weighted_mean(np.abs(mapped_dist - binder_dist), ew)
    jump_fraction = float(np.sum(ew[mapped_dist > jump_cutoff]) / (np.sum(ew) + EPS))
    return {"contact_map_edges": int(len(ei)),
            "contact_map_continuity_mean_dist": mean_mapped_dist,
            "contact_map_continuity_score": math.exp(-mean_mapped_dist / 10.0),
            "contact_map_distortion_mean": mean_distortion,
            "contact_map_jump_fraction": jump_fraction}


def region_metrics(t_field: np.ndarray, mask: np.ndarray, total_weight: float, contact_threshold: float) -> Dict[str, float]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0, "effective_coverage": 0.0, "weight_fraction": 0.0, "mean_field": 0.0}
    region_field = t_field[mask]
    return {"count": count,
            "effective_coverage": float(np.mean(region_field > contact_threshold)),
            "weight_fraction": float(np.sum(region_field) / (total_weight + EPS)),
            "mean_field": float(np.mean(region_field))}


def compute_metrics(pdb_path: Path, args: argparse.Namespace, desired_specs: List[str], undesired_specs: List[str]) -> Dict[str, float]:
    binder, target = parse_pdb_backbone(pdb_path, binder_chain=args.binder_chain)
    n_b, n_t = len(binder), len(target)

    data = compute_contact_weights(
        binder=binder, target=target, shell_max=args.shell_max, ideal_dist=args.ideal_dist,
        density_sigma=args.density_sigma, clash_cutoff=args.clash_cutoff,
        tangent_floor=args.tangent_floor, front_dot=args.front_dot, back_dot=args.back_dot,
        dot_sigma=args.dot_sigma, back_penalty=args.back_penalty, cb_far_delta=args.cb_far_delta,
        cb_delta_sigma=args.cb_delta_sigma, cb_near_cutoff=args.cb_near_cutoff,
        cb_support_cutoff=args.cb_support_cutoff, cb_sigma=args.cb_sigma)

    b_ca, t_ca, b_dirs, t_dirs = data["b_ca"], data["t_ca"], data["b_dirs"], data["t_dirs"]
    D, cb_D, cb_delta = data["D"], data["cb_D"], data["cb_delta"]
    raw_mask, dist_w, W = data["raw_mask"], data["dist_w"], data["W"]

    raw_weight_sum = float(np.sum(dist_w))
    effective_weight_sum = float(np.sum(W))
    effective_to_raw_ratio = effective_weight_sum / (raw_weight_sum + EPS)

    b_field = np.sum(W, axis=1)
    t_field = np.sum(W, axis=0)

    contacted_b = b_ca[b_field > args.contact_threshold]
    contacted_t = t_ca[t_field > args.contact_threshold]

    binder_rough = graph_field_roughness(b_ca, b_field, args.graph_cutoff, dirs=b_dirs, face_aware=True,
                                         contact_threshold=args.contact_threshold)
    target_rough = graph_field_roughness(t_ca, t_field, args.graph_cutoff, dirs=t_dirs, face_aware=False,
                                         contact_threshold=args.contact_threshold)
    continuity = compute_contact_map_continuity(W, b_ca, t_ca, b_field, b_dirs, args.graph_cutoff,
                                                args.contact_threshold, args.jump_cutoff)

    target_linearity, target_planarity, target_compactness = pca_shape_features(t_ca)
    if len(contacted_t) >= 3:
        ct_linearity, ct_planarity, ct_compactness = pca_shape_features(contacted_t)
    else:
        ct_linearity, ct_planarity, ct_compactness = 0.0, 0.0, 0.0

    micro = compute_microdomain_plane_metrics(W, b_ca, t_ca, t_field, args.contact_threshold,
                                              args.micro_radius, args.ideal_dist,
                                              args.micro_dist_sigma, args.max_microdomains)

    desired_mask = region_mask_for_target(target, desired_specs)
    undesired_mask = region_mask_for_target(target, undesired_specs)
    total_t_weight = float(np.sum(t_field))
    desired = region_metrics(t_field, desired_mask, total_t_weight, args.contact_threshold)
    undesired = region_metrics(t_field, undesired_mask, total_t_weight, args.contact_threshold)

    cb_delta_weighted_mean = float(np.sum(dist_w * cb_delta) / (raw_weight_sum + EPS)) if raw_weight_sum > EPS else 0.0
    cb_dist_weighted_mean = float(np.sum(dist_w * cb_D) / (raw_weight_sum + EPS)) if raw_weight_sum > EPS else 0.0

    row = {
        "pdb_name": pdb_path.stem, "binder_len": n_b, "target_len": n_t,
        "raw_mean_shell_count": float(np.mean(np.sum(raw_mask, axis=1))),
        "raw_mean_shell_6": float(np.mean(np.sum(D < 6.0, axis=1))),
        "raw_mean_shell_7": float(np.mean(np.sum(D < 7.0, axis=1))),
        "raw_mean_shell_8": float(np.mean(np.sum(D < 8.0, axis=1))),
        "raw_mean_shell_10": float(np.mean(np.sum(D < 10.0, axis=1))),
        "raw_mean_shell_12": float(np.mean(np.sum(D < 12.0, axis=1))),
        "raw_weight_sum": raw_weight_sum,
        "effective_weight_sum": effective_weight_sum,
        "effective_to_raw_ratio": effective_to_raw_ratio,
        "orientation_discount_ratio": max(0.0, min(1.0, 1.0 - effective_to_raw_ratio)),
        "front_facing_weight_ratio": weighted_ratio(dist_w, data["front_mask"], raw_weight_sum),
        "tangential_weight_ratio": weighted_ratio(dist_w, data["tangential_mask"], raw_weight_sum),
        "backfacing_weight_ratio": weighted_ratio(dist_w, data["backfacing_mask"], raw_weight_sum),
        "backfacing_cb_far_weight_ratio": weighted_ratio(dist_w, data["backfacing_cb_far_mask"], raw_weight_sum),
        "cb_near_weight_ratio": weighted_ratio(dist_w, data["cb_near_mask"], raw_weight_sum),
        "cb_supported_weight_ratio": weighted_ratio(dist_w, data["cb_supported_mask"], raw_weight_sum),
        "cb_closer_weight_ratio": weighted_ratio(dist_w, data["cb_closer_mask"], raw_weight_sum),
        "cb_delta_weighted_mean": cb_delta_weighted_mean,
        "cb_dist_weighted_mean": cb_dist_weighted_mean,
        "clash_pairs": int(np.sum(D < args.clash_cutoff)),
        "min_distance_global": float(np.min(D)),
        "binder_effective_hit_ratio": float(np.mean(b_field > args.contact_threshold)),
        "target_effective_coverage": float(np.mean(t_field > args.contact_threshold)),
        "binder_contact_neff_norm": effective_number(b_field) / n_b,
        "target_contact_neff_norm": effective_number(t_field) / n_t,
        "binder_contact_entropy": entropy_norm(b_field),
        "target_contact_entropy": entropy_norm(t_field),
        "binder_contact_span_norm": max_pairwise_distance(contacted_b) / (max_pairwise_distance(b_ca) + EPS),
        "target_contact_span": max_pairwise_distance(contacted_t),
        "target_contact_span_norm": max_pairwise_distance(contacted_t) / (max_pairwise_distance(t_ca) + EPS),
        "binder_field_active_roughness": binder_rough["active_roughness"],
        "binder_field_active_smoothness": binder_rough["active_smoothness"],
        "binder_field_active_edges": binder_rough["active_edges"],
        "binder_field_core_roughness": binder_rough["core_roughness"],
        "binder_field_core_smoothness": binder_rough["core_smoothness"],
        "binder_field_core_edges": binder_rough["core_edges"],
        "target_field_active_roughness": target_rough["active_roughness"],
        "target_field_active_smoothness": target_rough["active_smoothness"],
        "target_field_active_edges": target_rough["active_edges"],
        "target_field_core_roughness": target_rough["core_roughness"],
        "target_field_core_smoothness": target_rough["core_smoothness"],
        "target_field_core_edges": target_rough["core_edges"],
        **continuity,
        "target_linearity": target_linearity,
        "target_planarity": target_planarity,
        "target_compactness": target_compactness,
        "contacted_target_linearity": ct_linearity,
        "contacted_target_planarity": ct_planarity,
        "contacted_target_compactness": ct_compactness,
        **micro,
        "desired_region_count": desired["count"],
        "desired_effective_coverage": desired["effective_coverage"],
        "desired_weight_fraction": desired["weight_fraction"],
        "desired_mean_field": desired["mean_field"],
        "undesired_region_count": undesired["count"],
        "undesired_effective_coverage": undesired["effective_coverage"],
        "undesired_weight_fraction": undesired["weight_fraction"],
        "undesired_mean_field": undesired["mean_field"],
    }
    return row


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


def collect_pdbs(input_path: Optional[str], input_dir: Optional[str], max_files: Optional[int]) -> List[Path]:
    pdbs: List[Path] = []
    if input_path:
        pdbs.append(Path(input_path))
    if input_dir:
        pdbs.extend(sorted(Path(input_dir).glob("*.pdb")))
    if max_files is not None:
        pdbs = pdbs[:max_files]
    return pdbs


def main():
    parser = argparse.ArgumentParser(description="v3 contact-field descriptors for RFdiffusion binder backbones.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--binder_chain", type=str, default="A")
    parser.add_argument("--desired_regions", type=str, default="")
    parser.add_argument("--undesired_regions", type=str, default="")
    parser.add_argument("--shell_max", type=float, default=10.0)
    parser.add_argument("--ideal_dist", type=float, default=7.5)
    parser.add_argument("--density_sigma", type=float, default=2.0)
    parser.add_argument("--clash_cutoff", type=float, default=3.5)
    parser.add_argument("--graph_cutoff", type=float, default=10.0)
    parser.add_argument("--contact_threshold", type=float, default=0.1)
    parser.add_argument("--tangent_floor", type=float, default=0.45)
    parser.add_argument("--front_dot", type=float, default=0.25)
    parser.add_argument("--back_dot", type=float, default=-0.25)
    parser.add_argument("--dot_sigma", type=float, default=0.20)
    parser.add_argument("--back_penalty", type=float, default=0.25)
    parser.add_argument("--cb_far_delta", type=float, default=0.8)
    parser.add_argument("--cb_delta_sigma", type=float, default=0.5)
    parser.add_argument("--cb_near_cutoff", type=float, default=10.0)
    parser.add_argument("--cb_support_cutoff", type=float, default=12.0)
    parser.add_argument("--cb_sigma", type=float, default=1.5)
    parser.add_argument("--micro_radius", type=float, default=8.0)
    parser.add_argument("--micro_dist_sigma", type=float, default=3.0)
    parser.add_argument("--max_microdomains", type=int, default=80)
    parser.add_argument("--jump_cutoff", type=float, default=15.0)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    desired_specs = parse_region_specs(args.desired_regions)
    undesired_specs = parse_region_specs(args.undesired_regions)
    pdbs = collect_pdbs(args.input, args.input_dir, args.max_files)
    if not pdbs:
        raise SystemExit("No PDB files found.")

    rows = []
    for idx, pdb in enumerate(pdbs, start=1):
        try:
            row = compute_metrics(pdb, args, desired_specs, undesired_specs)
            row["error"] = ""
        except Exception as e:
            row = {"pdb_name": pdb.stem, "error": str(e)}
        rows.append(row)
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(pdbs)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
