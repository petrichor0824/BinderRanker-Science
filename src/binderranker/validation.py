"""Target-pool identity checks tolerant of residue renumbering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from binderranker.exceptions import TargetPoolValidationError
from binderranker.io.pdb import ParsedPDB, Residue


@dataclass
class TargetComparison:
    candidate: str
    sequence_match: bool
    residue_ids_match: bool
    ca_rmsd: float | None
    max_ca_deviation: float | None
    accepted: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chains(residues: list[Residue]) -> dict[str, list[Residue]]:
    result: dict[str, list[Residue]] = {}
    for residue in residues:
        result.setdefault(residue.chain, []).append(residue)
    return result


def target_sequence_signature(residues: list[Residue]) -> str:
    """Hash target chain/residue-name sequences, ignoring residue IDs and chain names."""

    sequences = sorted(tuple(residue.resname for residue in chain) for chain in _chains(residues).values())
    encoded = json.dumps(sequences, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _match_chains(reference: list[Residue], candidate: list[Residue]) -> list[tuple[list[Residue], list[Residue]]]:
    ref_chains = _chains(reference)
    cand_chains = _chains(candidate)
    pairs: list[tuple[list[Residue], list[Residue]]] = []
    used: set[str] = set()

    for ref_id in sorted(ref_chains):
        ref_chain = ref_chains[ref_id]
        ref_sequence = tuple(residue.resname for residue in ref_chain)
        if ref_id in cand_chains:
            same_id_sequence = tuple(residue.resname for residue in cand_chains[ref_id])
            if same_id_sequence == ref_sequence:
                pairs.append((ref_chain, cand_chains[ref_id]))
                used.add(ref_id)
                continue
        matches = [
            chain_id
            for chain_id, chain in cand_chains.items()
            if chain_id not in used
            and tuple(residue.resname for residue in chain) == ref_sequence
        ]
        if len(matches) != 1:
            raise TargetPoolValidationError(
                "target chain mapping is missing or ambiguous after residue-ID-independent sequence matching"
            )
        match = matches[0]
        pairs.append((ref_chain, cand_chains[match]))
        used.add(match)
    if len(used) != len(cand_chains):
        raise TargetPoolValidationError("candidate target contains unmatched extra chains")
    return pairs


def _kabsch(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref_centered = reference - np.mean(reference, axis=0, keepdims=True)
    cand_centered = candidate - np.mean(candidate, axis=0, keepdims=True)
    covariance = cand_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    aligned = (rotation @ cand_centered.T).T
    deviations = np.linalg.norm(aligned - ref_centered, axis=1)
    return float(np.sqrt(np.mean(deviations * deviations))), float(np.max(deviations))


def compare_targets(
    reference: ParsedPDB,
    candidate: ParsedPDB,
    *,
    rmsd_tolerance: float,
) -> TargetComparison:
    """Compare target identity without requiring residue IDs or raw coordinates to match."""

    if target_sequence_signature(reference.target) != target_sequence_signature(candidate.target):
        return TargetComparison(
            candidate=candidate.path.name,
            sequence_match=False,
            residue_ids_match=False,
            ca_rmsd=None,
            max_ca_deviation=None,
            accepted=False,
            reason="target chain/residue-name sequence differs",
        )
    try:
        pairs = _match_chains(reference.target, candidate.target)
    except TargetPoolValidationError as exc:
        return TargetComparison(
            candidate=candidate.path.name,
            sequence_match=True,
            residue_ids_match=False,
            ca_rmsd=None,
            max_ca_deviation=None,
            accepted=False,
            reason=str(exc),
        )

    reference_residues = [residue for ref_chain, _ in pairs for residue in ref_chain]
    candidate_residues = [residue for _, cand_chain in pairs for residue in cand_chain]
    residue_ids_match = all(
        (ref.chain, ref.resid, ref.icode) == (cand.chain, cand.resid, cand.icode)
        for ref, cand in zip(reference_residues, candidate_residues)
    )
    reference_ca = np.array([residue.ca for residue in reference_residues], dtype=float)
    candidate_ca = np.array([residue.ca for residue in candidate_residues], dtype=float)
    rmsd, max_deviation = _kabsch(reference_ca, candidate_ca)
    accepted = rmsd <= rmsd_tolerance
    return TargetComparison(
        candidate=candidate.path.name,
        sequence_match=True,
        residue_ids_match=residue_ids_match,
        ca_rmsd=rmsd,
        max_ca_deviation=max_deviation,
        accepted=accepted,
        reason="" if accepted else f"target CA RMSD {rmsd:.6g} exceeds tolerance {rmsd_tolerance:.6g}",
    )


def validate_target_pool(
    structures: list[ParsedPDB], *, rmsd_tolerance: float = 1.0
) -> dict[str, Any]:
    """Validate one target-specific pool and return an auditable summary."""

    if not structures:
        raise TargetPoolValidationError("target pool is empty")
    if rmsd_tolerance < 0:
        raise TargetPoolValidationError("target RMSD tolerance must be non-negative")
    reference = structures[0]
    comparisons = [
        TargetComparison(
            candidate=reference.path.name,
            sequence_match=True,
            residue_ids_match=True,
            ca_rmsd=0.0,
            max_ca_deviation=0.0,
            accepted=True,
        )
    ]
    comparisons.extend(
        compare_targets(reference, candidate, rmsd_tolerance=rmsd_tolerance)
        for candidate in structures[1:]
    )
    rejected = [comparison for comparison in comparisons if not comparison.accepted]
    summary = {
        "reference": reference.path.name,
        "target_sequence_signature": target_sequence_signature(reference.target),
        "rmsd_tolerance": rmsd_tolerance,
        "candidate_count": len(structures),
        "accepted_count": len(structures) - len(rejected),
        "rejected_count": len(rejected),
        "max_ca_rmsd": max(
            comparison.ca_rmsd or 0.0 for comparison in comparisons
        ),
        "residue_id_mismatch_count": sum(
            not comparison.residue_ids_match for comparison in comparisons
        ),
        "comparisons": [comparison.as_dict() for comparison in comparisons],
    }
    if rejected:
        details = "; ".join(
            f"{comparison.candidate}: {comparison.reason}" for comparison in rejected
        )
        raise TargetPoolValidationError(f"mixed/incompatible target pool: {details}")
    return summary

