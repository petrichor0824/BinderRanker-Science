"""Legacy-compatible PDB parsing plus independent strict validation.

The compatibility parser deliberately preserves contact_field_v3 semantics. Strict
validation inspects ambiguity and completeness but never repairs or rewrites input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from binderranker.exceptions import PDBValidationError


class Residue:
    """Minimal residue representation matching the historical extractor."""

    def __init__(self, chain: str, resid: int, resname: str, icode: str = ""):
        self.chain = chain if chain else "_"
        self.resid = resid
        self.resname = resname
        self.icode = icode.strip()
        self.atoms: dict[str, np.ndarray] = {}

    def has(self, atom: str) -> bool:
        return atom in self.atoms

    @property
    def ca(self) -> np.ndarray:
        return self.atoms["CA"]

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.resid}{self.icode}"


@dataclass
class PDBDiagnostics:
    path: str
    atom_records: int = 0
    retained_atom_records: int = 0
    model_records: int = 0
    altloc_records: int = 0
    duplicate_atom_records: int = 0
    malformed_atom_records: int = 0
    blank_chain_records: int = 0
    incomplete_backbone_residues: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def strict_valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strict_valid"] = self.strict_valid
        return result


@dataclass
class ParsedPDB:
    path: Path
    binder: list[Residue]
    target: list[Residue]
    diagnostics: PDBDiagnostics


def parse_pdb_backbone(
    pdb_path: Path, binder_chain: str = "A"
) -> tuple[list[Residue], list[Residue]]:
    """Parse with byte-level scientific semantics matching contact_field_v3."""

    residues: dict[tuple[str, int, str, str], Residue] = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
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

    binder: list[Residue] = []
    target: list[Residue] = []
    for residue in residues.values():
        if not residue.has("CA"):
            continue
        if residue.chain == binder_chain:
            binder.append(residue)
        else:
            target.append(residue)

    binder.sort(key=lambda residue: (residue.chain, residue.resid, residue.icode))
    target.sort(key=lambda residue: (residue.chain, residue.resid, residue.icode))

    if not binder:
        raise ValueError(f"No binder residues found for binder_chain={binder_chain}")
    if not target:
        raise ValueError("No target residues found. Target = all non-binder chains.")
    return binder, target


def inspect_pdb_records(pdb_path: Path) -> PDBDiagnostics:
    """Inspect ambiguity without changing compatibility-parser behavior."""

    diagnostics = PDBDiagnostics(path=str(pdb_path))
    seen_atoms: set[tuple[str, int, str, str, str]] = set()
    residue_atoms: dict[tuple[str, int, str, str], set[str]] = {}

    with open(pdb_path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith("MODEL"):
                diagnostics.model_records += 1
            if not line.startswith("ATOM"):
                continue
            diagnostics.atom_records += 1
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C", "O", "CB"}:
                continue
            diagnostics.retained_atom_records += 1
            altloc = line[16].strip() if len(line) > 16 else ""
            if altloc:
                diagnostics.altloc_records += 1
            chain_raw = line[21].strip() if len(line) > 21 else ""
            if not chain_raw:
                diagnostics.blank_chain_records += 1
            chain = chain_raw or "_"
            resname = line[17:20].strip()
            icode = line[26].strip() if len(line) > 26 else ""
            try:
                resid = int(line[22:26])
                float(line[30:38])
                float(line[38:46])
                float(line[46:54])
            except (ValueError, IndexError):
                diagnostics.malformed_atom_records += 1
                diagnostics.issues.append(f"line {line_number}: malformed retained ATOM record")
                continue
            residue_key = (chain, resid, icode, resname)
            atom_key = (*residue_key, atom)
            if atom_key in seen_atoms:
                diagnostics.duplicate_atom_records += 1
            seen_atoms.add(atom_key)
            residue_atoms.setdefault(residue_key, set()).add(atom)

    if diagnostics.model_records > 1:
        diagnostics.issues.append(
            f"multiple MODEL records ({diagnostics.model_records}) are ambiguous to the legacy parser"
        )
    if diagnostics.altloc_records:
        diagnostics.issues.append(
            f"alternate-location ATOM records present ({diagnostics.altloc_records})"
        )
    if diagnostics.duplicate_atom_records:
        diagnostics.issues.append(
            f"duplicate retained atom records present ({diagnostics.duplicate_atom_records})"
        )
    if diagnostics.blank_chain_records:
        diagnostics.issues.append(
            f"blank-chain retained atom records present ({diagnostics.blank_chain_records})"
        )

    incomplete = 0
    for atoms in residue_atoms.values():
        if "CA" in atoms and not {"N", "CA", "C"}.issubset(atoms):
            incomplete += 1
    diagnostics.incomplete_backbone_residues = incomplete
    if incomplete:
        diagnostics.issues.append(
            f"CA-containing residues missing N and/or C ({incomplete})"
        )
    if diagnostics.retained_atom_records == 0:
        diagnostics.issues.append("no retained ATOM records")
    return diagnostics


def parse_structure(
    pdb_path: Path, binder_chain: str = "A", strict: bool = False
) -> ParsedPDB:
    """Inspect and parse a structure; strict mode rejects but never repairs."""

    path = Path(pdb_path)
    diagnostics = inspect_pdb_records(path)
    if strict and diagnostics.issues:
        raise PDBValidationError(f"{path}: " + "; ".join(diagnostics.issues))
    try:
        binder, target = parse_pdb_backbone(path, binder_chain=binder_chain)
    except ValueError as exc:
        if strict:
            raise PDBValidationError(f"{path}: {exc}") from exc
        raise
    return ParsedPDB(path=path, binder=binder, target=target, diagnostics=diagnostics)

