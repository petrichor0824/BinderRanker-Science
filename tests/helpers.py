from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def atom_line(
    serial: int,
    atom: str,
    resname: str,
    chain: str,
    resid: int,
    xyz: Iterable[float],
    *,
    altloc: str = "",
) -> str:
    x, y, z = xyz
    element = atom.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s}{altloc:1s}{resname:>3s} {chain:1s}"
        f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


def write_complex(
    path: Path,
    *,
    target_offset: int = 0,
    target_rotation: np.ndarray | None = None,
    target_translation: np.ndarray | None = None,
    target_noise: float = 0.0,
    changed_target_resname: bool = False,
    altloc_first_atom: bool = False,
) -> Path:
    binder_names = ["ALA", "GLY", "SER", "LEU", "THR"]
    target_names = ["VAL", "ASN", "TYR", "ASP", "LYS", "PHE"]
    if changed_target_resname:
        target_names[2] = "TRP"
    rotation = np.eye(3) if target_rotation is None else target_rotation
    translation = np.zeros(3) if target_translation is None else target_translation
    serial = 1
    lines: list[str] = []

    def append_residue(chain: str, resid: int, resname: str, ca: np.ndarray) -> None:
        nonlocal serial
        atom_positions = {
            "N": ca + np.array([-0.55, 0.10, 0.00]),
            "CA": ca,
            "C": ca + np.array([0.55, -0.10, 0.05]),
            "O": ca + np.array([0.85, -0.25, 0.05]),
            "CB": ca + np.array([0.00, 1.20, 0.45]),
        }
        for atom, xyz in atom_positions.items():
            altloc = "A" if altloc_first_atom and serial == 1 else ""
            lines.append(atom_line(serial, atom, resname, chain, resid, xyz, altloc=altloc))
            serial += 1

    for index, resname in enumerate(binder_names, 1):
        ca = np.array([2.0 * (index - 1), 0.25 * ((index % 2) * 2 - 1), 0.0])
        append_residue("A", index, resname, ca)

    for index, resname in enumerate(target_names, 1):
        base = np.array([1.5 * (index - 1), 0.6 * ((index % 3) - 1), 7.4])
        transformed = rotation @ base + translation
        if target_noise:
            transformed = transformed + np.array(
                [target_noise * ((index % 2) * 2 - 1), 0.0, target_noise * 0.5]
            )
        append_residue("B", index + target_offset, resname, transformed)

    lines.append("END\n")
    path.write_text("".join(lines), encoding="ascii", newline="")
    return path

