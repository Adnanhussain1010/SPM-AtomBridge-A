"""Structure wrapper around ase.Atoms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from spm_atombridge.core.provenance import ProvenanceRecord


@dataclass
class Structure:
    """A single atomic configuration plus its provenance."""

    atoms: Atoms
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path, index: int = -1) -> "Structure":
        """Load POSCAR / CIF / XYZ / LAMMPS dump / any ASE-readable format."""
        p = Path(path)
        atoms = ase_read(str(p), index=index)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        rec = ProvenanceRecord.from_file("load_structure", "ase", p)
        return cls(atoms=atoms, provenance=[rec])

    @classmethod
    def from_atoms(cls, atoms: Atoms) -> "Structure":
        return cls(atoms=atoms)

    def summary(self) -> dict:
        return {
            "formula": self.atoms.get_chemical_formula(),
            "n_atoms": len(self.atoms),
            "pbc": self.atoms.pbc.tolist(),
            "cell": np.asarray(self.atoms.cell).tolist(),
        }

    def copy(self) -> "Structure":
        return Structure(atoms=self.atoms.copy(), provenance=list(self.provenance))

    def save(self, path: str | Path) -> None:
        from ase.io import write as ase_write

        ase_write(str(path), self.atoms)