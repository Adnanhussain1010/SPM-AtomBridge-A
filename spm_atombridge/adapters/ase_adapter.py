"""ASE adapter: structure construction and defect creation."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111, add_adsorbate
from ase.io import read, write

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.provenance import ProvenanceRecord
from spm_atombridge.core.structure import Structure


class ASEAdapter(Adapter):
    name = "ase"
    requires = ("ase",)
    description = "Structure building, defect creation, file I/O"

    def build_fcc111(
        self,
        symbol: str = "Au",
        size: tuple[int, int, int] = (4, 4, 3),
        vacuum: float = 10.0,
    ) -> Structure:
        atoms = fcc111(symbol, size=size, vacuum=vacuum)
        rec = ProvenanceRecord(
            action="build_fcc111",
            adapter=self.name,
            parameters={"symbol": symbol, "size": list(size), "vacuum": vacuum},
        )
        return Structure(atoms=atoms, provenance=[rec])

    def make_vacancy(self, structure: Structure, index: int) -> Structure:
        atoms = structure.atoms.copy()
        del atoms[index]
        rec = ProvenanceRecord(
            action="make_vacancy",
            adapter=self.name,
            parameters={"removed_index": index},
        )
        return Structure(atoms=atoms, provenance=structure.provenance + [rec])

    def add_adsorbate(
        self,
        structure: Structure,
        symbol: str,
        position: tuple[float, float, float] = (0.0, 0.0, 2.0),
    ) -> Structure:
        atoms = structure.atoms.copy()
        atoms.append(Atoms(symbol, positions=[position]))
        rec = ProvenanceRecord(
            action="add_adsorbate",
            adapter=self.name,
            parameters={"symbol": symbol, "position": list(position)},
        )
        return Structure(atoms=atoms, provenance=structure.provenance + [rec])

    def run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use explicit methods of ASEAdapter")