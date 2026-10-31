"""Bader charge analysis: BaderKit (preferred) or pymatgen fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.provenance import ProvenanceRecord
from spm_atombridge.core.scalarfield import ScalarField
from spm_atombridge.core.structure import Structure


class BaderAdapter(Adapter):
    """Wraps BaderKit or pymatgen to compute atomic Bader charges.

    Requires a charge-density file (CHGCAR, AECCAR0+AECCAR2, or a cube file).
    Fails with a clear diagnostic if only a bare trajectory is supplied.
    """

    name = "bader"
    requires = ("pymatgen",)  # always present; BaderKit is optional
    fallback = "pymatgen BaderAnalysis (requires external 'bader' executable)"
    description = "Bader charge transfer from DFT charge density"

    _DENSITY_FILES = ("CHGCAR", "AECCAR0", "AECCAR2")

    def __init__(self):
        self._has_baderkit = self._check_baderkit()

    @staticmethod
    def _check_baderkit() -> bool:
        try:
            import baderkit  # noqa: F401
            return True
        except ImportError:
            return False

    def check_inputs(self, directory: str | Path) -> dict:
        """Pre-flight check. Returns status and diagnostic message."""
        d = Path(directory)
        found = {f: (d / f).exists() for f in self._DENSITY_FILES}
        has_density = any(found.values())
        has_poscar = (d / "POSCAR").exists() or any(d.glob("POSCAR*"))
        return {
            "directory": str(d),
            "has_density": has_density,
            "has_poscar": has_poscar,
            "files_found": found,
            "baderkit_available": self._has_baderkit,
            "ready": has_density and has_poscar,
            "diagnostic": (
                "Ready" if (has_density and has_poscar)
                else ("Missing charge density (CHGCAR or AECCAR0/2). "
                      "Run a DFT single-point calculation first.")
                if not has_density
                else "Missing POSCAR."
            ),
        }

    def compute(self, directory: str | Path) -> dict:
        """Compute Bader charges. Returns dict with charges and a ScalarField."""
        d = Path(directory)
        status = self.check_inputs(d)
        if not status["ready"]:
            raise FileNotFoundError(status["diagnostic"])

        if self._has_baderkit:
            charges, symbols = self._run_baderkit(d)
            engine = "baderkit"
        else:
            charges, symbols = self._run_pymatgen(d)
            engine = "pymatgen"

        rec = ProvenanceRecord.from_file(
            action="bader_analysis",
            adapter=self.name,
            path=d / ("CHGCAR" if (d / "CHGCAR").exists() else "AECCAR2"),
            parameters={"engine": engine, "n_atoms": len(charges)},
        )
        return {
            "charges": np.asarray(charges),
            "symbols": symbols,
            "engine": engine,
            "provenance": rec,
            "scalarfield": ScalarField(
                data=np.asarray(charges),
                units="e",
                axis="bader_charge",
                provenance=[rec],
            ),
        }

    def _run_baderkit(self, d: Path):
        from baderkit.core import Bader
        from pymatgen.io.vasp import Poscar
        from pymatgen.io.vasp.outputs import Chgcar

        struct = Poscar.from_file(str(d / "POSCAR")).structure
        chg = Chgcar.from_file(str(d / "CHGCAR"))
        b = Bader.from_charge_density(struct, chg.data["total"] / chg.structure.volume)
        b.compute()
        charges = np.array([b.get_charge(i) for i in range(len(struct))])
        symbols = [s.species_string for s in struct]
        return charges, symbols

    def _run_pymatgen(self, d: Path):
        from pymatgen.command_line.battery import BaderAnalysis
        analysis = BaderAnalysis.from_path(str(d))
        struct = analysis.structure
        charges = np.array(analysis.charges)
        symbols = [s.species_string for s in struct]
        return charges, symbols

    def run(self, directory: str | Path) -> dict:  # pragma: no cover
        return self.compute(directory)