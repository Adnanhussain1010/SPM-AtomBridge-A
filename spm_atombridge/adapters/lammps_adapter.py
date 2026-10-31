"""LAMMPS dump adapter: read trajectories and per-atom fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.provenance import ProvenanceRecord
from spm_atombridge.core.scalarfield import ScalarField
from spm_atombridge.core.structure import Structure


class LAMMPSAdapter(Adapter):
    """Reads LAMMPS dump files into Structure + per-atom ScalarFields."""

    name = "lammps"
    requires = ("ase",)
    description = "LAMMPS dump reader with per-atom field extraction"

    # Fields we know how to extract as ScalarFields
    KNOWN_FIELDS = ("c_pe", "pe", "c_ke", "fx", "fy", "fz")

    def __init__(self):
        self._last_header = None

    def _parse_header(self, path: Path) -> dict:
        """Read only the first frame header - fast, no full file load."""
        info = {"n_atoms": 0, "box": None, "columns": [], "n_frames": 0}
        with open(path) as f:
            for line in f:
                if line.startswith("ITEM: TIMESTEP"):
                    info["n_frames"] += 1
                    if info["n_frames"] > 1:
                        break
                elif line.startswith("ITEM: NUMBER OF ATOMS"):
                    info["n_atoms"] = int(f.readline().strip())
                elif line.startswith("ITEM: BOX BOUNDS"):
                    bounds = []
                    for _ in range(3):
                        parts = f.readline().split()
                        bounds.append((float(parts[0]), float(parts[1])))
                    info["box"] = bounds
                elif line.startswith("ITEM: ATOMS"):
                    info["columns"] = line.strip().split()[2:]
                    break
        return info

    def load(self, path: str | Path, frame: int = -1) -> Structure:
        """Load one frame as a Structure. Per-atom fields stored in info dict."""
        from ase.io import read as ase_read

        p = Path(path)
        header = self._parse_header(p)
        self._last_header = header

        atoms = ase_read(str(p), format="lammps-dump-text", index=frame)
        if isinstance(atoms, list):
            atoms = atoms[-1]

        rec = ProvenanceRecord.from_file(
            action="load_lammps_dump",
            adapter=self.name,
            path=p,
            parameters={"frame": frame, "n_atoms": len(atoms)},
        )
        struct = Structure(atoms=atoms, provenance=[rec])
        # Stash the parsed fields for later extraction
        struct._lammps_fields = self._read_fields(p, frame, header)
        return struct

    def _read_fields(self, path: Path, frame: int, header: dict) -> dict:
        """Extract all numeric per-atom columns as numpy arrays."""
        cols = header["columns"]
        numeric_idx = [i for i, c in enumerate(cols) if c not in ("element", "type")]
        target = frame if frame >= 0 else header["n_frames"] + frame

        # State machine: find target frame's atom block
        data_rows = []
        state = "seeking_timestep"
        frame_count = 0
        in_target_frame = False
        reading_atoms = False

        with open(path) as f:
            for line in f:
                if line.startswith("ITEM: TIMESTEP"):
                    frame_count += 1
                    in_target_frame = (frame_count - 1) == target
                    reading_atoms = False
                    continue
                if not in_target_frame:
                    continue
                if line.startswith("ITEM: ATOMS"):
                    reading_atoms = True
                    continue
                if not reading_atoms:
                    continue
                if line.startswith("ITEM:"):
                    break
                parts = line.split()
                if len(parts) != len(cols):
                    continue
                data_rows.append(parts)
                if len(data_rows) == header["n_atoms"]:
                    break

        if not data_rows:
            return {}
        arr = np.array(data_rows, dtype=float)
        return {cols[i]: arr[:, i] for i in numeric_idx}

    def extract_field(self, struct: Structure, name: str) -> ScalarField:
        """Return any per-atom field as a ScalarField."""
        fields = getattr(struct, "_lammps_fields", {})
        if name not in fields:
            raise KeyError(
                f"Field '{name}' not in {list(fields)}. "
                f"Available: {self.available_fields(struct)}"
            )
        rec = ProvenanceRecord(
            action="extract_lammps_field",
            adapter=self.name,
            parameters={"field": name},
        )
        return ScalarField(
            data=fields[name],
            units={"c_pe": "eV", "pe": "eV"}.get(name, "unknown"),
            axis=name,
            provenance=[rec],
        )

    def available_fields(self, struct: Structure) -> list:
        return list(getattr(struct, "_lammps_fields", {}).keys())

    def surface_height_map(
        self, struct: Structure, grid: int = 128, layer_thickness: float = 2.0
    ) -> np.ndarray:
        """Interpolate the topmost atomic layer z-coordinate onto a grid.

        Identifies the top atomic layer (within `layer_thickness` Angstrom of
        the maximum z), then uses scipy.interpolate.griddata to fill a dense
        grid. Returns a smooth 2D array with NaN outside the convex hull.
        """
        from scipy.interpolate import griddata

        pos = struct.atoms.get_positions()
        z = pos[:, 2]
        zmax = z.max()
        mask = z >= (zmax - layer_thickness)
        top = pos[mask]

        cell = struct.atoms.cell
        xlen, ylen = cell[0][0], cell[1][1]

        gx = np.linspace(0, xlen, grid)
        gy = np.linspace(0, ylen, grid)
        GX, GY = np.meshgrid(gx, gy)
        H = griddata(
            top[:, :2], top[:, 2],
            (GX, GY), method="linear", fill_value=np.nan,
        )
        return H

    def binned_field_map(
        self, struct: Structure, field: str, grid: int = 128,
        layer_thickness: float = 2.0,
    ) -> ScalarField:
        """Interpolate any per-atom field onto a dense grid (top layer only)."""
        from scipy.interpolate import griddata

        fields = getattr(struct, "_lammps_fields", {})
        if field not in fields:
            raise KeyError(f"Field '{field}' not available. "
                           f"Got: {list(fields)}")
        values = fields[field]
        pos = struct.atoms.get_positions()
        z = pos[:, 2]
        zmax = z.max()
        mask = z >= (zmax - layer_thickness)
        top_pos = pos[mask]
        top_vals = values[mask]

        cell = struct.atoms.cell
        xlen, ylen = cell[0][0], cell[1][1]
        gx = np.linspace(0, xlen, grid)
        gy = np.linspace(0, ylen, grid)
        GX, GY = np.meshgrid(gx, gy)
        M = griddata(
            top_pos[:, :2], top_vals,
            (GX, GY), method="linear", fill_value=np.nan,
        )
        return ScalarField(
            data=M,
            units={"c_pe": "eV", "pe": "eV"}.get(field, "unknown"),
            axis=field,
            provenance=list(struct.provenance),
        )

    def run(self, *args, **kwargs):  # pragma: no cover
        return self.load(*args, **kwargs)
