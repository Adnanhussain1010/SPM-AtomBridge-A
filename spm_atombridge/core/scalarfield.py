"""Scalar fields: charge density, work function, CPD maps."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spm_atombridge.core.provenance import ProvenanceRecord


@dataclass
class ScalarField:
    """A scalar field sampled on a 3D grid or a 2D surface map."""

    data: np.ndarray
    units: str = "dimensionless"
    axis: str = ""  # e.g. "charge_density", "work_function", "cpd"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    def __post_init__(self):
        self.data = np.asarray(self.data)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def stats(self) -> dict:
        d = self.data
        return {
            "shape": list(d.shape),
            "min": float(np.nanmin(d)),
            "max": float(np.nanmax(d)),
            "mean": float(np.nanmean(d)),
            "units": self.units,
            "axis": self.axis,
        }

    def plane(self, axis: str = "z", index: int = -1) -> np.ndarray:
        """Extract a 2D slice along a given axis."""
        ax = {"x": 0, "y": 1, "z": 2}[axis]
        return np.take(self.data, index, axis=ax)