"""SPM images: experimental or simulated."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spm_atombridge.core.provenance import ProvenanceRecord


@dataclass
class Image:
    """A 2D SPM image with metadata and provenance."""

    data: np.ndarray
    kind: str = "topography"  # topography, kpfm, current, simulated_afm, simulated_cpd
    units: str = "m"
    scan_size: tuple[float, float] | None = None  # in metres
    metadata: dict = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)

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
            "kind": self.kind,
        }

    def save_npy(self, path: str | Path) -> None:
        np.save(str(path), self.data)

    @classmethod
    def from_npy(cls, path: str | Path, **kwargs) -> "Image":
        p = Path(path)
        arr = np.load(str(p))
        rec = ProvenanceRecord.from_file("load_image", "numpy", p, kwargs)
        return cls(data=arr, provenance=[rec], **kwargs)