"""Spectroscopy: I–V curves, dI/dV, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spm_atombridge.core.provenance import ProvenanceRecord


@dataclass
class Spectrum:
    """A 1D spectrum (bias vs current, energy vs DOS, ...)."""

    x: np.ndarray
    y: np.ndarray
    xlabel: str = "bias (V)"
    ylabel: str = "current (A)"
    origin: str = "experiment"  # experiment, simulation
    metadata: dict = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same shape")

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        xcol: int = 0,
        ycol: int = 1,
        skiprows: int = 0,
        **kwargs,
    ) -> "Spectrum":
        p = Path(path)
        arr = np.loadtxt(str(p), skiprows=skiprows)
        rec = ProvenanceRecord.from_file("load_spectrum", "numpy", p, kwargs)
        return cls(
            x=arr[:, xcol], y=arr[:, ycol], provenance=[rec], **kwargs
        )

    def zero_bias_conductance(self, window: float = 0.05) -> float:
        """Linear slope of y(x) near x=0 over +/- window."""
        mask = np.abs(self.x) <= window
        if mask.sum() < 2:
            return float("nan")
        return float(np.polyfit(self.x[mask], self.y[mask], 1)[0])

    def stats(self) -> dict:
        return {
            "n_points": int(self.x.size),
            "x_range": [float(self.x.min()), float(self.x.max())],
            "y_range": [float(self.y.min()), float(self.y.max())],
            "zero_bias_conductance": self.zero_bias_conductance(),
            "origin": self.origin,
        }