"""Plotting adapter: publication-quality figures from data-model objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np

from spm_atombridge.adapters.base import Adapter
from spm_atombridge.core.image import Image
from spm_atombridge.core.scalarfield import ScalarField
from spm_atombridge.core.spectrum import Spectrum


class PlottingAdapter(Adapter):
    name = "matplotlib"
    requires = ("matplotlib",)
    description = "Publication-quality plots of Image, ScalarField, Spectrum"

    # Sensible defaults; user can override per-call
    DEFAULTS = {
        "dpi": 300,
        "fontsize": 9,
        "figsize": (3.4, 2.8),   # single column (Nature/ACS)
        "cmap_img": "afmhot",
        "cmap_scalar": "coolwarm",
    }

    def plot_image(self, img: Image, output: str | Path,
                   cmap: str | None = None, title: str = "") -> Path:
        fig, ax = plt.subplots(figsize=self.DEFAULTS["figsize"],
                               dpi=self.DEFAULTS["dpi"])
        im = ax.imshow(img.data, cmap=cmap or self.DEFAULTS["cmap_img"],
                       origin="lower")
        ax.set_title(title or f"{img.kind} ({img.units})",
                     fontsize=self.DEFAULTS["fontsize"])
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=self.DEFAULTS["dpi"])
        plt.close(fig)
        return out

    def plot_scalarfield(self, sf: ScalarField, output: str | Path,
                         title: str = "") -> Path:
        fig, ax = plt.subplots(figsize=self.DEFAULTS["figsize"],
                               dpi=self.DEFAULTS["dpi"])
        if sf.data.ndim == 3:
            data = sf.plane("z", -1)
        else:
            data = sf.data
        im = ax.imshow(data, cmap=self.DEFAULTS["cmap_scalar"],
                       origin="lower")
        ax.set_title(title or f"{sf.axis} ({sf.units})",
                     fontsize=self.DEFAULTS["fontsize"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=self.DEFAULTS["dpi"])
        plt.close(fig)
        return out

    def plot_spectrum(self, sp: Spectrum, output: str | Path,
                      title: str = "") -> Path:
        fig, ax = plt.subplots(figsize=self.DEFAULTS["figsize"],
                               dpi=self.DEFAULTS["dpi"])
        ax.plot(sp.x, sp.y, lw=1.2, color="k")
        ax.axhline(0, lw=0.5, color="grey")
        ax.axvline(0, lw=0.5, color="grey")
        ax.set_xlabel(sp.xlabel, fontsize=self.DEFAULTS["fontsize"])
        ax.set_ylabel(sp.ylabel, fontsize=self.DEFAULTS["fontsize"])
        ax.set_title(title or f"{sp.origin} spectrum",
                     fontsize=self.DEFAULTS["fontsize"])
        out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=self.DEFAULTS["dpi"])
        plt.close(fig)
        return out

    def run(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use explicit plot_* methods")