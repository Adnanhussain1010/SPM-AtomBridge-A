"""Publication-quality graphing from raw DFT/LAMMPS output."""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from spm_atombridge.adapters.base import Adapter


class GraphAdapter(Adapter):
    """Generate publication-ready graphs from standard DFT/LAMMPS files."""

    name = "grapher"
    requires = ("matplotlib", "numpy")
    description = "Plot DOS, band structures, and LAMMPS thermodynamics"

    def plot_lammps_thermo(self, log_path: str | Path, output_png: str | Path) -> Path:
        """Parse a LAMMPS log file and plot Temperature and Total Energy."""
        p = Path(log_path)
        if not p.exists():
            raise FileNotFoundError(p)

        temps, energies, steps = [], [], []
        pattern = re.compile(r"^\s*(\d+)\s+([0-9.\-eE+]+)\s+([0-9.\-eE+]+)")

        in_thermo = False
        with open(p) as f:
            for line in f:
                if "Step" in line and "Temp" in line and "PotEng" in line:
                    in_thermo = True
                    continue
                if in_thermo:
                    match = pattern.match(line)
                    if match:
                        steps.append(int(match.group(1)))
                        temps.append(float(match.group(2)))
                        energies.append(float(match.group(3)))
                    elif line.strip() == "":
                        in_thermo = False

        if not steps:
            raise ValueError("No thermo data found in LAMMPS log.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=300)

        ax1.plot(steps, temps, color="tab:red", lw=1.5)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Temperature (K)")
        ax1.set_title("LAMMPS: Temperature")

        ax2.plot(steps, energies, color="tab:blue", lw=1.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Potential Energy (eV)")
        ax2.set_title("LAMMPS: Energy")

        fig.tight_layout()
        out = Path(output_png)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def plot_dos(self, dos_path: str | Path, output_png: str | Path,
                 fermi_energy: float = 0.0) -> Path:
        """Parse a simple 2-column DOS file (Energy, DOS) and plot it."""
        data = np.loadtxt(dos_path)
        if data.shape[1] < 2:
            raise ValueError("DOS file must have at least 2 columns.")

        energy = data[:, 0]
        dos = data[:, 1]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
        ax.plot(energy, dos, color="k", lw=1.2)
        ax.axvline(fermi_energy, color="r", ls="--", lw=1, label="E_F")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS (states/eV)")
        ax.set_title("Density of States")
        ax.legend(frameon=False)

        fig.tight_layout()
        out = Path(output_png)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        return out

    def run(self, *args, **kwargs):
        raise NotImplementedError("Use explicit plot_* methods")
