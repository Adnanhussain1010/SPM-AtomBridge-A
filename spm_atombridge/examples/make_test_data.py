"""Generate tiny synthetic files so every adapter can be tested end-to-end."""

from pathlib import Path
import numpy as np
from ase.build import fcc111
from ase.io import write

out = Path(__file__).parent / "_output" / "testdata"
out.mkdir(parents=True, exist_ok=True)

# --- 1. POSCAR for Bader testing ---
atoms = fcc111("Au", size=(2, 2, 2), vacuum=8.0)
write(out / "POSCAR", atoms, format="vasp")

# --- 2. A tiny fake CHGCAR (valid VASP text format) ---
# Real CHGCAR structure: header + charge grid; we generate a small 2x2x2 grid
poscar_text = (out / "POSCAR").read_text()
lines = poscar_text.split("\n")
# Find the first blank-line-separated block (VASP 5+ POSCAR has species line)
# Simpler: build CHGCAR header manually
cell = atoms.cell
ngrid = (4, 4, 4)
chgcar = []
chgcar.append("Au synthetic")           # comment
chgcar.append("   1.0")                  # scale
for v in cell:
    chgcar.append(f"  {v[0]:18.10f} {v[1]:18.10f} {v[2]:18.10f}")
chgcar.append("   Au")
chgcar.append(f"   {len(atoms)}")
chgcar.append("Direct")
for pos in atoms.get_scaled_positions():
    chgcar.append(f"  {pos[0]:18.10f} {pos[1]:18.10f} {pos[2]:18.10f}")
chgcar.append("")
chgcar.append(f"  {ngrid[0]}  {ngrid[1]}  {ngrid[2]}")
# 4x4x4=64 values, each on a new line, 5 per line is VASP convention
vals = np.ones(np.prod(ngrid)) * 0.5
for i in range(0, len(vals), 5):
    chunk = " ".join(f"{v:.8e}" for v in vals[i:i+5])
    chgcar.append("  " + chunk)
chgcar.append("")
(out / "CHGCAR").write_text("\n".join(chgcar))

# --- 3. A tiny LAMMPS dump ---
n_atoms = len(atoms)
dump = [f"ITEM: TIMESTEP\n0"]
dump.append(f"ITEM: NUMBER OF ATOMS\n{n_atoms}")
xlo, xhi = 0.0, cell[0][0]
ylo, yhi = 0.0, cell[1][1]
zlo, zhi = 0.0, cell[2][2]
dump.append(f"ITEM: BOX BOUNDS pp pp pp\n{xlo} {xhi}\n{ylo} {yhi}\n{zlo} {zhi}")
dump.append("ITEM: ATOMS id type x y z")
for i, (p, s) in enumerate(zip(atoms.get_positions(), atoms.get_chemical_symbols()), 1):
    dump.append(f"{i} 1 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
(out / "traj.dump").write_text("\n".join(dump))

# --- 4. A tiny fake AFM image as .npy ---
# Simulated atomic corrugation on Au(111): hexagonal lattice
N = 128
xx, yy = np.meshgrid(np.linspace(0, 5, N), np.linspace(0, 5, N))
# Au(111) lattice constant 2.88 Å
a = 2.88
k = 2 * np.pi / a
z = 0.3 * np.cos(k * xx) * np.cos(k * yy * np.sqrt(3) / 2) + 0.3
np.save(out / "fake_afm.npy", z.astype(np.float32))

# --- 5. A tiny fake I-V curve ---
V = np.linspace(-1.0, 1.0, 201)
I = 0.4 * V + 0.05 * V**3      # nonlinear response
np.savetxt(out / "fake_iv.dat", np.c_[V, I], header="bias_V current_nA",
           comments="# ")

# --- 6. A tiny fake KPFM image (CPD map, volts) ---
cpd = 0.05 * np.cos(k * xx) * np.cos(k * yy * np.sqrt(3) / 2) + 0.30
np.save(out / "fake_cpd.npy", cpd.astype(np.float32))

print(f"Wrote {len(list(out.iterdir()))} files to {out}:")
for f in sorted(out.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")