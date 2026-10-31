"""First end-to-end run: build Au(111), create a vacancy, save outputs."""

from pathlib import Path

from spm_atombridge.adapters.ase_adapter import ASEAdapter
from spm_atombridge.core.image import Image
import numpy as np

out = Path(__file__).parent / "_output"
out.mkdir(exist_ok=True)

# 1. Build Au(111)
a = ASEAdapter()
s = a.build_fcc111(symbol="Au", size=(3, 3, 3), vacuum=10.0)
print("Built:", s.summary()["formula"], "n_atoms =", len(s.atoms))

# 2. Introduce a vacancy
s2 = a.make_vacancy(s, index=0)
print("After vacancy: n_atoms =", len(s2.atoms))

# 3. Save structure
s2.save(out / "au111_vacancy.xyz")
print("Saved:", out / "au111_vacancy.xyz")

# 4. Save provenance
s2.provenance[-1].save(out / "provenance.json")
print("Saved:", out / "provenance.json")

# 5. Fake experimental image to test Image class
img = Image(data=np.random.rand(64, 64), kind="topography", units="m")
img.save_npy(out / "dummy_image.npy")
print("Saved:", out / "dummy_image.npy")

print("\nAll output files generated in:", out)