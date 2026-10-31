import numpy as np

from spm_atombridge.core.image import Image
from spm_atombridge.core.provenance import ProvenanceRecord
from spm_atombridge.core.scalarfield import ScalarField
from spm_atombridge.core.spectrum import Spectrum
from spm_atombridge.core.structure import Structure
from spm_atombridge.adapters.base import Adapter
from spm_atombridge.adapters.ase_adapter import ASEAdapter


def test_structure_from_xyz(tmp_path):
    f = tmp_path / "h2.xyz"
    f.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n")
    s = Structure.from_file(str(f))
    assert s.summary()["n_atoms"] == 2
    assert s.provenance[0].action == "load_structure"


def test_provenance_hash(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    rec = ProvenanceRecord.from_file("test", "unit", f)
    assert len(rec.input_hash) == 64
    assert rec.versions["python"]


def test_scalarfield_stats():
    sf = ScalarField(data=np.arange(8).reshape(2, 2, 2), units="e/A^3", axis="charge_density")
    s = sf.stats()
    assert s["shape"] == [2, 2, 2]
    assert s["max"] == 7.0
    assert sf.plane("z", -1).shape == (2, 2)


def test_image_stats():
    img = Image(data=np.random.rand(10, 10), kind="kpfm", units="V")
    assert img.stats()["kind"] == "kpfm"
    assert img.shape == (10, 10)


def test_spectrum_zero_bias():
    x = np.linspace(-1, 1, 101)
    y = 0.5 * x
    sp = Spectrum(x=x, y=y)
    g = sp.zero_bias_conductance()
    assert abs(g - 0.5) < 1e-6


def test_adapter_capability_report():
    rep = Adapter.capability_report()
    assert "available" in rep


def test_ase_build_and_vacancy():
    a = ASEAdapter()
    assert a.is_available()
    s = a.build_fcc111(symbol="Au", size=(2, 2, 2), vacuum=8.0)
    n0 = len(s.atoms)
    s2 = a.make_vacancy(s, index=0)
    assert len(s2.atoms) == n0 - 1
    assert s2.provenance[-1].action == "make_vacancy"