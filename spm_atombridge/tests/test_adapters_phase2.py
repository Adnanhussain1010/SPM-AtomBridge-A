import numpy as np
import pytest

from spm_atombridge.adapters.bader_adapter import BaderAdapter
from spm_atombridge.adapters.plotting_adapter import PlottingAdapter
from spm_atombridge.core.image import Image
from spm_atombridge.core.scalarfield import ScalarField
from spm_atombridge.core.spectrum import Spectrum


def test_bader_input_check_missing(tmp_path):
    a = BaderAdapter()
    status = a.check_inputs(tmp_path)
    assert status["ready"] is False
    assert "charge density" in status["diagnostic"].lower()


def test_bader_input_check_ready(tmp_path):
    (tmp_path / "CHGCAR").write_text("dummy")
    (tmp_path / "POSCAR").write_text("dummy")
    status = BaderAdapter().check_inputs(tmp_path)
    assert status["ready"] is True


def test_plot_image(tmp_path):
    img = Image(data=np.random.rand(32, 32), kind="kpfm", units="V")
    out = PlottingAdapter().plot_image(img, tmp_path / "img.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_scalarfield(tmp_path):
    sf = ScalarField(data=np.random.rand(16, 16), units="e", axis="bader_charge")
    out = PlottingAdapter().plot_scalarfield(sf, tmp_path / "sf.png")
    assert out.exists()


def test_plot_spectrum(tmp_path):
    x = np.linspace(-1, 1, 50); y = 0.3 * x
    sp = Spectrum(x=x, y=y)
    out = PlottingAdapter().plot_spectrum(sp, tmp_path / "sp.png")
    assert out.exists()