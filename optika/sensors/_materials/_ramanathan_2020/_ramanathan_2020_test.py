import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from . import _ramanathan_2020

_wavelength = [
    300 * u.nm,
    5 * u.eV,
    na.geomspace(1, 10000, axis="wavelength", num=7) << u.AA,
]

_temperture = [
    300 * u.K,
    na.linspace(0, 300, axis="temperature", num=5) << u.K,
]


@pytest.mark.parametrize("wavelength", _wavelength)
@pytest.mark.parametrize("temperature", _temperture)
def test_quantum_yield_ideal(
    wavelength: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.quantum_yield_ideal(
        wavelength=wavelength,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron / u.photon)
    assert result.shape == na.shape_broadcasted(wavelength, temperature)


@pytest.mark.parametrize("wavelength", _wavelength)
@pytest.mark.parametrize("temperature", _temperture)
def test_fano_factor(
    wavelength: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.fano_factor(
        wavelength=wavelength,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron / u.photon)
    assert result.shape == na.shape_broadcasted(wavelength, temperature)
