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


@pytest.mark.parametrize("temperature", _temperture)
def test_energy_bandgap(
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.energy_bandgap(
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.eV)
    assert result.shape == temperature.shape


@pytest.mark.parametrize("wavelength", _wavelength)
@pytest.mark.parametrize("temperature", _temperture)
def test_energy_pair(
    wavelength: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.energy_pair(
        wavelength=wavelength,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.eV)
    assert result.shape == na.shape_broadcasted(wavelength, temperature)


@pytest.mark.parametrize("temperature", _temperture)
def test_energy_pair_inf(
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.energy_pair_inf(
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.eV)
    assert result.shape == temperature.shape


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


@pytest.mark.parametrize("temperature", _temperture)
def test_fano_factor_inf(
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.fano_factor_inf(
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron / u.photon)
    assert result.shape == temperature.shape


@pytest.mark.parametrize(
    argnames="photons_absorbed",
    argvalues=[
        100 * u.photon,
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength,absorption",
    argvalues=[
        (
            w,
            None,
        )
        for w in _wavelength
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        2000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.5,
    ],
)
@pytest.mark.parametrize("temperature", _temperture)
def test_electrons_measured(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.ScalarArray,
):
    result = _ramanathan_2020.electrons_measured(
        photons_absorbed=photons_absorbed,
        wavelength=wavelength,
        absorption=absorption,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron)

    shape = na.shape_broadcasted(
        photons_absorbed,
        wavelength,
        absorption,
        thickness_implant,
        cce_backsurface,
        temperature,
    )

    assert result.shape == shape
