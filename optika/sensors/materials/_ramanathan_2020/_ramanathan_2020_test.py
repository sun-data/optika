import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
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
        na.broadcast_to((100 * u.photon).astype(int), dict(pixel_x=2, pixel_y=2)),
    ],
)
@pytest.mark.parametrize(
    argnames="axis_xy",
    argvalues=[
        None,
        ("pixel_x", "pixel_y"),
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
@pytest.mark.parametrize(
    argnames="width_pixel",
    argvalues=[
        15 * u.um,
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
    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    axis_xy: None | tuple[str, str],
):
    result = _ramanathan_2020.electrons_measured(
        photons_absorbed=photons_absorbed,
        wavelength=wavelength,
        absorption=absorption,
        thickness_implant=thickness_implant,
        width_pixel=width_pixel,
        cce_backsurface=cce_backsurface,
        temperature=temperature,
        axis_xy=axis_xy,
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


def test_electrons_measured_diffusion():
    """
    Inject many photons into a single pixel and check that the spatial spread
    of the diffused charge matches the analytic charge-diffusion width given by
    :func:`optika.sensors.charge_diffusion`.
    """
    num = 41
    axis_xy = ("pixel_x", "pixel_y")

    absorption = 1 / u.um
    thickness_substrate = 14 * u.um
    thickness_depletion = 0 * u.um
    width_pixel = 3 * u.um

    # Place all of the photons in the central pixel of an otherwise empty grid.
    photons = np.zeros((num, num))
    photons[num // 2, num // 2] = 20000
    photons = na.ScalarArray(photons << u.photon, axes=axis_xy).astype(int)

    electrons = _ramanathan_2020.electrons_measured(
        photons_absorbed=photons,
        wavelength=500 * u.nm,
        absorption=absorption,
        thickness_implant=0 * u.um,
        thickness_depletion=thickness_depletion,
        thickness_substrate=thickness_substrate,
        width_pixel=width_pixel,
        cce_backsurface=1,
        axis_xy=axis_xy,
    )

    # The diffused charge should spread beyond the central pixel.
    assert electrons[{axis_xy[0]: num // 2, axis_xy[1]: num // 2}] < electrons.sum(
        axis_xy
    )

    # Physical offset of each pixel from the center of the grid.
    offset_x = (na.arange(0, num, axis=axis_xy[0]) - num // 2) * width_pixel
    offset_y = (na.arange(0, num, axis=axis_xy[1]) - num // 2) * width_pixel

    total = electrons.sum(axis_xy)
    mean_x = (electrons * offset_x).sum(axis_xy) / total
    mean_y = (electrons * offset_y).sum(axis_xy) / total
    var_x = (electrons * np.square(offset_x - mean_x)).sum(axis_xy) / total
    var_y = (electrons * np.square(offset_y - mean_y)).sum(axis_xy) / total
    std_measured = np.sqrt((var_x + var_y) / 2)

    std_expected = optika.sensors.charge_diffusion(
        absorption=absorption,
        thickness_substrate=thickness_substrate,
        thickness_depletion=thickness_depletion,
    )

    assert np.allclose(std_measured, std_expected, rtol=0.05)


def test_electrons_measured_wrap():
    """
    On a grid small compared to the diffusion width, charge that diffuses off
    the grid is lost when ``wrap=False`` but retained (toroidally) when
    ``wrap=True``, so the wrapped grid holds strictly more charge.
    """
    num = 3
    axis_xy = ("pixel_x", "pixel_y")

    photons = np.zeros((num, num))
    photons[num // 2, num // 2] = 5000
    photons = na.ScalarArray(photons << u.photon, axes=axis_xy).astype(int)

    kwargs = dict(
        photons_absorbed=photons,
        wavelength=500 * u.nm,
        absorption=1 / u.um,
        thickness_implant=0 * u.um,
        thickness_depletion=0 * u.um,
        thickness_substrate=14 * u.um,
        width_pixel=2 * u.um,
        cce_backsurface=1,
        axis_xy=axis_xy,
    )

    total_drop = _ramanathan_2020.electrons_measured(**kwargs, wrap=False).sum(axis_xy)
    total_wrap = _ramanathan_2020.electrons_measured(**kwargs, wrap=True).sum(axis_xy)

    assert total_wrap > total_drop
