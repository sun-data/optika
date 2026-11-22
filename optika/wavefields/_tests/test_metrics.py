"""Tests of the point-spread-function metrics against an analytic Gaussian."""

import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


def _gaussian_psf(
    sigma: u.Quantity = 5 * u.um,
    center: u.Quantity = 1 * u.um,
    width: u.Quantity = 80 * u.um,
    num: int = 401,
) -> tuple[na.AbstractScalar, na.Cartesian2dVectorArray]:
    position = na.Cartesian2dVectorLinearSpace(
        start=center - width / 2,
        stop=center + width / 2,
        axis=na.Cartesian2dVectorArray("psf_x", "psf_y"),
        num=num,
    ).explicit
    r2 = np.square(position.x - center) + np.square(position.y - center)
    intensity = np.exp(-r2 / (2 * np.square(sigma)))
    return intensity, position


def test_encircled_energy_radius():
    """
    The 50% encircled-energy radius of a symmetric Gaussian is
    :math:`\\sigma \\sqrt{2 \\ln 2}`.
    """
    sigma = 5 * u.um
    intensity, position = _gaussian_psf(sigma=sigma)

    result = optika.wavefields.encircled_energy_radius(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        fraction=0.5,
    )

    expected = sigma * np.sqrt(2 * np.log(2))
    assert u.isclose(result, expected, rtol=2e-2)


def test_encircled_energy_radius_broadcast():
    """
    The metric must support additional broadcast axes, for example a grid
    of point-spread functions over the field of view.
    """
    sigma = na.ScalarArray(np.array([3, 6]) * u.um, axes=("field",))
    intensity, position = _gaussian_psf()
    r2 = np.square(position.x - 1 * u.um) + np.square(position.y - 1 * u.um)
    intensity = np.exp(-r2 / (2 * np.square(sigma)))

    result = optika.wavefields.encircled_energy_radius(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        fraction=0.5,
    )

    assert result.shape == {"field": 2}
    expected = sigma * np.sqrt(2 * np.log(2))
    assert np.all(np.abs(result - expected) < 0.05 * expected)


def test_ensquared_energy():
    """
    The energy of a symmetric Gaussian inside a centered square of width
    :math:`w` is :math:`\\mathrm{erf}^2(w / 2 \\sqrt{2} \\sigma)`.
    """
    import scipy.special

    sigma = 5 * u.um
    width = 15 * u.um
    intensity, position = _gaussian_psf(sigma=sigma)

    result = optika.wavefields.ensquared_energy(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        width=width,
    )

    arg = (width / (2 * np.sqrt(2) * sigma)).to_value(u.dimensionless_unscaled)
    expected = np.square(scipy.special.erf(arg))
    assert np.abs(float(na.as_named_array(result).ndarray) - expected) < 2e-2


def test_metrics_position_3d():
    """
    Each metric must accept a 3D position (e.g. from a wavefield sampled on
    a curved sensor) and give the same result as its 2D projection.
    """
    sigma = 5 * u.um
    intensity, position = _gaussian_psf(sigma=sigma)
    position_3d = na.Cartesian3dVectorArray(
        x=position.x,
        y=position.y,
        z=1 * u.um,
    )
    axis = ("psf_x", "psf_y")

    result = optika.wavefields.encircled_energy_radius(
        intensity=intensity,
        position=position_3d,
        axis=axis,
    )
    expected = optika.wavefields.encircled_energy_radius(
        intensity=intensity,
        position=position,
        axis=axis,
    )
    assert u.isclose(result, expected)

    width = 15 * u.um
    result = optika.wavefields.ensquared_energy(
        intensity=intensity,
        position=position_3d,
        axis=axis,
        width=width,
    )
    expected = optika.wavefields.ensquared_energy(
        intensity=intensity,
        position=position,
        axis=axis,
        width=width,
    )
    assert u.isclose(
        na.as_named_array(result).ndarray,
        na.as_named_array(expected).ndarray,
    )

    result = optika.wavefields.fwhm(
        intensity=intensity,
        position=position_3d,
        axis=axis,
    )
    expected = optika.wavefields.fwhm(
        intensity=intensity,
        position=position,
        axis=axis,
    )
    assert u.isclose(result.x, expected.x)
    assert u.isclose(result.y, expected.y)


def test_fwhm():
    """
    The full width at half maximum of a Gaussian is
    :math:`2 \\sigma \\sqrt{2 \\ln 2}` along both axes.
    """
    sigma = 5 * u.um
    intensity, position = _gaussian_psf(sigma=sigma)

    result = optika.wavefields.fwhm(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
    )

    expected = 2 * sigma * np.sqrt(2 * np.log(2))
    assert u.isclose(result.x, expected, rtol=2e-2)
    assert u.isclose(result.y, expected, rtol=2e-2)


def test_fwhm_outside() -> None:
    """
    The full width at half maximum of a point-spread function whose cut does
    not fall below half of its peak within the samples is not-a-number, and
    the width of the others along the same axes is unchanged.
    """
    sigma = na.ScalarArray(np.array([5, 200]) * u.um, axes="field")
    _, position = _gaussian_psf()
    r2 = np.square(position.x - 1 * u.um) + np.square(position.y - 1 * u.um)
    intensity = np.exp(-r2 / (2 * np.square(sigma)))

    result = optika.wavefields.fwhm(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
    )

    assert result.shape == dict(field=2)
    expected = 2 * sigma[dict(field=0)] * np.sqrt(2 * np.log(2))
    assert np.abs(result.x[dict(field=0)] - expected) < 2e-2 * expected
    assert np.abs(result.y[dict(field=0)] - expected) < 2e-2 * expected
    assert np.isnan(result.x[dict(field=1)])
    assert np.isnan(result.y[dict(field=1)])


@pytest.mark.parametrize(
    argnames="sigma",
    argvalues=[
        5 * u.um,
        na.ScalarArray(np.array([3, 6]) * u.um, axes="field"),
    ],
)
@pytest.mark.parametrize(
    argnames="unit",
    argvalues=[
        u.dimensionless_unscaled,
        u.W / u.m**2,
    ],
)
def test_mtf(
    sigma: u.Quantity | na.AbstractScalar,
    unit: u.UnitBase,
) -> None:
    """
    The MTF of a Gaussian is :math:`\\exp(-2 \\pi^2 \\sigma^2 f^2)`, on a grid
    of frequencies centered on zero and spaced by one over the width of the
    grid, whatever the unit of the intensity, and for each of a grid of
    point-spread functions.
    """
    width = 80 * u.um
    num = 401
    intensity, position = _gaussian_psf(width=width, num=num)
    r2 = np.square(position.x - 1 * u.um) + np.square(position.y - 1 * u.um)
    intensity = np.exp(-r2 / (2 * np.square(sigma))) * unit

    result = optika.wavefields.mtf(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
    )

    spacing = width / (num - 1)
    frequency_x = (na.arange(0, num, axis="psf_x") - num // 2) / (num * spacing)
    frequency_y = (na.arange(0, num, axis="psf_y") - num // 2) / (num * spacing)
    assert np.allclose(result.inputs.x, frequency_x, rtol=1e-12, atol=0 / u.um)
    assert np.allclose(result.inputs.y, frequency_y, rtol=1e-12, atol=0 / u.um)

    frequency = np.square(frequency_x) + np.square(frequency_y)
    expected = np.exp(-2 * np.square(np.pi * sigma) * frequency)
    assert np.allclose(result.outputs, expected, rtol=0, atol=1e-9)


def test_mtf_position_3d() -> None:
    """The MTF of samples with 3D positions is that of their 2D projection."""
    intensity, position = _gaussian_psf()
    position_3d = na.Cartesian3dVectorArray(position.x, position.y, 1 * u.um)
    axis = ("psf_x", "psf_y")

    result = optika.wavefields.mtf(intensity, position_3d, axis)
    expected = optika.wavefields.mtf(intensity, position, axis)

    assert np.all(result.inputs == expected.inputs)
    assert np.all(result.outputs == expected.outputs)


@pytest.mark.parametrize(
    argnames="component",
    argvalues=["x", "y"],
)
def test_mtf_irregular(component: str) -> None:
    """A point-spread function off a regular grid has no MTF."""
    num = 11
    intensity, position = _gaussian_psf(num=num)
    index_x = na.arange(0, num, axis="psf_x")
    index_y = na.arange(0, num, axis="psf_y")
    bump = ((index_x == 3) & (index_y == 4)) * 0.01 * u.um
    if component == "x":
        position = na.Cartesian2dVectorArray(position.x + bump, position.y)
    else:
        position = na.Cartesian2dVectorArray(position.x, position.y + bump)

    with pytest.raises(ValueError, match="regular grid"):
        optika.wavefields.mtf(intensity, position, ("psf_x", "psf_y"))


def test_mtf_single() -> None:
    """A grid must have at least two samples along each axis."""
    intensity, position = _gaussian_psf(num=11)
    index = dict(psf_y=slice(0, 1))

    with pytest.raises(ValueError, match="at least two"):
        optika.wavefields.mtf(
            intensity[index],
            position[index],
            ("psf_x", "psf_y"),
        )


def _gaussian_density(
    sigma: u.Quantity | na.AbstractScalar,
    width: u.Quantity,
    num: int = 401,
) -> tuple[na.AbstractScalar, na.Cartesian2dVectorArray]:
    """
    A Gaussian point-spread function as an intensity per unit area which
    integrates to one over the whole plane, on a grid which may hold only
    part of it.

    Parameters
    ----------
    sigma
        The standard deviation of the Gaussian.
    width
        The width of the grid.
    num
        The number of samples along each axis of the grid.
    """
    _, position = _gaussian_psf(width=width, num=num)
    r2 = np.square(position.x - 1 * u.um) + np.square(position.y - 1 * u.um)
    variance = np.square(sigma)
    intensity = np.exp(-r2 / (2 * variance)) / (2 * np.pi * variance)
    return intensity, position


def test_mtf_normalize() -> None:
    """
    Normalized to the power of the whole image, the MTF of a Gaussian is
    :math:`\\exp(-2 \\pi^2 \\sigma^2 f^2)` on a grid which holds all of it,
    and that times the fraction of the power on a grid which does not,
    :math:`\\mathrm{erf}^2(w / 2 \\sqrt{2} \\sigma)`.
    """
    import scipy.special

    sigma = 5 * u.um
    width = na.ScalarArray(np.array([80, 15]) * u.um, axes="grid")
    num = 401
    intensity, position = _gaussian_density(sigma, width=width, num=num)

    result = optika.wavefields.mtf(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        normalize=False,
    )

    spacing = width / (num - 1)
    frequency_x = (na.arange(0, num, axis="psf_x") - num // 2) / (num * spacing)
    frequency_y = (na.arange(0, num, axis="psf_y") - num // 2) / (num * spacing)
    frequency = np.square(frequency_x) + np.square(frequency_y)
    expected = np.exp(-2 * np.square(np.pi * sigma) * frequency)

    # each sample stands for a cell of the grid, so the cells reach half of
    # a spacing past the outermost samples
    arg = num * spacing / (2 * np.sqrt(2) * sigma)
    fraction = np.square(scipy.special.erf(arg.to(u.dimensionless_unscaled).value))
    origin = dict(psf_x=num // 2, psf_y=num // 2)
    assert np.allclose(result.outputs[origin], fraction, rtol=1e-5)

    wide = dict(grid=0)
    assert np.allclose(result.outputs[wide], expected[wide], rtol=0, atol=1e-9)

    result_normalized = optika.wavefields.mtf(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
    )
    assert np.allclose(
        result.outputs,
        result_normalized.outputs * result.outputs[origin],
        rtol=1e-12,
    )


def test_mtf_normalize_not_density() -> None:
    """
    Normalized to the power of the whole image, the intensity must be per
    unit area, so that its integral over the grid is dimensionless.
    """
    intensity, position = _gaussian_psf(num=11)

    with pytest.raises(u.UnitConversionError):
        optika.wavefields.mtf(
            intensity=intensity,
            position=position,
            axis=("psf_x", "psf_y"),
            normalize=False,
        )


def test_encircled_energy_radius_normalize() -> None:
    """
    Normalized to the power of the whole image, the 50% encircled-energy
    radius of a Gaussian is :math:`\\sigma \\sqrt{2 \\ln 2}` on a grid which
    holds that circle but not all of the light, and not-a-number on a grid
    which holds less than half of the light.
    """
    sigma = na.ScalarArray(np.array([5, 20]) * u.um, axes="field")
    intensity, position = _gaussian_density(sigma, width=16 * u.um)
    kwargs = dict(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        fraction=0.5,
    )

    result = optika.wavefields.encircled_energy_radius(normalize=False, **kwargs)

    expected = sigma * np.sqrt(2 * np.log(2))
    expected = np.where(sigma < 10 * u.um, expected, np.nan * u.um)
    assert np.allclose(result, expected, rtol=1e-2, equal_nan=True)

    # normalized to the light on the grid, the radius is smaller
    index = dict(field=0)
    result_normalized = optika.wavefields.encircled_energy_radius(**kwargs)
    assert result_normalized[index] < 0.9 * expected[index]


def test_ensquared_energy_normalize() -> None:
    """
    Normalized to the power of the whole image, the energy of a Gaussian
    inside a centered square of width :math:`w` is
    :math:`\\mathrm{erf}^2(w / 2 \\sqrt{2} \\sigma)`, however much of the
    light is outside of the grid.
    """
    import scipy.special

    sigma = 5 * u.um
    width = 10 * u.um
    width_grid = na.ScalarArray(np.array([80, 16]) * u.um, axes="grid")
    intensity, position = _gaussian_density(sigma, width=width_grid)

    result = optika.wavefields.ensquared_energy(
        intensity=intensity,
        position=position,
        axis=("psf_x", "psf_y"),
        width=width,
        normalize=False,
    )

    arg = (width / (2 * np.sqrt(2) * sigma)).to_value(u.dimensionless_unscaled)
    expected = np.square(scipy.special.erf(arg))
    assert na.unit(result) is None
    assert np.allclose(result, expected, rtol=2e-2)
