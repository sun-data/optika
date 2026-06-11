"""Tests of the point-spread-function metrics against an analytic Gaussian."""

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
