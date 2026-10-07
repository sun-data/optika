"""
The quantities of a Gaussian charge cloud, used by
:class:`optika.sensors.diffusion.JanesickDiffusionModel`.
"""

import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na

__all__ = []


def _absorbed_ramp(
    optical_depth: float | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The integral of a weight falling linearly from one to zero across a layer,
    weighted by the probability of a photon being absorbed at each depth.

    For a layer of thickness :math:`L` and an absorption coefficient
    :math:`\alpha`, this is

    .. math::

        \int_0^L \left( 1 - \frac{x}{L} \right) \alpha e^{-\alpha x} dx
            = \frac{\alpha L + e^{-\alpha L} - 1}{\alpha L},

    which goes to zero with the optical depth :math:`\alpha L`,
    so that a layer of zero thickness contributes nothing.

    Parameters
    ----------
    optical_depth
        The optical depth :math:`\alpha L` of the layer.
    """
    where = optical_depth > 0
    optical_depth = np.where(where, optical_depth, 1)
    result = (optical_depth + np.expm1(-optical_depth)) / optical_depth
    return np.where(where, result, 0)


def _width_average(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
    width_backsurface: None | u.Quantity | na.AbstractScalar,
    width_depletion: None | u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The square root of the variance of the width profile of
    :class:`~optika.sensors.diffusion.JanesickDiffusionModel` averaged over the
    absorption depth, in closed form.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive region.
    thickness_substrate
        The thickness of the light-sensitive region.
    thickness_depletion
        The thickness of the depletion region.
    width_backsurface
        The width at the back surface, or :obj:`None` for the thickness of the
        field-free region.
    width_depletion
        The spread acquired crossing the depletion region, or :obj:`None` for
        none.
    """
    s = thickness_substrate
    d = thickness_depletion

    # The field-free region, which vanishes if the depletion region is
    # thicker than the light-sensitive region.
    f = np.maximum(s - d, 0 * s)

    if width_backsurface is None:
        width_backsurface = f

    az_s = (absorption * s).to(u.dimensionless_unscaled).value
    az_f = (absorption * f).to(u.dimensionless_unscaled).value
    az_d = az_s - az_f

    # The fraction of the photons entering the sensor which are absorbed
    # in the light-sensitive region.
    absorbed = -np.expm1(-az_s)

    variance = np.square(width_backsurface) * _absorbed_ramp(az_f)

    if width_depletion is not None:
        # The fraction of the depletion region crossed by charge created at
        # the back surface, less than all of it only if the depletion region
        # extends beyond the light-sensitive region.
        share = np.minimum(s / np.where(d > 0 * d, d, s), 1)
        share = share.to(u.dimensionless_unscaled).value
        crossed = -np.expm1(-az_f) + np.exp(-az_f) * _absorbed_ramp(az_d)
        variance = variance + np.square(width_depletion) * share * crossed

    return np.sqrt(variance / absorbed).to(u.um)


def _ratio(
    width: u.Quantity | na.AbstractScalar,
    width_pixel: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The width of a charge cloud in units of the pixel width,
    zero for a pixel of zero width.

    Parameters
    ----------
    width
        The standard deviation of the charge cloud.
    width_pixel
        The width of a pixel.
    """
    where = width_pixel > 0 * width_pixel
    width_pixel = np.where(where, width_pixel, 1 * u.um)
    r = (width / width_pixel).to(u.dimensionless_unscaled).value
    return np.where(where, r, 0)


def _probability_same_pixel(
    sigma: float | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The probability that two electrons from the same photon land in the same
    column (or row) of pixels, averaged over the uniformly-distributed
    sub-pixel position of the photon.

    Both electrons start from the same sub-pixel position
    :math:`u \sim \mathcal{U}(-1/2, 1/2)` and are displaced independently by
    :math:`\mathcal{N}(0, \sigma^2)`, so their separation is
    :math:`\delta \sim \mathcal{N}(0, 2 \sigma^2)`, and the probability that
    they are binned into the same pixel is
    :math:`\left\langle (1 - |\delta|)_+ \right\rangle`,
    which evaluates to

    .. math::

        d(\sigma) = \text{erf} \left( \frac{1}{2 \sigma} \right)
            - \frac{2 \sigma}{\sqrt{\pi}} \left( 1 - e^{-1 / 4 \sigma^2} \right).

    Parameters
    ----------
    sigma
        The standard deviation of the charge cloud in units of the pixel width.
    """
    where = sigma > 0
    sigma = np.where(where, sigma, 1)
    result = scipy.special.erf(1 / (2 * sigma)) + (
        2 * sigma / np.sqrt(np.pi) * np.expm1(-1 / (4 * np.square(sigma)))
    )
    return np.where(where, result, 1)


def _kernel_1d(
    sigma: float | na.AbstractScalar,
    index_pixel: na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The fraction of a Gaussian charge cloud collected in each column (or row)
    of pixels, relative to the one it was created in, averaged over the
    uniformly-distributed sub-pixel position of the photon.

    Parameters
    ----------
    sigma
        The standard deviation of the charge cloud in units of the pixel width.
    index_pixel
        The indices of the pixels, relative to the one the photon was
        absorbed in.
    """
    where = sigma > 0
    n = index_pixel

    x = 1 / np.where(where, sigma, 1)
    x2 = np.square(x)

    c = 1 / (x * np.sqrt(2 * np.pi))

    def g(m: na.AbstractScalar) -> na.AbstractScalar:
        return np.exp(-x2 * m / 2)

    # The kernel is a second difference of m erf(x m / sqrt(2)).
    # Writing erf as 1 - erfc, the linear part |m| has a second difference
    # of two at the center and zero elsewhere, so it is taken out
    # analytically, leaving terms which are small away from the center
    # instead of a difference of terms near one.
    def e(m: na.AbstractScalar) -> na.AbstractScalar:
        m = np.abs(m)
        return m * scipy.special.erfc(x * m / np.sqrt(2))

    g1 = g(np.square(n - 1))
    g2 = -2 * g(np.square(n))
    g3 = g(np.square(n + 1))

    e1 = e(n - 1) / 2
    e2 = -e(n)
    e3 = e(n + 1) / 2

    result = c * (g1 + g2 + g3) - (e1 + e2 + e3) + (n == 0)
    result = np.maximum(result, 0)

    return np.where(where, result, n == 0)
