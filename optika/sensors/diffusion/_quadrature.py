"""
The quadrature used to average the quantities of a charge cloud over the depth
at which photons are absorbed.
"""

from typing import Callable
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na

__all__ = []

_optical_depth_min = 1e-30
"""
The smallest optical depth of the light-sensitive region used by the averages
over depth, see :func:`_absorption_positive`.
"""


def _absorption_positive(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
) -> u.Quantity | na.AbstractScalar:
    """
    The absorption coefficient, raised if necessary so that the optical depth
    of the light-sensitive region is at least :obj:`_optical_depth_min`.

    The few photons that a weakly absorbing sensor absorbs are absorbed
    uniformly in depth, and the averages over depth of a sensor which does not
    absorb at all are taken to be that limit.
    Evaluating them at a tiny optical depth instead of zero gives the limit to
    machine precision, since the depth distribution then differs from a
    uniform one by about that optical depth.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive region.
    thickness_substrate
        The thickness of the light-sensitive region.
    """
    return np.maximum(absorption, _optical_depth_min / thickness_substrate)


_num_gauss_legendre = 32
"""
The number of Gauss-Legendre nodes used on each subinterval by
:func:`_integrate_gauss_legendre`.
Chosen so that the averages over depth computed by
:meth:`optika.sensors.diffusion.AbstractDiffusionModel.average_depth`
are accurate to about one part in :math:`10^6` over the full range of
optical depths encountered by a silicon sensor between 1 and 10000 angstroms.
"""


def _integrate_gauss_legendre(
    integrand: Callable[[na.AbstractScalar], na.AbstractScalar],
    lower: float | na.AbstractScalar,
    upper: float | na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    """
    Integrate `integrand` between `lower` and `upper` using Gauss-Legendre
    quadrature with :obj:`_num_gauss_legendre` nodes.

    The limits may be arrays, in which case a separate quadrature rule is
    applied to every element, and `axis` is the logical axis along which the
    nodes are placed.

    Parameters
    ----------
    integrand
        The function to integrate.
    lower
        The lower limit of integration.
    upper
        The upper limit of integration.
    axis
        The logical axis along which to place the quadrature nodes.
        Consumed by the sum, so it does not appear in the result.
    """
    nodes, weights = scipy.special.roots_legendre(_num_gauss_legendre)

    nodes = na.ScalarArray(nodes, axes=(axis,))
    weights = na.ScalarArray(weights, axes=(axis,))

    half = (upper - lower) / 2
    center = (upper + lower) / 2

    return half * (weights * integrand(half * nodes + center)).sum(axis)
