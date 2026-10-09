r"""
The time that charge created in the field-free region of a back-illuminated
sensor takes to diffuse to the depletion region,
used by :class:`optika.sensors.diffusion.SlabDiffusionModel`.

The vertical motion of an electron in the field-free region is Brownian,
reflected at the back surface and absorbed at the edge of the depletion
region, and its lateral motion is Brownian and independent of it.
So given the time :math:`T` it takes to reach the depletion region,
the lateral offset of the electron is Gaussian with a variance of
:math:`2 D T` along each axis, where :math:`D` is the diffusion coefficient.
In units of the thickness :math:`L` of the field-free region,
that variance is

.. math::

    s = \frac{2 D T}{L^2},

which is the time a standard Brownian motion started at :math:`u = z / L`
takes to first reach one, reflected at zero.
Its distribution depends only on :math:`\delta = 1 - u`,
the distance from the edge of the depletion region in units of :math:`L`.

Two series give it.
Expanding in the eigenfunctions of the field-free region gives its survival
function,

.. math::

    P(s' > s) = \sum_{j=0}^\infty c_j e^{-\lambda_j s},
    \qquad
    c_j = \frac{4}{\pi k} \sin \frac{k \pi \delta}{2},
    \qquad
    \lambda_j = \frac{k^2 \pi^2}{8},

with :math:`k = 2 j + 1`, which converges quickly for large :math:`s`,
and the method of images gives its distribution function,

.. math::

    P(s' \leq s) = \sum_{n=0}^\infty (-1)^n \left[
        \text{erfc} \frac{2 n + \delta}{\sqrt{2 s}}
        + \text{erfc} \frac{2 n + 2 - \delta}{\sqrt{2 s}}
    \right],

which converges quickly for small :math:`s`.

The charge cloud is the average of a Gaussian over this distribution,
which the models take by quadrature over its quantiles :math:`p`:
Gauss-Legendre quadrature with :obj:`_num_nodes` nodes in :math:`\theta`,
where :math:`p = \sin^2 (\pi \theta / 2)`,
which crowds the nodes toward both tails of the distribution.
So the charge cloud at each depth is a mixture of :obj:`_num_nodes`
Gaussians, with the same weights at every depth,
which the Monte Carlo simulation of
:func:`optika.sensors.electrons_measured` samples directly.

The variance at each node is tabulated once against :math:`\delta`,
as :math:`\log(s / \delta^2)`,
which tends to a constant near the depletion region,
where :math:`s = \delta^2 / 2 w^2` with :math:`w = \text{erfc}^{-1}(p)`.
"""

import math
import functools
import numba
import numpy as np
import named_arrays as na

__all__ = []

_num_nodes = 8
r"""
The number of nodes of the quadrature over the quantiles,
and so the number of Gaussians in the mixture which makes up the charge cloud.
"""

_u_switch = 0.9
"""
The depth, in units of the thickness of the field-free region, below which
the rows of the table are spaced uniformly in depth, and above which they are
spaced uniformly in the logarithm of the distance to the depletion region.
"""

_num_u = 4000
"""The number of rows of the table spaced uniformly in depth."""

_delta_min = 1e-6
"""
The smallest distance to the depletion region, in units of the thickness of
the field-free region, in the table.
Closer to the depletion region the variances scale with the square of the
distance.
"""

_num_delta = 2000
"""The number of rows of the table spaced uniformly in the logarithm of the distance."""

_num_rows = _num_u + _num_delta + 1
"""The number of rows of the table."""

_num_terms = 12
"""The number of terms kept of each series."""

_s_switch = 0.25
"""The variance below which the image series is used."""

_num_bisection = 64
"""The number of bisections of the logarithm of the variance that find a quantile."""


@numba.njit(cache=True)
def _rate(j: int) -> float:  # pragma: nocover
    """The rate :math:`\\lambda_j` of the :math:`j`-th term of the eigenfunction series."""
    k = 2 * j + 1
    return k * k * math.pi**2 / 8


@numba.njit(cache=True)
def _weight(j: int, delta: float) -> float:  # pragma: nocover
    """The weight :math:`c_j` of the :math:`j`-th term of the eigenfunction series."""
    k = 2 * j + 1
    return 4 / (math.pi * k) * math.sin(k * math.pi * delta / 2)


@numba.njit(cache=True)
def _cdf_survival(
    s: float,
    delta: float,
) -> tuple[float, float]:  # pragma: nocover
    """
    The distribution function and the survival function of the variance at
    `s`, each accurate where it is small.

    Parameters
    ----------
    s
        The variance in units of the square of the thickness of the
        field-free region.
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    """
    if s <= 0:
        return 0.0, 1.0

    if s < _s_switch:
        r = math.sqrt(2 * s)
        total = 0.0
        for n in range(_num_terms):
            sign = 1.0 if n % 2 == 0 else -1.0
            total += sign * (
                math.erfc((2 * n + delta) / r) + math.erfc((2 * n + 2 - delta) / r)
            )
        return total, 1 - total

    total = 0.0
    for j in range(_num_terms):
        total += _weight(j, delta) * math.exp(-_rate(j) * s)
    return 1 - total, total


@numba.njit(cache=True)
def _solve(
    delta: float,
    p: float,
    q: float,
) -> float:  # pragma: nocover
    """
    The quantile :math:`p` of the variance, by bisection of its logarithm.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    p
        The quantile.
    q
        One minus the quantile, given separately so that quantiles near one
        keep their precision.
    """
    lower = -120.0
    upper = 10.0
    for _ in range(_num_bisection):
        middle = (lower + upper) / 2
        cdf, survival = _cdf_survival(math.exp(middle), delta)
        if p < 0.5:
            below = cdf < p
        else:
            below = survival > q
        if below:
            lower = middle
        else:
            upper = middle
    return math.exp((lower + upper) / 2)


@functools.cache
def _nodes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    The quantiles :math:`p` at the nodes of the quadrature,
    one minus each quantile, and the weights of the nodes.

    The quadrature is Gauss-Legendre in :math:`\theta`,
    where :math:`p = \sin^2 (\pi \theta / 2)`,
    whose weights are normalized to sum to one.
    """
    x, weight = np.polynomial.legendre.leggauss(_num_nodes)
    theta = (x + 1) / 2
    p = np.square(np.sin(np.pi * theta / 2))
    q = np.square(np.cos(np.pi * theta / 2))
    weight = weight * np.pi / 2 * np.sin(np.pi * theta)
    weight = weight / weight.sum()
    for a in (p, q, weight):
        a.flags.writeable = False
    return p, q, weight


@numba.njit(cache=True)
def _row_delta(
    i: int,
) -> float:  # pragma: nocover
    """
    The distance from the depletion region of row `i` of the table.

    Parameters
    ----------
    i
        The index of the row.
    """
    if i <= _num_u:
        return 1 - _u_switch * i / _num_u
    log_switch = math.log(1 - _u_switch)
    fraction = (i - _num_u) / _num_delta
    return math.exp(log_switch + (math.log(_delta_min) - log_switch) * fraction)


@numba.njit(cache=True)
def _row_coordinate(
    delta: float,
) -> float:  # pragma: nocover
    """
    The fractional index of the row of the table for a distance from the
    depletion region, the last row for distances closer than :obj:`_delta_min`.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    """
    u = 1 - delta
    if u <= _u_switch:
        return u / _u_switch * _num_u
    log_switch = math.log(1 - _u_switch)
    log_delta = max(math.log(delta), math.log(_delta_min))
    fraction = (log_delta - log_switch) / (math.log(_delta_min) - log_switch)
    return _num_u + fraction * _num_delta


@numba.njit(cache=True, parallel=True)
def _build_table(
    p: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    Tabulate :math:`\\log(s / \\delta^2)` at the quantiles `p` on rows of the
    distance :math:`\\delta` from the depletion region.

    Parameters
    ----------
    p
        The quantiles at the nodes of the quadrature.
    q
        One minus each quantile.
    """
    result = np.empty((_num_rows, p.size))
    for i in numba.prange(_num_rows):
        delta = _row_delta(i)
        for k in range(p.size):
            s = _solve(delta, p[k], q[k])
            result[i, k] = math.log(s / (delta * delta))
    return result


@functools.cache
def _table() -> np.ndarray:
    """The table of :func:`_build_table` at the nodes of the quadrature, built once per process."""
    p, q, _ = _nodes()
    result = _build_table(p, q)
    result.flags.writeable = False
    return result


@numba.njit(cache=True)
def _variances(
    delta: float,
    table: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: nocover
    """
    The variance at each node of the quadrature, in units of the square of
    the thickness of the field-free region, for charge created a distance
    `delta` from the depletion region.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero or less in the depletion region.
    table
        The table from :func:`_table`.
    out
        The array of the variance at each node, filled in place.
    """
    if delta <= 0:
        out[:] = 0
        return

    fi = _row_coordinate(delta)
    i = min(int(fi), _num_rows - 2)
    weight = fi - i

    scale = delta * delta
    for k in range(out.size):
        c = (1 - weight) * table[i, k] + weight * table[i + 1, k]
        out[k] = scale * math.exp(c)


@numba.njit(cache=True)
def _variances_array(
    delta: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    :func:`_variances` for each element of an array of distances.

    Parameters
    ----------
    delta
        The distances from the depletion region.
    table
        The table from :func:`_table`.
    """
    result = np.empty((delta.size, table.shape[1]))
    for i in range(delta.size):
        _variances(delta[i], table, result[i])
    return result


def _transit(
    delta: float | na.AbstractScalar,
    axis: str,
) -> tuple[na.ScalarArray, na.ScalarArray]:
    """
    The variance at the nodes of the quadrature over its quantiles,
    in units of the square of the thickness of the field-free region,
    and the weights of the nodes.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero in the depletion region.
    axis
        The logical axis of the nodes.
    """
    _, _, weights = _nodes()
    weight = na.ScalarArray(weights, axes=axis)

    delta = na.as_named_array(delta).explicit
    delta_nd = np.asarray(delta.ndarray, dtype=float)

    s = _variances_array(delta_nd.reshape(-1), _table())
    s = s.reshape(delta_nd.shape + (_num_nodes,))

    return na.ScalarArray(s, axes=delta.axes + (axis,)), weight
