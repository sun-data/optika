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
The two electrons created by one photon diffuse independently,
so the sum of their variances, which sets the probability that they are
collected in the same pixel, has a distribution of its own:

.. math::

    P(t' > t) &= \sum_{j=0}^\infty \left( A_j + B_j t \right) e^{-\lambda_j t}, \\
    P(t' \leq t) &= \sum_{n=0}^\infty (-1)^n (n + 1) \left[
        \text{erfc} \frac{2 n + 2 \delta}{\sqrt{2 t}}
        + 2 \, \text{erfc} \frac{2 n + 2}{\sqrt{2 t}}
        + \text{erfc} \frac{2 n + 4 - 2 \delta}{\sqrt{2 t}}
    \right],

where :math:`B_j = c_j^2 \lambda_j` and

.. math::

    A_j = c_j \left( 2 - c_j + 2 \lambda_j \sum_{m \neq j} \frac{c_m}{\lambda_m - \lambda_j} \right).

The averages of the models are taken over the quantiles of these
distributions, which are tabulated once for every sensor,
since they are dimensionless.
A quantile :math:`p` is reached through :math:`w = \text{erfc}^{-1}(p)`,
which is :math:`|Z| / \sqrt{2}` for a standard normal :math:`Z`,
so that the Monte Carlo simulation draws it without an inverse error
function, and near the depletion region, where :math:`s = \delta^2 / 2 w^2`,
the table holds :math:`\log(2 w^2 s / \delta^2)`, which is nearly zero,
as a function of :math:`\log(w / \delta)`, which it depends on alone.
"""

import math
import functools
import random
import numba
import numpy as np
import named_arrays as na

__all__ = []

_kind_single = 0
"""Tabulate the variance of the cloud of a single electron."""

_kind_pair = 1
"""Tabulate the sum of the variances of the clouds of two electrons."""

_u_switch = 0.9
"""
The depth, in units of the thickness of the field-free region, below which
the rows of the table are spaced uniformly in depth, and above which they are
spaced uniformly in the logarithm of the distance to the depletion region.
"""

_num_u = 180
"""The number of rows of the table spaced uniformly in depth."""

_delta_min = 1e-6
"""
The smallest distance to the depletion region, in units of the thickness of
the field-free region, in the table.
Closer to the depletion region the quantiles scale with the distance.
"""

_num_delta = 100
"""The number of rows of the table spaced uniformly in the logarithm of the distance."""

_num_rows = _num_u + _num_delta + 1
"""The number of rows of the table."""

_w_min = 1e-9
"""
The smallest value of :math:`w = \\text{erfc}^{-1}(p)` in the table.
Below it, which a standard normal reaches with a probability of about
:math:`10^{-9}`, the quantile is found directly.
"""

_w_max = 6.0
"""
The largest value of :math:`w` in the table,
beyond which the quantile is extrapolated as the distribution near the
depletion region, which a standard normal reaches with a probability of about
:math:`10^{-17}`.
"""

_v_min = math.log(_w_min)
"""The smallest value of :math:`\\log(w / \\delta)` in the table."""

_v_max = math.log(_w_max) - math.log(_delta_min)
"""The largest value of :math:`\\log(w / \\delta)` in the table."""

_num_v = 3201
"""
The number of columns of the table, which interpolates the quantiles to
about one part in :math:`10^{5}`.
"""

_num_terms = 12
"""The number of terms kept of each series."""

_num_terms_coefficients = 1000
"""The number of terms of the sum over :math:`m` in :math:`A_j`."""

_s_switch_single = 0.25
"""The variance below which the image series is used for a single electron."""

_s_switch_pair = 0.5
"""The variance below which the image series is used for a pair of electrons."""

_num_bisection = 64
"""The number of bisections of the logarithm of the variance that find a quantile."""

_num_nodes = 32
"""
The number of nodes of the quadrature over the quantiles,
which averages a function of the variance to a few parts in :math:`10^{6}`.
"""


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
def _coefficients(
    delta: float,
    kind: int,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: nocover
    """
    The coefficients :math:`A_j` and :math:`B_j` of the eigenfunction series of
    the survival function, for charge created a distance `delta` from the
    depletion region.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    """
    a = np.empty(_num_terms)
    b = np.empty(_num_terms)
    for j in range(_num_terms):
        c_j = _weight(j, delta)
        if kind == _kind_single:
            a[j] = c_j
            b[j] = 0.0
        else:
            rate_j = _rate(j)
            total = 0.0
            for m in range(_num_terms_coefficients):
                if m != j:
                    total += _weight(m, delta) / (_rate(m) - rate_j)
            a[j] = c_j * (2 - c_j + 2 * rate_j * total)
            b[j] = c_j * c_j * rate_j
    return a, b


@numba.njit(cache=True)
def _cdf_survival(
    s: float,
    delta: float,
    kind: int,
    a: np.ndarray,
    b: np.ndarray,
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
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    a
        The coefficients :math:`A_j` from :func:`_coefficients`.
    b
        The coefficients :math:`B_j` from :func:`_coefficients`.
    """
    if s <= 0:
        return 0.0, 1.0

    if kind == _kind_single:
        switch = _s_switch_single
    else:
        switch = _s_switch_pair

    if s < switch:
        r = math.sqrt(2 * s)
        total = 0.0
        for n in range(_num_terms):
            sign = 1.0 if n % 2 == 0 else -1.0
            if kind == _kind_single:
                total += sign * (
                    math.erfc((2 * n + delta) / r) + math.erfc((2 * n + 2 - delta) / r)
                )
            else:
                total += (
                    sign
                    * (n + 1)
                    * (
                        math.erfc((2 * n + 2 * delta) / r)
                        + 2 * math.erfc((2 * n + 2) / r)
                        + math.erfc((2 * n + 4 - 2 * delta) / r)
                    )
                )
        return total, 1 - total

    total = 0.0
    for j in range(_num_terms):
        total += (a[j] + b[j] * s) * math.exp(-_rate(j) * s)
    return 1 - total, total


@numba.njit(cache=True)
def _solve(
    delta: float,
    w: float,
    kind: int,
    a: np.ndarray,
    b: np.ndarray,
) -> float:  # pragma: nocover
    """
    The quantile :math:`p = \\text{erfc}(w)` of the variance,
    by bisection of its logarithm.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    w
        The inverse complementary error function of the quantile.
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    a
        The coefficients :math:`A_j` from :func:`_coefficients`.
    b
        The coefficients :math:`B_j` from :func:`_coefficients`.
    """
    p = math.erfc(w)
    q = math.erf(w)
    lower = -120.0
    upper = 10.0
    for _ in range(_num_bisection):
        middle = (lower + upper) / 2
        cdf, survival = _cdf_survival(math.exp(middle), delta, kind, a, b)
        if p < 0.5:
            below = cdf < p
        else:
            below = survival > q
        if below:
            lower = middle
        else:
            upper = middle
    return math.exp((lower + upper) / 2)


@numba.njit(cache=True)
def _level(
    delta: float,
    kind: int,
) -> float:  # pragma: nocover
    """
    The distance whose first-passage time the variance is near the depletion
    region, in units of the thickness of the field-free region:
    `delta` for one electron, and twice that for the sum over two.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    """
    if kind == _kind_single:
        return delta
    return 2 * delta


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
    kind: int,
) -> np.ndarray:  # pragma: nocover
    """
    Tabulate :math:`\\log(2 w^2 s / h^2)` on rows of the distance from the
    depletion region and columns of :math:`\\log(w / \\delta)`,
    where :math:`h` is :func:`_level`.

    Parameters
    ----------
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    """
    result = np.empty((_num_rows, _num_v))
    for i in numba.prange(_num_rows):
        delta = _row_delta(i)
        h = _level(delta, kind)
        a, b = _coefficients(delta, kind)
        for j in range(_num_v):
            v = _v_min + (_v_max - _v_min) * j / (_num_v - 1)
            w = min(delta * math.exp(v), _w_max)
            s = _solve(delta, w, kind, a, b)
            result[i, j] = math.log(2 * w * w * s / (h * h))
    return result


@functools.cache
def _table(
    kind: int,
) -> np.ndarray:
    """
    The table of :func:`_build_table`, built once per process.

    Parameters
    ----------
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    """
    result = _build_table(kind)
    result.flags.writeable = False
    return result


@numba.njit(cache=True)
def _quantile(
    delta: float,
    w: float,
    kind: int,
    table: np.ndarray,
) -> float:  # pragma: nocover
    """
    The variance, in units of the square of the thickness of the field-free
    region, at the quantile :math:`p = \\text{erfc}(w)`.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    w
        The inverse complementary error function of the quantile.
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    table
        The table of `kind` from :func:`_table`.
    """
    if delta <= 0:
        return 0.0

    if w < _w_min:
        a, b = _coefficients(delta, kind)
        return _solve(delta, w, kind, a, b)

    fi = _row_coordinate(delta)
    i = min(int(fi), _num_rows - 2)
    weight_i = fi - i

    v = math.log(w) - math.log(delta)
    fj = (v - _v_min) / (_v_max - _v_min) * (_num_v - 1)
    fj = min(max(fj, 0.0), _num_v - 1.0)
    j = min(int(fj), _num_v - 2)
    weight_j = fj - j

    c = (1 - weight_i) * (
        (1 - weight_j) * table[i, j] + weight_j * table[i, j + 1]
    ) + weight_i * ((1 - weight_j) * table[i + 1, j] + weight_j * table[i + 1, j + 1])

    h = _level(delta, kind)
    return math.exp(c) * h * h / (2 * w * w)


@numba.njit(cache=True)
def _quantiles(
    delta: np.ndarray,
    w: np.ndarray,
    kind: int,
    table: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    :func:`_quantile` of each pair of elements of two arrays of the same size.

    Parameters
    ----------
    delta
        The distances from the depletion region.
    w
        The inverse complementary error functions of the quantiles.
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    table
        The table of `kind` from :func:`_table`.
    """
    result = np.empty(delta.size)
    for k in range(delta.size):
        result[k] = _quantile(delta[k], w[k], kind, table)
    return result


@numba.njit(cache=True)
def _sample(
    delta: float,
    table: np.ndarray,
) -> float:  # pragma: nocover
    """
    Draw the variance of the cloud of one electron, in units of the square of
    the thickness of the field-free region.

    Parameters
    ----------
    delta
        The distance from the depletion region at which the electron was
        created, in units of the thickness of the field-free region.
    table
        The table of :obj:`_kind_single` from :func:`_table`.
    """
    w = abs(random.gauss(0.0, 1.0)) / math.sqrt(2)
    return _quantile(delta, max(w, 1e-300), _kind_single, table)


@functools.cache
def _nodes() -> tuple[np.ndarray, np.ndarray]:
    r"""
    The nodes in :math:`w` and the weights of the quadrature over the
    quantiles,

    .. math::

        \left\langle f(s) \right\rangle = \int_0^\infty f\big(s(w)\big)
            \frac{2}{\sqrt{\pi}} e^{-w^2} dw,

    a trapezoidal rule in :math:`t`,
    where :math:`\log w = -1 + 2 \sinh t`,
    which places the nodes where the variance changes the most and decays
    doubly exponentially in both tails.
    The weights are normalized to sum to one.
    """
    t = np.linspace(-2.9, 1.3, _num_nodes)
    log_w = -1 + 2 * np.sinh(t)
    w = np.exp(log_w)
    weight = 2 * np.cosh(t) * 2 / np.sqrt(np.pi) * np.exp(-np.square(w)) * w
    weight = weight / weight.sum()
    return w, weight


def _transit(
    delta: float | na.AbstractScalar,
    kind: int,
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
    kind
        :obj:`_kind_single` or :obj:`_kind_pair`.
    axis
        The logical axis of the nodes.
    """
    nodes, weights = _nodes()
    w = na.ScalarArray(nodes, axes=axis)
    weight = na.ScalarArray(weights, axes=axis)

    shape = na.broadcast_shapes(na.shape(delta), w.shape)
    delta_nd = na.broadcast_to(delta, shape).ndarray
    w_nd = na.broadcast_to(w, shape).ndarray
    delta_nd = np.ascontiguousarray(delta_nd, dtype=float)
    w_nd = np.ascontiguousarray(w_nd, dtype=float)

    s = _quantiles(delta_nd.reshape(-1), w_nd.reshape(-1), kind, _table(kind))
    s = s.reshape(delta_nd.shape)

    return na.ScalarArray(s, axes=tuple(shape)), weight
