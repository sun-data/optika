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

Two series give its probability density,
those of :cite:t:`Pavlov1999` for a reflecting back surface and no
recombination.
Expanding in the eigenfunctions of the field-free region gives

.. math::

    f(s) = \sum_{j=0}^\infty c_j \lambda_j e^{-\lambda_j s},
    \qquad
    c_j = \frac{4}{\pi k} \sin \frac{k \pi \delta}{2},
    \qquad
    \lambda_j = \frac{k^2 \pi^2}{8},

with :math:`k = 2 j + 1`, which converges quickly for large :math:`s`,
and the method of images gives

.. math::

    f(s) = \sum_{n=0}^\infty \frac{(-1)^n}{\sqrt{2 \pi s^3}} \left[
        a_n e^{-a_n^2 / 2 s} + b_n e^{-b_n^2 / 2 s}
    \right],
    \qquad
    a_n = 2 n + \delta,
    \qquad
    b_n = 2 n + 2 - \delta,

which converges quickly for small :math:`s`.

The charge cloud is the average of a Gaussian over this distribution,
Equation 66 of :cite:t:`Pavlov1999`, which the models take by Gauss
quadrature in :math:`\ln s` with :obj:`_num_nodes` nodes at each depth.
So the charge cloud at each depth is a mixture of :obj:`_num_nodes`
Gaussians, whose variances and weights both depend on the depth,
which the Monte Carlo simulation of
:func:`optika.sensors.electrons_measured` samples directly.

The Gauss rule at each depth is found by the Lanczos process on a grid in
:math:`\ln s` fine enough that the trapezoid rule on it is exact to
round-off for the moments the rule matches,
since the density in :math:`\ln s` falls off faster than exponentially at
both ends.
The rules are tabulated once against :math:`\delta`, with each node stored
as the width of its Gaussian over the distance, :math:`\sqrt{s} / \delta`,
which tends to a constant near the depletion region,
and with the reach of each Gaussian, the number of its own standard
deviations within which the window the Monte Carlo simulation samples it
over must reach, which depends on its weight,
so that looking up the mixture takes no transcendental functions.
"""

from typing import Callable
import math
import functools
import numba
import numpy as np
import named_arrays as na

__all__ = []

_num_nodes = 5
r"""
The number of nodes of the Gauss rule in :math:`\ln s`,
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
distance, and the weights are those at this distance.
"""

_num_delta = 2000
"""The number of rows of the table spaced uniformly in the logarithm of the distance."""

_num_rows = _num_u + _num_delta + 1
"""The number of rows of the table."""

_num_terms = 12
"""The number of terms kept of each series."""

_s_switch = 0.25
"""The variance below which the image series is used."""

_x_min = -6.0
r"""
The smallest :math:`\ln(s / \delta^2)` of the grid the Gauss rules are found
on, below which less than :math:`10^{-80}` of the distribution lies.
"""

_s_max = 40.0
"""
The largest variance of the grid the Gauss rules are found on, beyond which
less than :math:`10^{-20}` of the distribution lies.
"""

_step = 0.25
r"""The spacing in :math:`\ln s` of the grid the Gauss rules are found on."""

_mass_beyond_reach = 2e-9
"""
The most charge, as a fraction of the charge of the cloud, which a mixture
of Gaussians holds beyond the reach of :func:`_reach` of each of them,
on both sides together.
"""

_num_newton = 20
"""The most steps of Newton's method :func:`_reach` takes."""


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
def _density(
    s: float,
    delta: float,
) -> float:  # pragma: nocover
    """
    The probability density of the variance at `s`.

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
        return 0.0

    if s < _s_switch:
        total = 0.0
        for n in range(_num_terms):
            sign = 1.0 if n % 2 == 0 else -1.0
            a = 2 * n + delta
            b = 2 * n + 2 - delta
            total += sign * (
                a * math.exp(-a * a / (2 * s)) + b * math.exp(-b * b / (2 * s))
            )
        return total / math.sqrt(2 * math.pi * s**3)

    total = 0.0
    for j in range(_num_terms):
        rate = _rate(j)
        total += _weight(j, delta) * rate * math.exp(-rate * s)
    return total


@numba.njit(cache=True)
def _rule(
    delta: float,
    num: int,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: nocover
    r"""
    The nodes, as :math:`\ln(s / \delta^2)`, and the weights of the Gauss
    rule in :math:`\ln s` for the distribution of the variance.

    The rule is found by the Lanczos process with full reorthogonalization
    on a grid in :math:`\ln s` weighted by the trapezoid rule,
    which gives the Jacobi matrix of the orthogonal polynomials,
    whose eigenvalues are the nodes and the squares of the first components
    of whose eigenvectors are the weights.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region.
    num
        The number of nodes.
    """
    scale = delta * delta
    size = int((math.log(_s_max / scale) - _x_min) / _step) + 1

    x = np.empty(size)
    w = np.empty(size)
    for i in range(size):
        x[i] = _x_min + i * _step
        s = scale * math.exp(x[i])
        w[i] = max(s * _density(s, delta), 0.0)
    w = w / w.sum()

    basis = np.zeros((num, size))
    alpha = np.zeros(num)
    beta = np.zeros(num)
    basis[0] = np.sqrt(w)
    for k in range(num):
        z = x * basis[k]
        alpha[k] = np.dot(basis[k], z)
        for m in range(k + 1):
            z = z - np.dot(basis[m], z) * basis[m]
        if k + 1 < num:
            beta[k] = math.sqrt(np.dot(z, z))
            basis[k + 1] = z / beta[k]

    jacobi = np.diag(alpha)
    for k in range(num - 1):
        jacobi[k, k + 1] = beta[k]
        jacobi[k + 1, k] = beta[k]

    nodes, vectors = np.linalg.eigh(jacobi)
    return nodes, vectors[0] ** 2


@numba.njit(cache=True)
def _reach(
    weight: float,
    num: int,
) -> float:  # pragma: nocover
    r"""
    The number of its own standard deviations beyond which one of `num`
    Gaussians of a mixture holds :obj:`_mass_beyond_reach` over `num` of the
    charge of the mixture, on both sides together,
    the root :math:`n` of :math:`w \operatorname{erfc}(n / \sqrt{2}) = \epsilon / N`,
    or zero if the Gaussian holds less than that.

    Newton's method on the logarithm of the complementary error function,
    which is concave, approaches the root from above,
    starting from the root of the bound
    :math:`\operatorname{erfc}(x) \leq e^{-x^2}`,
    so every step reaches at least as far as the root.

    Parameters
    ----------
    weight
        The weight of the Gaussian.
    num
        The number of Gaussians of the mixture.
    """
    if num * weight <= _mass_beyond_reach:
        return 0.0
    log_target = math.log(_mass_beyond_reach / (num * weight))
    n = math.sqrt(-2 * log_target)
    for _ in range(_num_newton):
        t = n / math.sqrt(2)
        tail = math.erfc(t)
        slope = -math.sqrt(2 / math.pi) * math.exp(-t * t) / tail
        step = (math.log(tail) - log_target) / slope
        n = n - step
        if abs(step) < 1e-12 * n:
            break
    return n


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
    num: int,
) -> np.ndarray:  # pragma: nocover
    """
    Tabulate the Gauss rule of :func:`_rule` on rows of the distance from
    the depletion region: the widths :math:`\\sqrt{s} / \\delta` of the
    nodes along index 0 of the second axis, their weights along index 1,
    and the reach of :func:`_reach` of their Gaussians along index 2.

    Parameters
    ----------
    num
        The number of nodes of each rule.
    """
    result = np.empty((_num_rows, 3, num))
    for i in numba.prange(_num_rows):
        nodes, weights = _rule(_row_delta(i), num)
        result[i, 0] = np.exp(nodes / 2)
        result[i, 1] = weights
        for k in range(num):
            result[i, 2, k] = _reach(weights[k], num)
    return result


@functools.cache
def _table() -> np.ndarray:
    """The table of :func:`_build_table` with :obj:`_num_nodes` nodes, built once per process."""
    result = _build_table(_num_nodes)
    result.flags.writeable = False
    return result


@numba.njit(cache=True)
def _mixture(
    delta: float,
    table: np.ndarray,
    width: np.ndarray,
    weight: np.ndarray,
    reach: np.ndarray,
) -> None:  # pragma: nocover
    """
    The width, the weight and the reach of each Gaussian of the mixture,
    the width, the square root of its variance, in units of the thickness of
    the field-free region, for charge created a distance `delta` from the
    depletion region.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero or less in the depletion region.
    table
        The table from :func:`_table`.
    width
        The array of the width of each Gaussian, filled in place.
    weight
        The array of the weight of each Gaussian, filled in place.
    reach
        The array of the reach of :func:`_reach` of each Gaussian,
        filled in place.
    """
    if delta <= 0:
        width[:] = 0
        weight[:] = table[_num_rows - 1, 1]
        reach[:] = table[_num_rows - 1, 2]
        return

    fi = _row_coordinate(delta)
    i = min(int(fi), _num_rows - 2)
    t = fi - i

    for k in range(width.size):
        width[k] = delta * ((1 - t) * table[i, 0, k] + t * table[i + 1, 0, k])
        weight[k] = (1 - t) * table[i, 1, k] + t * table[i + 1, 1, k]
        reach[k] = (1 - t) * table[i, 2, k] + t * table[i + 1, 2, k]


@numba.njit(cache=True)
def _mixture_array(
    delta: np.ndarray,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: nocover
    """
    The variance and the weight of each Gaussian of :func:`_mixture`
    for each element of an array of distances.

    Parameters
    ----------
    delta
        The distances from the depletion region.
    table
        The table from :func:`_table`.
    """
    num = table.shape[2]
    width = np.empty(num)
    reach = np.empty(num)
    variance = np.empty((delta.size, num))
    weight = np.empty((delta.size, num))
    for i in range(delta.size):
        _mixture(delta[i], table, width, weight[i], reach)
        for k in range(num):
            variance[i, k] = width[k] * width[k]
    return variance, weight


@numba.njit(cache=True)
def _pairs_array(
    delta: np.ndarray,
    table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: nocover
    """
    For each pair of Gaussians of :func:`_mixture`, in the order of
    :func:`numpy.triu_indices`, the sum of their variances and the weight of
    the pair, the product of their weights, doubled for different Gaussians,
    for each element of an array of distances.

    Parameters
    ----------
    delta
        The distances from the depletion region.
    table
        The table from :func:`_table`.
    """
    num = table.shape[2]
    num_pairs = num * (num + 1) // 2
    width = np.empty(num)
    weight = np.empty(num)
    reach = np.empty(num)
    variance_pair = np.empty((delta.size, num_pairs))
    weight_pair = np.empty((delta.size, num_pairs))
    for e in range(delta.size):
        _mixture(delta[e], table, width, weight, reach)
        p = 0
        for i in range(num):
            for j in range(i, num):
                variance_pair[e, p] = width[i] * width[i] + width[j] * width[j]
                factor = 1.0 if i == j else 2.0
                weight_pair[e, p] = factor * weight[i] * weight[j]
                p += 1
    return variance_pair, weight_pair


def _lookup(
    function: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    delta: float | na.AbstractScalar,
    axis: str,
) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    """
    Evaluate :func:`_mixture_array` or :func:`_pairs_array` for a named array
    of distances, with the Gaussians or their pairs along `axis`.

    Parameters
    ----------
    function
        The compiled function to evaluate.
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero in the depletion region.
    axis
        The logical axis of the Gaussians or of their pairs.
    """
    delta = na.as_named_array(delta)

    # the mixture at the nominal distance and at each sample of its
    # distribution, since the table is looked up element by element
    if isinstance(delta, na.AbstractUncertainScalarArray):
        s_nominal, weight_nominal = _lookup(function, delta.nominal, axis)
        s_distribution, weight_distribution = _lookup(
            function, delta.distribution, axis
        )
        return (
            na.UncertainScalarArray(s_nominal, s_distribution),
            na.UncertainScalarArray(weight_nominal, weight_distribution),
        )

    delta = delta.explicit
    delta_nd = np.asarray(delta.ndarray, dtype=float)

    s, weight = function(delta_nd.reshape(-1), _table())
    shape = delta_nd.shape + (s.shape[~0],)
    axes = delta.axes + (axis,)

    return (
        na.ScalarArray(s.reshape(shape), axes=axes),
        na.ScalarArray(weight.reshape(shape), axes=axes),
    )


def _transit(
    delta: float | na.AbstractScalar,
    axis: str,
) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    """
    The variance of each Gaussian of the mixture,
    in units of the square of the thickness of the field-free region,
    and its weight.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero in the depletion region.
    axis
        The logical axis of the Gaussians.
    """
    return _lookup(_mixture_array, delta, axis)


def _transit_pairs(
    delta: float | na.AbstractScalar,
    axis: str,
) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    """
    The variance of the separation of two electrons drawn from each pair of
    Gaussians of the mixture,
    in units of the square of the thickness of the field-free region,
    and the weight of the pair, as in :func:`_pairs_array`.

    Parameters
    ----------
    delta
        The distance from the depletion region in units of the thickness of
        the field-free region, zero in the depletion region.
    axis
        The logical axis of the pairs of Gaussians.
    """
    return _lookup(_pairs_array, delta, axis)
