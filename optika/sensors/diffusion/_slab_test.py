import math
import pytest
import numba
import numpy as np
import scipy.special
import named_arrays as na
from . import _slab

_kinds = [_slab._kind_single, _slab._kind_pair]


def _delta_w(num: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Random distances from the depletion region, half uniform and half spread
    over the decades below one, and random quantiles drawn as the Monte Carlo
    simulation draws them.
    """
    rng = np.random.default_rng(seed)
    delta = np.concatenate(
        [
            rng.uniform(0, 1, num // 2),
            10 ** rng.uniform(-8, 0, num - num // 2),
        ]
    )
    w = np.abs(rng.normal(size=num)) / math.sqrt(2)
    return delta, w


@pytest.mark.parametrize("kind", _kinds)
def test_series(kind: int):
    """
    The image series and the eigenfunction series of the distribution agree
    where one takes over from the other, and the distribution rises from
    zero to one.
    """
    switch = (
        _slab._s_switch_single if kind == _slab._kind_single else _slab._s_switch_pair
    )
    for delta in [1, 0.5, 0.1, 1e-3, 1e-6]:
        a, b = _slab._coefficients(delta, kind)
        for s in [switch / 2, switch, 2 * switch]:
            r = math.sqrt(2 * s)
            image = 0.0
            for n in range(_slab._num_terms):
                sign = 1 if n % 2 == 0 else -1
                if kind == _slab._kind_single:
                    image += sign * (
                        math.erfc((2 * n + delta) / r)
                        + math.erfc((2 * n + 2 - delta) / r)
                    )
                else:
                    image += (
                        sign
                        * (n + 1)
                        * (
                            math.erfc((2 * n + 2 * delta) / r)
                            + 2 * math.erfc((2 * n + 2) / r)
                            + math.erfc((2 * n + 4 - 2 * delta) / r)
                        )
                    )
            eigen = 0.0
            for j in range(_slab._num_terms):
                eigen += (a[j] + b[j] * s) * math.exp(-_slab._rate(j) * s)
            assert math.isclose(image + eigen, 1, abs_tol=1e-12)

        cdf, _ = _slab._cdf_survival(1e-6, delta, kind, a, b)
        assert cdf <= 1e-10 or delta < 1e-2
        cdf, survival = _slab._cdf_survival(100, delta, kind, a, b)
        assert survival < 1e-20


@pytest.mark.parametrize("kind", _kinds)
def test_table(kind: int):
    """The table interpolates the quantiles to a few parts in a hundred thousand."""
    table = _slab._table(kind)
    assert table.shape == (_slab._num_rows, _slab._num_v)
    assert np.all(np.isfinite(table))
    assert not table.flags.writeable

    delta, w = _delta_w(1000, seed=kind)
    for d, x in zip(delta, w):
        a, b = _slab._coefficients(d, kind)
        expected = _slab._solve(d, x, kind, a, b)
        result = _slab._quantile(d, x, kind, table)
        assert math.isclose(result, expected, rel_tol=1e-4)

        # the quantile is the one asked for
        cdf, survival = _slab._cdf_survival(expected, d, kind, a, b)
        assert math.isclose(cdf, math.erfc(x), rel_tol=1e-9, abs_tol=1e-15)

    # beyond the table, the quantile is found directly
    assert _slab._quantile(0.5, 1e-12, kind, table) > _slab._quantile(
        0.5, 1e-9, kind, table
    )
    assert _slab._quantile(0, 0.5, kind, table) == 0


def test_pair():
    """
    The sum of the variances of two electrons is distributed as the sum of
    two independent draws of the variance of one.
    """
    num = 400000
    rng = np.random.default_rng(1)
    table = _slab._table(_slab._kind_single)
    table_pair = _slab._table(_slab._kind_pair)
    for delta in [1, 0.5, 0.1, 0.01]:
        w = np.abs(rng.normal(size=(2, num))) / math.sqrt(2)
        d = np.full(num, delta)
        s = _slab._quantiles(d, w[0], _slab._kind_single, table)
        s = s + _slab._quantiles(d, w[1], _slab._kind_single, table)
        for p in [0.1, 0.5, 0.9, 0.99]:
            x = scipy.special.erfcinv(p)
            t = _slab._quantile(delta, x, _slab._kind_pair, table_pair)
            assert abs(np.mean(s <= t) - p) < 5 * math.sqrt(p * (1 - p) / num)


def test_nodes():
    """The quadrature weights are positive and sum to one."""
    w, weight = _slab._nodes()
    assert w.shape == weight.shape == (_slab._num_nodes,)
    assert np.all(w > _slab._w_min)
    assert np.all(weight > 0)
    assert math.isclose(weight.sum(), 1, rel_tol=1e-15)


@pytest.mark.parametrize("kind", _kinds)
def test_transit(kind: int):
    """
    The quadrature averages the variance to its mean,
    :math:`1 - u^2` for one electron and twice that for two,
    away from the depletion region, where the mean is set by the rare long
    excursions of the charge.
    """
    u = na.linspace(0, 0.9, axis="depth", num=10)
    s, weight = _slab._transit(1 - u, kind, axis="node")
    assert s.shape == dict(depth=10, node=_slab._num_nodes)
    mean = (weight * s).sum("node")
    factor = 1 if kind == _slab._kind_single else 2
    assert np.allclose(mean, factor * (1 - np.square(u)), rtol=1e-4)

    # charge created in the depletion region does not cross the field-free one
    s, weight = _slab._transit(0, kind, axis="node")
    assert np.all(s == 0)


def test_sample():
    """The Monte Carlo draws are distributed as the tabulated quantiles."""

    @numba.njit
    def draw(  # pragma: nocover
        delta: float,
        num: int,
        table: np.ndarray,
    ) -> np.ndarray:
        result = np.empty(num)
        for i in range(num):
            result[i] = _slab._sample(delta, table)
        return result

    num = 200000
    table = _slab._table(_slab._kind_single)
    for delta in [1, 0.3, 1e-3]:
        s = draw(delta, num, table)
        for p in [0.05, 0.5, 0.95]:
            x = scipy.special.erfcinv(p)
            q = _slab._quantile(delta, x, _slab._kind_single, table)
            assert abs(np.mean(s <= q) - p) < 5 * math.sqrt(p * (1 - p) / num)
