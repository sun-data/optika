import math
import numpy as np
import named_arrays as na
from . import _slab


def _deltas(num: int, seed: int) -> np.ndarray:
    """
    Random distances from the depletion region, half uniform and half spread
    over the decades below one.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.uniform(0, 1, num // 2),
            10 ** rng.uniform(-8, 0, num - num // 2),
        ]
    )


def test_series():
    """
    The image series and the eigenfunction series of the distribution agree
    where one takes over from the other, and the distribution rises from
    zero to one.
    """
    switch = _slab._s_switch
    for delta in [1, 0.5, 0.1, 1e-3, 1e-6]:
        for s in [switch / 2, switch, 2 * switch]:
            r = math.sqrt(2 * s)
            image = 0.0
            for n in range(_slab._num_terms):
                sign = 1 if n % 2 == 0 else -1
                image += sign * (
                    math.erfc((2 * n + delta) / r) + math.erfc((2 * n + 2 - delta) / r)
                )
            eigen = 0.0
            for j in range(_slab._num_terms):
                eigen += _slab._weight(j, delta) * math.exp(-_slab._rate(j) * s)
            assert math.isclose(image + eigen, 1, abs_tol=1e-12)

        cdf, _ = _slab._cdf_survival(1e-6, delta)
        assert cdf <= 1e-10 or delta < 1e-2
        _, survival = _slab._cdf_survival(100, delta)
        assert survival < 1e-20


def test_nodes():
    """
    The quantiles of the nodes lie inside the distribution and pair up
    symmetrically, and the weights are positive and sum to one.
    """
    p, q, weight = _slab._nodes()
    assert p.shape == q.shape == weight.shape == (_slab._num_nodes,)
    assert np.all((0 < p) & (p < 1))
    assert np.all(np.diff(p) > 0)
    assert np.allclose(p + q, 1, rtol=0, atol=1e-15)
    assert np.allclose(p, q[::-1], rtol=1e-12)
    assert np.all(weight > 0)
    assert np.allclose(weight, weight[::-1], rtol=1e-12)
    assert math.isclose(weight.sum(), 1, rel_tol=1e-15)


def test_table():
    """
    The table interpolates the variance at each node to a few parts in a
    million, and the variance is the quantile asked for.
    """
    table = _slab._table()
    assert table.shape == (_slab._num_rows, _slab._num_nodes)
    assert np.all(np.isfinite(table))
    assert not table.flags.writeable

    p, q, _ = _slab._nodes()
    result = np.empty(_slab._num_nodes)
    for delta in _deltas(400, seed=0):
        _slab._variances(delta, table, result)
        for k in range(_slab._num_nodes):
            expected = _slab._solve(delta, p[k], q[k])
            assert math.isclose(result[k], expected, rel_tol=1e-5)

            cdf, survival = _slab._cdf_survival(expected, delta)
            if p[k] < 0.5:
                assert math.isclose(cdf, p[k], rel_tol=1e-9)
            else:
                assert math.isclose(survival, q[k], rel_tol=1e-9)

        # the variance grows with the quantile
        assert np.all(np.diff(result) > 0)


def test_variances():
    """
    Charge created in the depletion region does not cross the field-free
    one, and closer to the depletion region than the table reaches, the
    variance scales with the square of the distance.
    """
    table = _slab._table()
    result = np.empty(_slab._num_nodes)
    for delta in [0, -0.5]:
        result[:] = 1
        _slab._variances(delta, table, result)
        assert np.all(result == 0)

    near = np.empty(_slab._num_nodes)
    nearer = np.empty(_slab._num_nodes)
    _slab._variances(_slab._delta_min, table, near)
    _slab._variances(_slab._delta_min / 10, table, nearer)
    assert np.allclose(nearer, near / 100, rtol=1e-12)


def test_transit():
    """
    The weights of the mixture average the variance to its mean,
    :math:`1 - u^2`, except near the depletion region,
    where the mean is set by the rare long excursions of the charge.
    """
    u = na.linspace(0, 0.5, axis="depth", num=6)
    s, weight = _slab._transit(1 - u, axis="node")
    assert s.shape == dict(depth=6, node=_slab._num_nodes)
    assert weight.shape == dict(node=_slab._num_nodes)
    mean = (weight * s).sum("node")
    assert np.allclose(mean, 1 - np.square(u), rtol=1e-3)

    # charge created in the depletion region does not cross the field-free one
    s, weight = _slab._transit(0, axis="node")
    assert s.shape == dict(node=_slab._num_nodes)
    assert np.all(s == 0)
