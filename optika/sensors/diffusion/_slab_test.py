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


def _grid(delta: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    A grid in :math:`\ln(s / \delta^2)` finer than that of the module,
    and the density of the variance on it, normalized by the trapezoid rule.
    """
    x = np.arange(_slab._x_min, math.log(_slab._s_max / delta**2), _slab._step / 4)
    s = delta**2 * np.exp(x)
    w = np.array([si * _slab._density(si, delta) for si in s])
    return x, w / w.sum()


def test_series() -> None:
    """
    The image series and the eigenfunction series of the density agree
    up to where one takes over from the other, beyond which the image series
    loses precision close to the depletion region,
    and the density integrates to one.
    """
    switch = _slab._s_switch
    for delta in [1, 0.5, 0.1, 1e-3, 1e-6]:
        for s in [switch / 4, switch / 2, switch]:
            image = 0.0
            for n in range(_slab._num_terms):
                sign = 1 if n % 2 == 0 else -1
                for a in [2 * n + delta, 2 * n + 2 - delta]:
                    image += sign * a * math.exp(-(a**2) / (2 * s))
            image = image / math.sqrt(2 * math.pi * s**3)
            eigen = 0.0
            for j in range(_slab._num_terms):
                rate = _slab._rate(j)
                eigen += _slab._weight(j, delta) * rate * math.exp(-rate * s)
            assert math.isclose(image, eigen, rel_tol=1e-12, abs_tol=1e-300)

        x = np.arange(_slab._x_min, math.log(_slab._s_max / delta**2), 0.05)
        s = delta**2 * np.exp(x)
        total = 0.05 * sum(si * _slab._density(si, delta) for si in s)
        assert math.isclose(total, 1, rel_tol=1e-12)

    assert _slab._density(0, 0.5) == 0


def test_rule() -> None:
    r"""
    The Gauss rule matches the moments of :math:`\ln s` up to twice its
    number of nodes, and its weights are positive and sum to one.
    """
    num = _slab._num_nodes
    for delta in [1, 0.5, 0.1, 1e-3, 1e-6]:
        nodes, weights = _slab._rule(delta, num)
        assert nodes.shape == weights.shape == (num,)
        assert np.all(np.diff(nodes) > 0)
        assert np.all(weights > 0)
        assert math.isclose(weights.sum(), 1, rel_tol=1e-13)

        x, w = _grid(delta)
        center = (w * x).sum()
        scale = np.sqrt((w * np.square(x - center)).sum())
        for m in range(2 * num):
            expected = (w * ((x - center) / scale) ** m).sum()
            result = (weights * ((nodes - center) / scale) ** m).sum()
            assert math.isclose(result, expected, rel_tol=1e-8, abs_tol=1e-8)


def test_reach() -> None:
    """
    Each Gaussian holds its share of the charge allowed beyond the window
    beyond its reach, and a Gaussian holding less than that reaches nowhere.
    """
    epsilon = _slab._mass_beyond_reach
    for num in [1, 8]:
        for weight in [1, 0.3, 1e-3, 1e-8, 2 * epsilon / num]:
            n = _slab._reach(weight, num)
            assert n > 0
            tail = weight * math.erfc(n / math.sqrt(2))
            assert math.isclose(tail, epsilon / num, rel_tol=1e-10)
        assert _slab._reach(epsilon / num, num) == 0
        assert _slab._reach(0, num) == 0


def test_table() -> None:
    """
    The table interpolates the widths, the weights and the reaches of the
    Gauss rule to a few parts in a million.
    """
    table = _slab._table()
    num = _slab._num_nodes
    assert table.shape == (_slab._num_rows, 3, num)
    assert np.all(np.isfinite(table))
    assert not table.flags.writeable

    width = np.empty(num)
    weight = np.empty(num)
    reach = np.empty(num)
    for delta in _deltas(400, seed=0):
        _slab._mixture(delta, table, width, weight, reach)
        nodes, weights = _slab._rule(max(delta, _slab._delta_min), num)
        expected = delta * np.exp(nodes / 2)
        assert np.allclose(width, expected, rtol=1e-5, atol=0)
        assert np.allclose(weight, weights, rtol=0, atol=1e-6)
        assert math.isclose(weight.sum(), 1, rel_tol=1e-12)
        expected = [_slab._reach(w, num) for w in weights]
        assert np.allclose(reach, expected, rtol=1e-5, atol=0)


def test_mixture() -> None:
    """
    Charge created in the depletion region does not cross the field-free
    one, and closer to the depletion region than the table reaches, the
    widths scale with the distance and the weights and the reaches stay the
    same.
    """
    table = _slab._table()
    num = _slab._num_nodes
    width = np.empty(num)
    weight = np.empty(num)
    reach = np.empty(num)
    for delta in [0, -0.5]:
        width[:] = 1
        _slab._mixture(delta, table, width, weight, reach)
        assert np.all(width == 0)
        assert math.isclose(weight.sum(), 1, rel_tol=1e-12)
        assert np.all(reach > 0)

    near = np.empty(num)
    nearer = np.empty(num)
    weight_near = np.empty(num)
    weight_nearer = np.empty(num)
    reach_near = np.empty(num)
    reach_nearer = np.empty(num)
    _slab._mixture(_slab._delta_min, table, near, weight_near, reach_near)
    _slab._mixture(_slab._delta_min / 10, table, nearer, weight_nearer, reach_nearer)
    assert np.allclose(nearer, near / 10, rtol=1e-12)
    assert np.all(weight_nearer == weight_near)
    assert np.all(reach_nearer == reach_near)


def test_transit() -> None:
    """
    The weights of the mixture average the variance to its mean,
    :math:`1 - u^2`, to a few parts in ten thousand at least a tenth of the
    field-free region from the depletion region.
    """
    u = na.linspace(0, 0.9, axis="depth", num=7)
    s, weight = _slab._transit(1 - u, axis="node")
    assert s.shape == weight.shape == dict(depth=7, node=_slab._num_nodes)
    assert np.allclose(weight.sum("node"), 1, rtol=1e-12)
    mean = (weight * s).sum("node")
    assert np.allclose(mean, 1 - np.square(u), rtol=3e-4)

    # charge created in the depletion region does not cross the field-free one
    s, weight = _slab._transit(0, axis="node")
    assert s.shape == weight.shape == dict(node=_slab._num_nodes)
    assert np.all(s == 0)
