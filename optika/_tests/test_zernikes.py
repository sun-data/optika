import pytest
import numpy as np
import named_arrays as na
import optika

_position_random = na.Cartesian2dVectorArray(
    x=na.random.uniform(-0.7, 0.7, shape_random=dict(s=101), seed=42),
    y=na.random.uniform(-0.7, 0.7, shape_random=dict(s=101), seed=43),
)

_position_origin = na.Cartesian2dVectorArray(
    x=na.ScalarArray(np.array([0.0, 0.5]), axes="s"),
    y=na.ScalarArray(np.array([0.0, -0.5]), axes="s"),
)


@pytest.mark.parametrize(
    argnames="j,n,m",
    argvalues=[
        (1, 0, 0),
        (2, 1, 1),
        (3, 1, -1),
        (4, 2, 0),
        (5, 2, -2),
        (6, 2, 2),
        (7, 3, -1),
        (8, 3, 1),
        (9, 3, -3),
        (10, 3, 3),
        (11, 4, 0),
        (12, 4, 2),
        (13, 4, -2),
        (14, 4, 4),
        (15, 4, -4),
    ],
)
def test_noll(j: int, n: int, m: int):
    assert optika.zernikes.noll(j) == (n, m)


def test_noll_invalid():
    with pytest.raises(ValueError):
        optika.zernikes.noll(0)


@pytest.mark.parametrize(
    argnames="position",
    argvalues=[
        _position_random,
        _position_origin,
    ],
)
class TestClosedForms:

    def test_piston(self, position: na.AbstractCartesian2dVectorArray):
        result = optika.zernikes.zernike(position, 1)
        assert np.allclose(result, 1 + 0 * position.x)

    def test_tilt_x(self, position: na.AbstractCartesian2dVectorArray):
        result = optika.zernikes.zernike(position, 2)
        assert np.allclose(result, 2 * position.x)

    def test_tilt_y(self, position: na.AbstractCartesian2dVectorArray):
        result = optika.zernikes.zernike(position, 3)
        assert np.allclose(result, 2 * position.y)

    def test_defocus(self, position: na.AbstractCartesian2dVectorArray):
        result = optika.zernikes.zernike(position, 4)
        rho2 = np.square(position.length)
        assert np.allclose(result, np.sqrt(3) * (2 * rho2 - 1))

    def test_spherical(self, position: na.AbstractCartesian2dVectorArray):
        result = optika.zernikes.zernike(position, 11)
        rho2 = np.square(position.length)
        expected = np.sqrt(5) * (6 * np.square(rho2) - 6 * rho2 + 1)
        assert np.allclose(result, expected)


def test_orthonormality():
    position = na.Cartesian2dVectorStratifiedRandomSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("x", "y"),
        num=512,
        seed=42,
    ).explicit

    where = position.length <= 1
    num_inside = where.sum()

    js = range(1, 12)
    basis = {j: optika.zernikes.zernike(position, j) for j in js}

    for j1 in js:
        for j2 in js:
            if j2 < j1:
                continue
            inner = (basis[j1] * basis[j2]).sum(where=where) / num_inside
            expected = 1 if j1 == j2 else 0
            assert np.abs(inner - expected) < 0.03, (j1, j2)


@pytest.mark.parametrize("j", range(1, 16))
@pytest.mark.parametrize(
    argnames="position",
    argvalues=[
        _position_random,
        _position_origin,
    ],
)
def test_zernike_gradient(j: int, position: na.AbstractCartesian2dVectorArray):
    result = optika.zernikes.zernike_gradient(position, j)

    assert isinstance(result, na.AbstractCartesian2dVectorArray)

    h = 1e-6
    dx = na.Cartesian2dVectorArray(h, 0)
    dy = na.Cartesian2dVectorArray(0, h)

    derivative_x = optika.zernikes.zernike(position + dx, j)
    derivative_x = derivative_x - optika.zernikes.zernike(position - dx, j)
    derivative_x = derivative_x / (2 * h)

    derivative_y = optika.zernikes.zernike(position + dy, j)
    derivative_y = derivative_y - optika.zernikes.zernike(position - dy, j)
    derivative_y = derivative_y / (2 * h)

    assert np.allclose(result.x, derivative_x, atol=1e-5)
    assert np.allclose(result.y, derivative_y, atol=1e-5)
