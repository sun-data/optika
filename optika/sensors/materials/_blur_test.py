from typing import Callable
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from ._blur import _blur, _axis_interpolation

_num_w = 2
_num_x = 7
_num_y = 5
_num_kernel_x = 5
_num_kernel_y = 3


def _matrix(
    kernel: Callable[[int, int, int], np.ndarray],
    wrap: bool,
) -> np.ndarray:
    """
    The spread as an explicit matrix for each wavelength,
    whose column for each pixel holds the kernel of that pixel, centered on
    it.

    Parameters
    ----------
    kernel
        The kernel of each pixel, given the wavelength and the pixel.
    wrap
        Whether the grid is periodic.
    """
    num = _num_x * _num_y
    result = np.zeros((_num_w, num, num))
    for w in range(_num_w):
        for jx in range(_num_x):
            for jy in range(_num_y):
                k = kernel(w, jx, jy)
                for a in range(_num_kernel_x):
                    for b in range(_num_kernel_y):
                        ix = jx + a - _num_kernel_x // 2
                        iy = jy + b - _num_kernel_y // 2
                        if wrap:
                            ix, iy = ix % _num_x, iy % _num_y
                        elif not (0 <= ix < _num_x and 0 <= iy < _num_y):
                            continue
                        result[w, ix * _num_y + iy, jx * _num_y + jy] += k[a, b]
    return result


_rng = np.random.default_rng(0)
_values = _rng.random((_num_w, _num_x, _num_y))
_uniform = _rng.random((_num_w, _num_kernel_x, _num_kernel_y))
_per_pixel = _rng.random((_num_w, _num_x, _num_y, _num_kernel_x, _num_kernel_y))
_table = _rng.random((4, _num_w, _num_kernel_x, _num_kernel_y))
_position = _rng.random((_num_x, _num_y))
_nodes = np.array([0, 0.3, 0.5, 1])


def _same(w: int, jx: int, jy: int) -> np.ndarray:
    """The kernel every pixel shares."""
    return _uniform[w]


def _own(w: int, jx: int, jy: int) -> np.ndarray:
    """The kernel of each pixel."""
    return _per_pixel[w, jx, jy]


def _interpolated(w: int, jx: int, jy: int) -> np.ndarray:
    """The kernel of a pixel interpolated between the nodes of the table."""
    result = np.empty((_num_kernel_x, _num_kernel_y))
    for a in range(_num_kernel_x):
        for b in range(_num_kernel_y):
            result[a, b] = np.interp(_position[jx, jy], _nodes, _table[:, w, a, b])
    return result


_kernels = dict(
    uniform=(
        na.ScalarArray(_uniform, axes=("w", "kx", "ky")),
        _same,
        None,
    ),
    per_pixel=(
        na.ScalarArray(_per_pixel, axes=("w", "x", "y", "kx", "ky")),
        _own,
        None,
    ),
    interpolated=(
        na.ScalarArray(_table, axes=(_axis_interpolation, "w", "kx", "ky")),
        _interpolated,
        (
            na.ScalarArray(_position, axes=("x", "y")),
            na.ScalarArray(_nodes, axes=_axis_interpolation),
        ),
    ),
)


@pytest.mark.parametrize("kind", list(_kernels))
@pytest.mark.parametrize("wrap", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_blur(
    kind: str,
    wrap: bool,
    transpose: bool,
) -> None:
    """
    The spread, and its transpose, is the product with the explicit matrix
    whose column for each pixel is its kernel,
    whether every pixel has the same kernel, its own, or one interpolated
    between nodes.
    """
    kernel, function, interpolation = _kernels[kind]
    values = na.ScalarArray(_values, axes=("w", "x", "y"))

    result = _blur(
        values=values,
        kernel=kernel,
        axis_xy=("x", "y"),
        axis_kernel=("kx", "ky"),
        wrap=wrap,
        transpose=transpose,
        interpolation=interpolation,
    )

    matrix = _matrix(function, wrap)
    if transpose:
        matrix = np.swapaxes(matrix, -1, -2)
    expected = matrix @ _values.reshape(_num_w, -1, 1)
    expected = expected.reshape(_num_w, _num_x, _num_y)

    assert isinstance(result, na.ScalarArray)
    assert np.allclose(result.ndarray_aligned(("w", "x", "y")), expected, rtol=1e-14)


def test_blur_units() -> None:
    """
    The result has the units of the values times the units of the kernel,
    and the nodes of the interpolation are converted to the units of the
    positions.
    """
    values = na.ScalarArray(_values * u.photon, axes=("w", "x", "y"))
    kernel = na.ScalarArray(
        _table * u.electron / u.photon,
        axes=(_axis_interpolation, "w", "kx", "ky"),
    )
    position = na.ScalarArray(_position * u.mm, axes=("x", "y"))
    nodes = na.ScalarArray(_nodes * 1000 * u.um, axes=_axis_interpolation)

    result = _blur(
        values=values,
        kernel=kernel,
        axis_xy=("x", "y"),
        axis_kernel=("kx", "ky"),
        wrap=False,
        transpose=False,
        interpolation=(position, nodes),
    )
    expected = _blur(
        values=values.value,
        kernel=kernel.value,
        axis_xy=("x", "y"),
        axis_kernel=("kx", "ky"),
        wrap=False,
        transpose=False,
        interpolation=(position.value, nodes.to(u.mm).value),
    )

    assert result.unit.is_equivalent(u.electron)
    assert np.allclose(result.to(u.electron).value, expected)


def test_blur_requires_grid() -> None:
    """The values must vary along both axes of the pixel grid."""
    with pytest.raises(ValueError, match="both axes"):
        _blur(
            values=na.ScalarArray(_values[:, :, 0], axes=("w", "x")),
            kernel=_kernels["uniform"][0],
            axis_xy=("x", "y"),
            axis_kernel=("kx", "ky"),
            wrap=False,
            transpose=False,
        )
