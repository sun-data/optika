"""
Spreading the values on a pixel grid with a kernel centered on each pixel,
and the transpose of that spread.
"""

import numpy as np
import named_arrays as na

__all__ = []

_axis_interpolation = "_blur_interpolation"
"""The logical axis of the nodes a kernel is interpolated between."""


def _slices(
    offset: int,
    num: int,
) -> None | tuple[slice, slice]:
    """
    The pixels along one axis of a grid which a kernel moves charge between,
    if it moves charge `offset` pixels and the charge which leaves the grid is
    lost: the pixels it arrives in and the pixels it leaves,
    or :obj:`None` if it leaves the grid entirely.

    Parameters
    ----------
    offset
        The number of pixels the charge moves.
    num
        The number of pixels along the axis.
    """
    if abs(offset) >= num:
        return None
    target = slice(max(offset, 0), num + min(offset, 0))
    source = slice(max(-offset, 0), num - max(offset, 0))
    return target, source


def _window(
    array: np.ndarray,
    index: tuple[int, int],
    slices: tuple[slice, slice],
) -> np.ndarray:
    """
    The pixels of an array within a window of the pixel grid,
    along the dimensions of the grid which the array varies along.

    Parameters
    ----------
    array
        An array whose dimensions at `index` are the axes of the pixel grid,
        or are of size one if the array does not vary along them.
    index
        The dimensions of the axes of the pixel grid.
    slices
        The window along each axis of the pixel grid.
    """
    key = [slice(None)] * array.ndim
    for i, s in zip(index, slices):
        if array.shape[i] > 1:
            key[i] = s
    return array[tuple(key)]


def _blur(
    values: na.AbstractScalar,
    kernel: na.AbstractScalar,
    axis_xy: tuple[str, str],
    axis_kernel: tuple[str, str],
    wrap: bool,
    transpose: bool,
    interpolation: None | tuple[na.AbstractScalar, na.AbstractScalar] = None,
) -> na.ScalarArray:
    r"""
    Spread the values on a pixel grid with a kernel centered on each pixel,
    or apply the transpose of that spread.

    The kernel may differ from pixel to pixel,
    so each value is spread with the kernel of the pixel it starts in,

    .. math::

        y_i = \sum_k K_k(i - k) \, x_{i - k},

    and the transpose collects each value with the kernel of the pixel it
    ends in,

    .. math::

        y_j = \sum_k K_k(j) \, x_{j + k},

    where :math:`k` runs over the pixels of the kernel relative to its center.
    Both are sums in a fixed order over the pixels of the kernel,
    so the result does not depend on the order of any floating-point
    operations which could change from run to run.

    Parameters
    ----------
    values
        The values on the pixel grid,
        which must vary along both axes of `axis_xy`.
    kernel
        The fraction of each value which lands in the pixel it starts in and
        in each of the pixels around it,
        along the two axes of `axis_kernel`,
        which must have an odd number of pixels each.
        If it varies along the axes of `axis_xy`, each pixel has its own
        kernel.
    axis_xy
        The two logical axes of the pixel grid.
    axis_kernel
        The two logical axes of the kernel corresponding to each axis of the
        pixel grid.
    wrap
        If :obj:`False`, the values which leave the edge of the grid are lost.
        If :obj:`True`, the grid is periodic, and they re-enter the opposite
        edge.
    transpose
        Whether to apply the transpose of the spread instead.
    interpolation
        If given, a pair of arrays ``(x, nodes)``:
        the kernel varies along :obj:`_axis_interpolation` between `nodes`,
        and the kernel of each pixel is interpolated linearly at `x`,
        one pixel of the kernel at a time, so that the kernels of all the
        pixels are never held at once.
    """
    shape_values = na.shape(values)
    for axis in axis_xy:
        if axis not in shape_values:
            raise ValueError(
                f"`values` must vary along both axes of the pixel grid, {axis_xy}, "
                f"got {shape_values}."
            )

    shape_kernel = dict(na.shape(kernel))
    num_kernel = tuple(shape_kernel.pop(axis) for axis in axis_kernel)
    if interpolation is not None:
        shape_kernel.pop(_axis_interpolation)
        shape_kernel = na.broadcast_shapes(shape_kernel, na.shape(interpolation[0]))

    shape = na.broadcast_shapes(shape_values, shape_kernel)
    axes = tuple(shape)
    index = tuple(axes.index(axis) for axis in axis_xy)
    num = tuple(shape[axis] for axis in axis_xy)

    x = na.as_named_array(values).explicit.ndarray_aligned(axes)
    result = 0

    for i_x in range(num_kernel[0]):
        for i_y in range(num_kernel[1]):

            k = kernel[{axis_kernel[0]: i_x, axis_kernel[1]: i_y}]
            if interpolation is not None:
                position, nodes = interpolation
                k = na.interp(position, nodes, k, axis=_axis_interpolation)
            k = na.as_named_array(k).explicit.ndarray_aligned(axes)

            offset = (i_x - num_kernel[0] // 2, i_y - num_kernel[1] // 2)

            if wrap:
                if not transpose:
                    term = np.roll(k * x, offset, axis=index)
                else:
                    term = k * np.roll(x, tuple(-o for o in offset), axis=index)
                result = result + term
                continue

            pair_x = _slices(offset[0], num[0])
            pair_y = _slices(offset[1], num[1])
            if pair_x is None or pair_y is None:
                continue

            # the forward spread moves the values from the source window of
            # the grid to the target window, and the transpose moves them back
            target = (pair_x[0], pair_y[0])
            source = (pair_x[1], pair_y[1])
            if not transpose:
                term = _window(k, index, source) * _window(x, index, source)
                window = target
            else:
                term = _window(k, index, source) * _window(x, index, target)
                window = source

            if isinstance(result, int):
                result = np.zeros_like(term, shape=tuple(shape.values()))
            key = [slice(None)] * result.ndim
            for i, s in zip(index, window):
                key[i] = s
            result[tuple(key)] += term

    return na.ScalarArray(
        ndarray=result,
        axes=axes,
    )
