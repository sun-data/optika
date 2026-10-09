"""
Spreading the values on a pixel grid with a kernel centered on each pixel,
and the transpose of that spread.
"""

import math
import numpy as np
import numba
import astropy.units as u
import named_arrays as na

__all__ = []

_axis_interpolation = "_blur_interpolation"
"""The logical axis of the nodes a kernel is interpolated between."""


def _magnitude(
    array: float | np.ndarray | u.Quantity,
    unit: None | u.UnitBase = None,
) -> tuple[np.ndarray, None | u.UnitBase]:
    """
    The values of an array as floating-point numbers, in `unit` if given,
    and the unit they are in, or :obj:`None` if the array has none.

    Parameters
    ----------
    array
        An array, which may have a unit.
    unit
        The unit to express the values in, if the array has one.
    """
    if isinstance(array, u.Quantity):
        if unit is None:
            unit = array.unit
        return np.asarray(array.to_value(unit), dtype=float), unit
    return np.asarray(array, dtype=float), None


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

    Each pixel of the result is a sum over the pixels of the kernel in a
    fixed order, computed by a single thread of :func:`_blur_compiled`,
    so the result does not depend on the number of threads.

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
        If given, a pair of arrays ``(position, nodes)``:
        the kernel varies along :obj:`_axis_interpolation` between `nodes`,
        which must increase along it,
        and the kernel of each pixel is interpolated linearly at `position`
        as each pixel of the kernel is used,
        so that the kernels of all the pixels are never held at once.
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
    num_node = shape_kernel.pop(_axis_interpolation, 1)
    if interpolation is not None:
        shape_kernel = na.broadcast_shapes(
            shape_kernel,
            na.shape(interpolation[0]),
            {
                axis: num
                for axis, num in na.shape(interpolation[1]).items()
                if axis != _axis_interpolation
            },
        )

    shape = na.broadcast_shapes(shape_values, shape_kernel)
    shape_batch = {axis: num for axis, num in shape.items() if axis not in axis_xy}
    shape_grid = shape_batch | {axis: shape[axis] for axis in axis_xy}
    shape_offset = dict(zip(axis_kernel, num_kernel))
    num_batch = math.prod(shape_batch.values())
    num_x, num_y = (shape[axis] for axis in axis_xy)

    def batched(
        array: na.AbstractScalar,
        shape_array: dict[str, int],
        unit: None | u.UnitBase = None,
    ) -> tuple[np.ndarray, None | u.UnitBase]:
        """
        An array broadcast to `shape_array`, whose leading axes are those of
        the batch, with those axes collapsed into one,
        as floating-point numbers in `unit` if given, and its unit.
        """
        array = na.broadcast_to(array, shape_array)
        ndarray, unit = _magnitude(array.ndarray_aligned(tuple(shape_array)), unit)
        shape_trailing = tuple(shape_array.values())[len(shape_batch) :]
        return ndarray.reshape((num_batch,) + shape_trailing), unit

    x, unit_values = batched(values, shape_grid)

    if interpolation is not None:
        # each pixel has the kernel between two nodes
        shape_nodes = shape_batch | {_axis_interpolation: num_node}
        table, unit_kernel = batched(kernel, shape_nodes | shape_offset)
        position, unit_position = batched(interpolation[0], shape_grid)
        nodes, _ = batched(interpolation[1], shape_nodes, unit_position)
        index = np.empty(x.shape, dtype=np.int64)
        weight = np.empty(x.shape)
        _interpolation_weights(position, nodes, index, weight)
        uniform = False
    elif any(axis in na.shape(kernel) for axis in axis_xy):
        # each pixel has its own kernel
        table, unit_kernel = batched(kernel, shape_grid | shape_offset)
        table = table.reshape((num_batch, num_x * num_y) + num_kernel)
        index = np.arange(num_x * num_y, dtype=np.int64).reshape((1, num_x, num_y))
        index = np.broadcast_to(index, x.shape)
        weight = np.broadcast_to(np.zeros((1, 1, 1)), x.shape)
        uniform = False
    else:
        # every pixel has the same kernel
        table, unit_kernel = batched(kernel, shape_batch | shape_offset)
        table = table.reshape((num_batch, 1) + num_kernel)
        index = np.broadcast_to(np.zeros((1, 1, 1), dtype=np.int64), x.shape)
        weight = np.broadcast_to(np.zeros((1, 1, 1)), x.shape)
        uniform = True

    result = _blur_compiled(
        values=np.ascontiguousarray(x),
        table=np.ascontiguousarray(table),
        index=index,
        weight=weight,
        uniform=uniform,
        wrap=wrap,
        transpose=transpose,
    )

    result = result.reshape(tuple(shape_grid.values()))
    if unit_values is not None or unit_kernel is not None:
        unit = u.dimensionless_unscaled
        for unit_factor in (unit_values, unit_kernel):
            if unit_factor is not None:
                unit = unit * unit_factor
        result = result << unit

    return na.ScalarArray(
        ndarray=result,
        axes=tuple(shape_grid),
    )


@numba.njit(cache=True, parallel=True)
def _interpolation_weights(
    position: np.ndarray,
    nodes: np.ndarray,
    index: np.ndarray,
    weight: np.ndarray,
) -> None:  # pragma: nocover
    """
    For each pixel, the node at or below its position and the weight of the
    node above it in the linear interpolation between them,
    clamped to the range of the nodes.

    Parameters
    ----------
    position
        The position of each pixel, of shape ``(batch, x, y)``.
    nodes
        The nodes of the interpolation, increasing along their last axis,
        of shape ``(batch, node)``.
    index
        The index of the node at or below each position, filled in place.
    weight
        The weight of the node above it, filled in place.
    """
    num_batch, num_x, num_y = position.shape
    num_node = nodes.shape[1]
    for t in numba.prange(num_batch * num_x):
        b = t // num_x
        i = t % num_x
        for j in range(num_y):
            p = position[b, i, j]
            n = np.searchsorted(nodes[b], p, side="right") - 1
            n = min(max(n, 0), max(num_node - 2, 0))
            w = 0.0
            if n + 1 < num_node:
                low = nodes[b, n]
                high = nodes[b, n + 1]
                if high > low:
                    w = min(max((p - low) / (high - low), 0.0), 1.0)
            index[b, i, j] = n
            weight[b, i, j] = w


@numba.njit(cache=True, parallel=True)
def _blur_compiled(
    values: np.ndarray,
    table: np.ndarray,
    index: np.ndarray,
    weight: np.ndarray,
    uniform: bool,
    wrap: bool,
    transpose: bool,
) -> np.ndarray:  # pragma: nocover
    """
    The compiled body of :func:`_blur`.

    Each row of the result is computed by one thread, which sums the
    contributions of the pixels of the kernel to each of its pixels in a
    fixed order.

    Parameters
    ----------
    values
        The values on the pixel grid, of shape ``(batch, x, y)``.
    table
        The kernels, of shape ``(batch, entry, kernel_x, kernel_y)``.
    index
        The entry of `table` holding the kernel of each pixel,
        of shape ``(batch, x, y)``.
    weight
        The weight of the next entry of `table`, which the kernel of each
        pixel is interpolated toward, of shape ``(batch, x, y)``.
    uniform
        Whether every pixel has the first entry of `table` as its kernel,
        in which case `index` and `weight` are not used.
    wrap
        Whether the values which leave the edge of the grid re-enter the
        opposite edge.
    transpose
        Whether to apply the transpose of the spread instead.
    """
    num_batch, num_x, num_y = values.shape
    num_kernel_x = table.shape[2]
    num_kernel_y = table.shape[3]
    half_x = num_kernel_x // 2
    half_y = num_kernel_y // 2

    # the forward spread gathers each pixel from the pixels the kernel moves
    # charge from, and the transpose from the pixels it moves charge to
    sign = 1 if transpose else -1

    result = np.zeros(values.shape)

    for t in numba.prange(num_batch * num_x):
        b = t // num_x
        i = t % num_x
        for a in range(num_kernel_x):
            s = i + sign * (a - half_x)
            if wrap:
                s = s % num_x
            elif s < 0 or s >= num_x:
                continue
            for c in range(num_kernel_y):
                shift = sign * (c - half_y)
                if uniform:
                    k = table[b, 0, a, c]
                    if wrap:
                        for j in range(num_y):
                            result[b, i, j] += k * values[b, s, (j + shift) % num_y]
                    else:
                        start = max(0, -shift)
                        stop = min(num_y, num_y - shift)
                        for j in range(start, stop):
                            result[b, i, j] += k * values[b, s, j + shift]
                    continue
                for j in range(num_y):
                    r = j + shift
                    if wrap:
                        r = r % num_y
                    elif r < 0 or r >= num_y:
                        continue
                    # the kernel is that of the pixel the charge starts in,
                    # the source of the forward spread and the target of the
                    # transpose
                    if transpose:
                        n = index[b, i, j]
                        w = weight[b, i, j]
                    else:
                        n = index[b, s, r]
                        w = weight[b, s, r]
                    k = table[b, n, a, c]
                    if w > 0:
                        k = (1 - w) * k + w * table[b, n + 1, a, c]
                    result[b, i, j] += k * values[b, s, r]

    return result
