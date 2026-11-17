from typing import Sequence
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "SlopeErrorParameters",
    "slope_error",
]


@dataclasses.dataclass(eq=False, repr=False)
class SlopeErrorParameters(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    """The parameters needed to compute the slope error."""

    step_size: na.ScalarLike = 0 * u.mm
    """The horizontal distance to use when measuring the slope"""

    kernel_size: na.ScalarLike = 0 * u.mm
    """The size of the boxcar kernel that is convolved with the wavefront error."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.step_size),
            optika.shape(self.kernel_size),
        )


def slope_error(
    figure_error: na.AbstractFunctionArray,
    parameters: SlopeErrorParameters,
    axis: Sequence[str],
) -> na.AbstractScalar:
    r"""
    The RMS slope error of a map of the figure error of a surface.

    The map is smoothed by a boxcar as wide as the
    :attr:`~SlopeErrorParameters.kernel_size`, and the slope at each pixel,
    along each axis of the map, is the difference in the smoothed height
    across the :attr:`~SlopeErrorParameters.step_size`, divided by the step.
    The slope error is the RMS of the slopes along each axis, added in
    quadrature.

    The boxcar spans the whole number of pixels nearest to the kernel size,
    plus one if that number is even, so that it is centered on a pixel.
    The step spans the even number of pixels nearest to the step size, or the
    larger one if the step size is an odd number of pixels, so that each
    slope is centered on a pixel too.
    Each difference is divided by the step it was taken across.

    The boxcar averages only the pixels inside it which were measured, so it
    also fills in the pixels within half its width of a measured pixel.
    A slope is taken only where the smoothed map has a height at every pixel
    along its step, so a step across a gap which the boxcar does not fill is
    not a slope, and there must be at least one slope along each axis.

    Nothing is removed from the map, so a term which is not counted as an
    error, like a tilt, should be removed from it first.

    The heights and the positions of the pixels must have units of the same
    kind, such as lengths, or no units at all.
    With units, the slope error is an angle, so that it can be compared with
    a specification in any unit of angle.

    Parameters
    ----------
    figure_error
        A map of the height of a surface, such as one measured with an
        interferometer, sampled on a regular grid.
        Its inputs are the positions of the centers of the pixels, as a
        :class:`named_arrays.Cartesian2dVectorArray`, and its outputs are the
        heights there, which are NaN wherever a pixel was not measured.
        An infinite height is treated the same way.
        The heights are broadcast against the positions, so a map which does
        not vary along one of its axes may leave that axis out.
        The heights may be uncertain, but the positions may not.
    parameters
        The step size and the kernel size of the slope error.
        They may not be uncertain.
        The step must span at least two pixels, and each must span fewer
        pixels than the map along each of its axes.
    axis
        The two logical axes along which the map is sampled.
        Any other axis of the map, or of the parameters, is broadcast, and
        each map along it may be sampled on its own grid.

    See Also
    --------
    :meth:`optika.sags.FigureErrorSag.slope_error` : The slope error of the
        map of a sag profile.

    Examples
    --------

    Plot the slope error of a ripple with an amplitude of 1 nm against its
    period, which shows that the step and the kernel leave out the ripples
    which are shorter than them.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        parameters = optika.metrology.SlopeErrorParameters(
            step_size=4 * u.mm,
            kernel_size=2 * u.mm,
        )

        # a ripple along x, for a range of periods
        amplitude = 1 * u.nm
        period = na.geomspace(1, 100, axis="period", num=41) * u.mm

        # a map of each ripple, sampled on a grid of pixels
        pixel = na.Cartesian2dVectorLinearSpace(
            start=-50 * u.mm,
            stop=50 * u.mm,
            axis=na.Cartesian2dVectorArray("pixel_x", "pixel_y"),
            num=401,
        ).explicit
        height = amplitude * np.sin(2 * np.pi * u.rad * pixel.x / period)

        slope_error = optika.metrology.slope_error(
            figure_error=na.FunctionArray(pixel, height),
            parameters=parameters,
            axis=("pixel_x", "pixel_y"),
        )

        # the RMS slope of the ripple itself
        slope_ripple = 2 * np.pi * u.rad * amplitude / period / np.sqrt(2)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                period,
                slope_ripple.to(u.urad),
                ax=ax,
                axis="period",
                label="ripple",
            )
            na.plt.plot(
                period,
                slope_error.to(u.urad),
                ax=ax,
                axis="period",
                label="slope error",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(f"period ({ax.get_xlabel()})")
            ax.set_ylabel(f"RMS slope ({ax.get_ylabel()})")
            ax.legend()
    """
    axis = _axis(figure_error, axis)

    position = figure_error.inputs
    height = figure_error.outputs

    shape_inputs = na.shape(position)
    shape_outputs = na.shape(height)
    for a in axis:
        num = shape_outputs.get(a, shape_inputs[a])
        if num != shape_inputs[a]:
            raise ValueError(
                f"The inputs of `figure_error` must be the centers of its "
                f"pixels, so along {a!r} there must be as many of them, "
                f"{shape_inputs[a]}, as there are outputs, {num}."
            )
        if shape_inputs[a] < 2:
            raise ValueError(
                f"`figure_error` must have at least two pixels along {a!r}."
            )

    unit_height = na.unit_normalized(height)
    unit_position = na.unit_normalized(position.x)
    if not unit_height.is_equivalent(unit_position):
        raise ValueError(
            f"The heights of `figure_error`, in {unit_height!r}, and the "
            f"positions of its pixels, in {unit_position!r}, must have units "
            f"of the same kind, or no units at all."
        )

    shape_parameters = optika.shape(parameters)
    if any(a in shape_parameters for a in axis):
        raise ValueError(
            f"`parameters` may not vary along the axes of the map, {axis}, "
            f"but its shape is {shape_parameters}."
        )
    step_size = _certain(parameters.step_size, "`step_size`")
    kernel_size = _certain(parameters.kernel_size, "`kernel_size`")

    shape = na.broadcast_shapes(shape_inputs, shape_outputs, shape_parameters)
    shape_batch = {
        a: num
        for a, num in shape.items()
        if a not in axis and (a in shape_inputs or a in shape_parameters)
    }
    shape_map = {a: shape_inputs[a] for a in axis}

    return _slope_error(
        position=position.broadcast_to(
            na.broadcast_shapes(shape_inputs, shape_batch),
        ),
        height=height.broadcast_to(
            na.broadcast_shapes(shape_outputs, shape_batch, shape_map),
        ),
        step_size=step_size.broadcast_to(shape_batch),
        kernel_size=kernel_size.broadcast_to(shape_batch),
        axis=axis,
        shape_batch=shape_batch,
    )


def _slope_error(
    position: na.AbstractCartesian2dVectorArray,
    height: na.AbstractScalar,
    step_size: na.ScalarArray,
    kernel_size: na.ScalarArray,
    axis: tuple[str, ...],
    shape_batch: dict[str, int],
) -> na.AbstractScalar:
    """
    The slope error of each map along the batch axes, one map at a time,
    since the size of the boxcar and of the step in pixels depends on the
    grid and the parameters of each map.
    The maps along any other axis of the heights share a grid and parameters,
    so they are worked out together.
    """
    if shape_batch:
        a, *rest = shape_batch
        shape_rest = {r: shape_batch[r] for r in rest}
        return na.stack(
            arrays=[
                _slope_error(
                    position=position[{a: i}],
                    height=height[{a: i}],
                    step_size=step_size[{a: i}],
                    kernel_size=kernel_size[{a: i}],
                    axis=axis,
                    shape_batch=shape_rest,
                )
                for i in range(shape_batch[a])
            ],
            axis=a,
        )

    shape = na.shape(position)

    spacing = dict()
    size = dict()
    step = dict()
    for a in axis:
        spacing[a] = _spacing(position, a)
        ratio_kernel = _ratio(kernel_size.ndarray, spacing[a])
        ratio_step = _ratio(step_size.ndarray, spacing[a])
        if not (np.isfinite(ratio_kernel) and ratio_kernel >= 0):
            raise ValueError(
                f"`kernel_size`, {kernel_size.ndarray}, must be finite and at "
                f"least zero."
            )
        if not np.isfinite(ratio_step):
            raise ValueError(f"`step_size`, {step_size.ndarray}, must be finite.")
        size[a] = _odd(ratio_kernel)
        step[a] = _even(ratio_step)
        if not size[a] < shape[a]:
            raise ValueError(
                f"`kernel_size`, {kernel_size.ndarray}, must span fewer pixels "
                f"than the whole map along {a!r}, which has {shape[a]} pixels "
                f"{spacing[a]} apart."
            )
        if not 2 <= step[a] < shape[a]:
            raise ValueError(
                f"`step_size`, {step_size.ndarray}, must span at least two "
                f"pixels and fewer than the whole map along {a!r}, which has "
                f"{shape[a]} pixels {spacing[a]} apart."
            )

    smoothed = na.ndfilters.mean_filter(
        array=height,
        size=size,
        where=np.isfinite(height),
        mode="truncate",
    )
    unmeasured = np.where(np.isfinite(smoothed), 0.0, 1.0)

    mean_square = []
    for a in axis:
        num = step[a]
        upper = smoothed[{a: slice(num, None)}]
        lower = smoothed[{a: slice(None, -num)}]
        slope = (upper - lower) / (num * spacing[a])
        gap = na.ndfilters.mean_filter(
            array=unmeasured,
            size={a: num + 1},
            mode="truncate",
        )
        gap = gap[{a: slice(num // 2, shape[a] - num // 2)}]
        slope = np.where(gap > 0, np.nan, slope)
        if not np.all(np.any(np.isfinite(slope), axis=axis)):
            raise ValueError(
                f"There is no slope along {a!r}, since there is no step, "
                f"{step_size.ndarray}, along which the smoothed map has a "
                f"height at every pixel."
            )
        mean_square.append(np.nanmean(np.square(slope), axis=axis))
    result = np.sqrt(mean_square[0] + mean_square[1])

    if na.unit(result) is not None:
        result = result.to(u.dimensionless_unscaled) * u.rad

    return result


def _axis(
    figure_error: na.AbstractFunctionArray,
    axis: Sequence[str],
) -> tuple[str, ...]:
    """
    The two axes along which a map of the figure error is sampled, checked
    against the positions of its pixels.
    """
    if isinstance(axis, str):
        raise TypeError(
            f"`axis` must be a sequence of two axis names, not a str, {axis!r}."
        )
    axis = tuple(axis)

    position = figure_error.inputs
    if not isinstance(position, na.AbstractCartesian2dVectorArray):
        raise TypeError(
            "The inputs of `figure_error` must be the positions of the centers "
            "of its pixels, as a `named_arrays.AbstractCartesian2dVectorArray`, "
            f"not a {type(position).__name__}."
        )

    shape_inputs = na.shape(position)
    distinct = len(set(axis)) == len(axis) == 2
    if not distinct or any(a not in shape_inputs for a in axis):
        raise ValueError(
            f"`axis`, {axis}, must name two axes along which the inputs of "
            f"`figure_error` vary, {tuple(shape_inputs)}."
        )

    return axis


def _certain(
    value: na.ScalarLike,
    name: str,
) -> na.ScalarArray:
    """A scalar which is not uncertain."""
    value = na.as_named_array(value).explicit
    if not isinstance(value, na.ScalarArray):
        raise TypeError(f"{name} may not be uncertain.")
    return value


def _spacing(
    position: na.AbstractCartesian2dVectorArray,
    axis: str,
) -> float | u.Quantity:
    """
    The mean distance between the centers of neighboring pixels along an
    axis, in the plane of the map.
    """
    position = na.Cartesian2dVectorArray(position.x, position.y)
    difference = position[{axis: slice(1, None)}] - position[{axis: slice(None, -1)}]
    result = _certain(np.mean(difference.length), "The positions of the pixels")
    result = result.ndarray
    if not (np.isfinite(result) and result > 0):
        raise ValueError(
            f"The positions of the pixels of `figure_error` must be finite and "
            f"distinct, but along {axis!r} they are {result} apart on average."
        )
    return result


def _ratio(
    numerator: float | u.Quantity,
    denominator: float | u.Quantity,
) -> float:
    """The ratio of two lengths, as a number."""
    result = numerator / denominator
    if isinstance(result, u.Quantity):
        result = result.to(u.dimensionless_unscaled).value
    return float(result)


def _odd(x: float) -> int:
    """
    The whole number nearest to ``x``, plus one if it is even.

    ``x`` is rounded to nine decimal places first, so that a rounding error
    in a ratio of round lengths, like 3.5, cannot decide which way it goes.
    """
    result = int(np.rint(round(x, 9)))
    if result % 2 == 0:
        result = result + 1
    return result


def _even(x: float) -> int:
    """
    The even number nearest to ``x``, or the larger one if ``x`` is odd.

    ``x`` is rounded to nine decimal places first, so that a rounding error
    in a ratio of round lengths, like 3, cannot decide which way it goes.
    """
    return 2 * int(np.floor((1 + round(x, 9)) / 2))
