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
    The step spans the whole number of pixels nearest to the step size, which
    is what each difference is divided by.
    The boxcar averages only the pixels inside it which were measured, so it
    also fills in the pixels within half its width of a measured pixel, and a
    slope is taken wherever both ends of its step have a height.

    Nothing is removed from the map, so a term which is not counted as an
    error, like a tilt, should be removed from it first.

    If the map has units, the slope error is an angle, so that it can be
    compared with a specification in any unit of angle.

    Parameters
    ----------
    figure_error
        A map of the height of a surface, such as one measured with an
        interferometer, sampled on a regular grid.
        Its inputs are the positions of the centers of the pixels, as a
        :class:`named_arrays.Cartesian2dVectorArray`, and its outputs are the
        heights there, which are NaN wherever a pixel was not measured.
        The heights may be uncertain, but the positions may not.
    parameters
        The step size and the kernel size of the slope error.
        They may not be uncertain.
    axis
        The two logical axes along which the map is sampled.
        Any other axis of the map, or of the parameters, is broadcast, and
        each map along it may be sampled on its own grid.

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
    if isinstance(axis, str):
        raise TypeError(
            f"`axis` must be a sequence of two axis names, not a str, {axis!r}."
        )
    axis = tuple(axis)

    shape_inputs = na.shape(figure_error.inputs)
    shape_outputs = na.shape(figure_error.outputs)
    distinct = len(set(axis)) == len(axis) == 2
    if not distinct or any(a not in shape_inputs for a in axis):
        raise ValueError(
            f"`axis`, {axis}, must name two axes along which the inputs of "
            f"`figure_error` vary, {tuple(shape_inputs)}."
        )
    for a in axis:
        if shape_outputs.get(a) != shape_inputs[a]:
            raise ValueError(
                f"The inputs of `figure_error` must be the centers of its "
                f"pixels, so along {a!r} there must be as many of them, "
                f"{shape_inputs[a]}, as there are outputs, {shape_outputs.get(a)}."
            )
        if shape_inputs[a] < 2:
            raise ValueError(
                f"`figure_error` must have at least two pixels along {a!r}."
            )

    shape_parameters = optika.shape(parameters)
    if any(a in shape_parameters for a in axis):
        raise ValueError(
            f"`parameters` may not vary along the axes of the map, {axis}, "
            f"but its shape is {shape_parameters}."
        )

    shape_batch = na.broadcast_shapes(
        {a: num for a, num in na.shape(figure_error).items() if a not in axis},
        shape_parameters,
    )

    return _slope_error(
        figure_error=figure_error,
        step_size=parameters.step_size,
        kernel_size=parameters.kernel_size,
        axis=axis,
        shape_batch=shape_batch,
    )


def _slope_error(
    figure_error: na.AbstractFunctionArray,
    step_size: na.ScalarLike,
    kernel_size: na.ScalarLike,
    axis: tuple[str, ...],
    shape_batch: dict[str, int],
) -> na.AbstractScalar:
    """
    The slope error of each map along the batch axes, one map at a time,
    since the size of the boxcar and of the step in pixels depends on the
    grid of each map.
    """
    if shape_batch:
        a, *rest = shape_batch
        shape_rest = {r: shape_batch[r] for r in rest}
        return na.stack(
            arrays=[
                _slope_error(
                    figure_error=figure_error[{a: i}],
                    step_size=na.as_named_array(step_size)[{a: i}],
                    kernel_size=na.as_named_array(kernel_size)[{a: i}],
                    axis=axis,
                    shape_batch=shape_rest,
                )
                for i in range(shape_batch[a])
            ],
            axis=a,
        )

    position = figure_error.inputs
    height = figure_error.outputs
    shape = na.shape(position)

    step_size = _certain(step_size, "`step_size`")
    kernel_size = _certain(kernel_size, "`kernel_size`")

    spacing = dict()
    size = dict()
    step = dict()
    for a in axis:
        spacing[a] = _spacing(position, a)
        size[a] = _odd(_ratio(kernel_size, spacing[a]))
        step[a] = int(np.rint(_ratio(step_size, spacing[a])))
        if not 1 <= step[a] < shape[a]:
            raise ValueError(
                f"`step_size`, {step_size}, must span at least one pixel and "
                f"less than the whole map along {a!r}, which has "
                f"{shape[a]} pixels {spacing[a]} apart."
            )

    smoothed = na.ndfilters.mean_filter(
        array=height,
        size=size,
        where=np.isfinite(height),
        mode="truncate",
    )

    result = 0
    for a in axis:
        num = step[a]
        upper = smoothed[{a: slice(num, None)}]
        lower = smoothed[{a: slice(None, -num)}]
        slope = (upper - lower) / (num * spacing[a])
        result = result + np.nanmean(np.square(slope))
    result = np.sqrt(result)

    if na.unit(result) is not None:
        result = result * u.rad

    return result


def _certain(
    value: na.ScalarLike,
    name: str,
) -> float | u.Quantity:
    """A certain scalar, as a number or a quantity."""
    value = na.as_named_array(value)
    if not isinstance(value, na.ScalarArray):
        raise TypeError(f"{name} may not be uncertain.")
    return value.ndarray


def _spacing(
    position: na.AbstractCartesian2dVectorArray,
    axis: str,
) -> float | u.Quantity:
    """The mean distance between the centers of neighboring pixels along an axis."""
    difference = position[{axis: slice(1, None)}] - position[{axis: slice(None, -1)}]
    return _certain(np.mean(difference.length), "The positions of the pixels")


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
    The whole number nearest to ``x``, plus one if it is even, and at least
    one.

    A ratio of round lengths, like 4, is a whole number which rounding errors
    can put on either side, but it gives the same odd number from either
    side.
    """
    result = int(np.rint(x))
    if result % 2 == 0:
        result = result + 1
    return max(1, result)
