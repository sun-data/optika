import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.metrology.SlopeErrorParameters(
            step_size=4 * u.mm,
            kernel_size=2 * u.mm,
        )
    ],
)
class TestSlopeErrorParameters(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test_step_size(self, a: optika.metrology.SlopeErrorParameters):
        assert np.issubdtype(na.get_dtype(a.step_size), float)
        assert na.unit_normalized(a.step_size).is_equivalent(u.mm)

    def test_kernel_size(self, a: optika.metrology.SlopeErrorParameters):
        assert np.issubdtype(na.get_dtype(a.kernel_size), float)
        assert na.unit_normalized(a.kernel_size).is_equivalent(u.mm)


_axis = ("pixel_x", "pixel_y")
"""The axes along which the test maps are sampled."""


def _pixels(
    num_x: int = 41,
    num_y: int = 37,
    spacing: u.Quantity = 0.5 * u.mm,
    angle: u.Quantity = 0 * u.deg,
) -> na.Cartesian2dVectorArray:
    """
    The centers of the pixels of a map, on a grid which is rotated by
    ``angle``.
    """
    index_x = na.arange(0, num_x, axis=_axis[0]) - (num_x - 1) / 2
    index_y = na.arange(0, num_y, axis=_axis[1]) - (num_y - 1) / 2
    x = index_x * spacing
    y = index_y * spacing
    return na.Cartesian2dVectorArray(
        x=x * np.cos(angle) - y * np.sin(angle),
        y=x * np.sin(angle) + y * np.cos(angle),
    )


def _height(
    pixel: na.AbstractCartesian2dVectorArray,
    seed: int = 0,
) -> na.ScalarArray:
    """
    Random heights at each pixel, which were not measured outside a circle
    and at a few pixels inside it.
    """
    pixel = pixel.explicit
    shape = na.shape(pixel)
    rng = np.random.default_rng(seed)
    height = na.ScalarArray(
        ndarray=rng.normal(size=tuple(shape.values())) * u.nm,
        axes=tuple(shape),
    )
    radius = 0.45 * np.minimum(
        (pixel.x.max() - pixel.x.min()).ndarray,
        (pixel.y.max() - pixel.y.min()).ndarray,
    )
    dropped = na.ScalarArray(
        ndarray=rng.uniform(size=tuple(shape.values())) < 0.02,
        axes=tuple(shape),
    )
    measured = (pixel.length < radius) & ~dropped
    return np.where(measured, height, np.nan)


def _half_reference(ratio: u.Quantity) -> int:
    """
    The number of pixels on either side of the center of a boxcar whose
    width is ``ratio`` pixels.
    """
    return int(np.rint(round(ratio.to(u.one).value, 9))) // 2


def _step_reference(ratio: u.Quantity) -> int:
    """
    The number of pixels in a step which is ``ratio`` pixels long, the even
    number nearest to it, or the larger one if it is odd.
    """
    ratio = round(ratio.to(u.one).value, 9)
    lower = 2 * int(np.floor(ratio / 2))
    if ratio - lower >= 1:
        return lower + 2
    return lower


def _slope_error_reference(
    position: na.AbstractCartesian2dVectorArray,
    height: na.AbstractScalar,
    step_size: u.Quantity,
    kernel_size: u.Quantity,
) -> u.Quantity:
    """
    The slope error of a single map, worked out pixel by pixel with numpy.
    """
    x = position.x.broadcast_to(na.shape(position)).ndarray_aligned(_axis)
    y = position.y.broadcast_to(na.shape(position)).ndarray_aligned(_axis)
    z = height.ndarray_aligned(_axis)
    num_x, num_y = z.shape

    spacing_x = np.mean(np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2))
    spacing_y = np.mean(np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2))

    half_x = _half_reference(kernel_size / spacing_x)
    half_y = _half_reference(kernel_size / spacing_y)
    smoothed = np.full(z.shape, np.nan) * z.unit
    for i in range(num_x):
        for j in range(num_y):
            box = z[
                max(0, i - half_x) : i + half_x + 1,
                max(0, j - half_y) : j + half_y + 1,
            ]
            if np.isfinite(box).any():
                smoothed[i, j] = np.nanmean(box)

    step_x = _step_reference(step_size / spacing_x)
    step_y = _step_reference(step_size / spacing_y)
    slope_x = [
        (smoothed[i + step_x, j] - smoothed[i, j]) / (step_x * spacing_x)
        for i in range(num_x - step_x)
        for j in range(num_y)
        if np.isfinite(smoothed[i : i + step_x + 1, j]).all()
    ]
    slope_y = [
        (smoothed[i, j + step_y] - smoothed[i, j]) / (step_y * spacing_y)
        for i in range(num_x)
        for j in range(num_y - step_y)
        if np.isfinite(smoothed[i, j : j + step_y + 1]).all()
    ]
    slope_x = u.Quantity(slope_x)
    slope_y = u.Quantity(slope_y)
    result = np.sqrt(np.mean(slope_x**2) + np.mean(slope_y**2))
    return result.to(u.rad, equivalencies=u.dimensionless_angles())


_pixels_default = _pixels()
"""The pixels of a map which is aligned with the coordinates."""

_pixels_rotated = _pixels(angle=30 * u.deg)
"""The pixels of a map which is rotated against the coordinates."""

_pixels_rectangular = na.Cartesian2dVectorArray(
    x=na.linspace(-10, 10, axis=_axis[0], num=41) * u.mm,
    y=na.linspace(-6, 6, axis=_axis[1], num=81) * u.mm,
)
"""The pixels of a map which are farther apart along x than along y."""


@pytest.mark.parametrize(
    argnames="position",
    argvalues=[
        _pixels_default,
        _pixels_rotated,
        _pixels_rectangular,
    ],
)
@pytest.mark.parametrize(
    argnames="parameters",
    argvalues=[
        optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm),
        optika.metrology.SlopeErrorParameters(1.2 * u.mm, 0 * u.mm),
        optika.metrology.SlopeErrorParameters(3 * u.mm, 1.6 * u.mm),
        optika.metrology.SlopeErrorParameters(5.6 * u.mm, 1 * u.mm),
    ],
)
def test_slope_error(
    position: na.AbstractCartesian2dVectorArray,
    parameters: optika.metrology.SlopeErrorParameters,
) -> None:
    """
    The slope error of a map with unmeasured pixels is the one worked out
    pixel by pixel.
    """
    height = _height(position)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    expected = _slope_error_reference(
        position=position,
        height=height,
        step_size=parameters.step_size,
        kernel_size=parameters.kernel_size,
    )
    assert isinstance(result, na.ScalarArray)
    assert result.shape == dict()
    assert result.unit.is_equivalent(u.rad)
    assert np.isclose(result.ndarray, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    argnames="angle",
    argvalues=[0 * u.deg, 30 * u.deg],
)
def test_slope_error_tilt(angle: u.Quantity) -> None:
    """
    The slope error of a tilted plane without smoothing is its slope, even
    with pixels which were not measured, whichever way the grid is rotated.
    """
    position = _pixels(angle=angle)
    height = 3 * u.nm / u.mm * position.x - 4 * u.nm / u.mm * position.y
    height = np.where(np.isfinite(_height(position)), height, np.nan)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=optika.metrology.SlopeErrorParameters(2 * u.mm, 0 * u.mm),
        axis=_axis,
    )
    assert np.isclose(result.ndarray, 5 * u.urad, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    argnames="step_size,num",
    argvalues=[
        (5.0 * u.mm, 10),
        (5.4 * u.mm, 10),
        (5.5 * u.mm, 12),
        (5.6 * u.mm, 12),
    ],
)
def test_slope_error_step(step_size: u.Quantity, num: int) -> None:
    """
    The step spans the even number of pixels nearest to the step size, or the
    larger one if the step size is an odd number of pixels.

    Along each row of 40 pixels, ``num`` of the ``40 - num`` steps straddle a
    jump in the height, so the slope error of the jump depends on the number
    of pixels in the step.
    """
    spacing = 0.5 * u.mm
    position = _pixels(num_x=40, num_y=15, spacing=spacing)
    jump = 1 * u.nm
    height = np.where(position.x > 0 * u.mm, jump, 0 * u.nm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=optika.metrology.SlopeErrorParameters(step_size, 0 * u.mm),
        axis=_axis,
    )
    expected = jump / (spacing * np.sqrt(num * (40 - num))) * u.rad
    assert np.isclose(result.ndarray, expected.to(u.rad), rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    argnames="parameters,parameters_equivalent",
    argvalues=[
        (
            optika.metrology.SlopeErrorParameters(0.3 * u.mm, 0 * u.mm),
            optika.metrology.SlopeErrorParameters(0.4 * u.mm, 0 * u.mm),
        ),
        (
            optika.metrology.SlopeErrorParameters(0.8 * u.mm, 0.35 * u.mm),
            optika.metrology.SlopeErrorParameters(0.8 * u.mm, 0.5 * u.mm),
        ),
    ],
)
def test_slope_error_round_lengths(
    parameters: optika.metrology.SlopeErrorParameters,
    parameters_equivalent: optika.metrology.SlopeErrorParameters,
) -> None:
    """
    A step of 3 pixels spans 4 of them, and a kernel of 3.5 pixels spans 5 of
    them, although the ratios of these lengths to the distance between the
    pixels are not quite 3 and 3.5 in floating point.
    """
    position = _pixels(spacing=0.1 * u.mm)
    height = _height(position)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    expected = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters_equivalent,
        axis=_axis,
    )
    assert np.all(result == expected)


def test_slope_error_implicit() -> None:
    """
    Implicit positions and parameters give the slope error of the explicit
    ones.
    """
    position = na.Cartesian2dVectorLinearSpace(
        start=-10 * u.mm,
        stop=10 * u.mm,
        axis=na.Cartesian2dVectorArray(*_axis),
        num=41,
    )
    height = _height(position)
    step_size = na.ScalarLinearSpace(2 * u.mm, 4 * u.mm, axis="case", num=2)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=optika.metrology.SlopeErrorParameters(step_size, 2 * u.mm),
        axis=_axis,
    )
    expected = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position.explicit, height),
        parameters=optika.metrology.SlopeErrorParameters(
            step_size=step_size.explicit,
            kernel_size=2 * u.mm,
        ),
        axis=_axis,
    )
    assert result.shape == dict(case=2)
    assert np.all(result == expected)


def test_slope_error_gap() -> None:
    """
    A step across a gap in the map which the boxcar does not fill is not a
    slope, so two flat halves of a map at different heights have no slope
    error.
    """
    position = _pixels()
    height = np.where(position.x > 0 * u.mm, 5 * u.nm, -5 * u.nm)
    height = np.where(np.abs(position.x) > 1 * u.mm, height, np.nan)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=optika.metrology.SlopeErrorParameters(4 * u.mm, 1 * u.mm),
        axis=_axis,
    )
    assert np.all(result == 0 * u.rad)


def test_slope_error_plane() -> None:
    """
    The pixels are spaced in the plane of the map, so positions which also
    have a height give the same slope error.
    """
    position = _pixels_rotated
    height = _height(position)
    parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(
            inputs=na.Cartesian3dVectorArray(
                x=position.x,
                y=position.y,
                z=np.square(position.length) / (100 * u.mm),
            ),
            outputs=height,
        ),
        parameters=parameters,
        axis=_axis,
    )
    expected = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    assert np.all(result == expected)


def test_slope_error_units() -> None:
    """
    The units of the positions and the heights may differ, and without units
    the slope error is a number.
    """
    position = _pixels_default
    height = _height(position)
    parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    result_meters = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position.to(u.m), height.to(u.mm)),
        parameters=optika.metrology.SlopeErrorParameters(4e-3 * u.m, 2e-3 * u.m),
        axis=_axis,
    )
    result_unitless = optika.metrology.slope_error(
        figure_error=na.FunctionArray(
            position.to(u.mm).value,
            height.to(u.mm).value,
        ),
        parameters=optika.metrology.SlopeErrorParameters(4, 2),
        axis=_axis,
    )
    assert np.isclose(result_meters.ndarray, result.ndarray, rtol=1e-12, atol=0)
    assert na.unit(result_unitless) is None
    assert np.isclose(
        result_unitless.ndarray,
        result.ndarray.to_value(u.rad),
        rtol=1e-12,
        atol=0,
    )


def test_slope_error_broadcast() -> None:
    """
    The slope error of a separate map for each channel, each on its own grid,
    for each of several steps and kernels, is that of each map with each step
    and kernel.
    """
    pixels = [_pixels_default, _pixels(spacing=0.3 * u.mm, angle=30 * u.deg)]
    position = na.stack(pixels, axis="channel")
    height = na.stack(
        [_height(p, seed=seed) for seed, p in enumerate(pixels)],
        axis="channel",
    )
    step_size = na.ScalarArray([2, 4] * u.mm, axes="case")
    kernel_size = na.ScalarArray([1, 2] * u.mm, axes="case")
    parameters = optika.metrology.SlopeErrorParameters(step_size, kernel_size)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    assert result.shape == dict(channel=2, case=2)
    for index in na.ndindex(result.shape):
        expected = optika.metrology.slope_error(
            figure_error=na.FunctionArray(position[index], height[index]),
            parameters=optika.metrology.SlopeErrorParameters(
                step_size=step_size[index].ndarray,
                kernel_size=kernel_size[index].ndarray,
            ),
            axis=_axis,
        )
        assert np.all(result[index] == expected)


def test_slope_error_broadcast_heights() -> None:
    """
    The slope error of several maps on one grid, which are worked out
    together, is that of each map.
    """
    position = _pixels_default
    height = na.stack(
        [_height(position, seed=seed) for seed in range(3)],
        axis="sample",
    )
    parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    assert result.shape == dict(sample=3)
    for index in na.ndindex(result.shape):
        expected = optika.metrology.slope_error(
            figure_error=na.FunctionArray(position, height[index]),
            parameters=parameters,
            axis=_axis,
        )
        assert np.all(result[index] == expected)


def test_slope_error_broadcast_size_one() -> None:
    """
    An axis of the map with a single map along it is broadcast against the
    same axis of the parameters.
    """
    position = _pixels_default
    height = _height(position).add_axes("channel")
    step_size = na.ScalarArray([2, 4] * u.mm, axes="channel")
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=optika.metrology.SlopeErrorParameters(step_size, 1 * u.mm),
        axis=_axis,
    )
    assert result.shape == dict(channel=2)
    for index in na.ndindex(result.shape):
        expected = optika.metrology.slope_error(
            figure_error=na.FunctionArray(position, height[dict(channel=0)]),
            parameters=optika.metrology.SlopeErrorParameters(
                step_size=step_size[index].ndarray,
                kernel_size=1 * u.mm,
            ),
            axis=_axis,
        )
        assert np.all(result[index] == expected)


def test_slope_error_constant_along_axis() -> None:
    """
    A map which does not vary along one of its axes may leave that axis out
    of its heights.
    """
    position = _pixels_rectangular
    height = 2 * u.nm * np.sin(2 * np.pi * u.rad * position.x / (7 * u.mm))
    assert _axis[1] not in height.shape
    parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    expected = optika.metrology.slope_error(
        figure_error=na.FunctionArray(
            inputs=position,
            outputs=height.broadcast_to(na.shape(position)),
        ),
        parameters=parameters,
        axis=_axis,
    )
    assert np.all(result == expected)


def test_slope_error_uncertain() -> None:
    """
    The slope error of an uncertain map is that of its nominal map and of
    each map in its distribution.
    """
    position = _pixels_default
    height = _height(position)
    noise = na.random.normal(
        loc=0 * u.nm,
        scale=0.1 * u.nm,
        shape_random=dict(_distribution=3),
        seed=0,
    )
    height_uncertain = na.UncertainScalarArray(
        nominal=height,
        distribution=height + noise,
    )
    parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height_uncertain),
        parameters=parameters,
        axis=_axis,
    )
    expected = na.UncertainScalarArray(
        nominal=optika.metrology.slope_error(
            figure_error=na.FunctionArray(position, height_uncertain.nominal),
            parameters=parameters,
            axis=_axis,
        ),
        distribution=optika.metrology.slope_error(
            figure_error=na.FunctionArray(position, height_uncertain.distribution),
            parameters=parameters,
            axis=_axis,
        ),
    )
    assert isinstance(result, na.UncertainScalarArray)
    assert np.all(result.nominal == expected.nominal)
    assert np.all(result.distribution == expected.distribution)


_figure_error = na.FunctionArray(_pixels_default, _height(_pixels_default))
"""A map which is valid, to break one thing at a time."""

_parameters = optika.metrology.SlopeErrorParameters(4 * u.mm, 2 * u.mm)
"""Parameters which are valid, to break one thing at a time."""


@pytest.mark.parametrize(
    argnames="figure_error,parameters,axis,error,match",
    argvalues=[
        (_figure_error, _parameters, "pixel_x", TypeError, "`axis`"),
        (_figure_error, _parameters, ("pixel_X", "pixel_y"), ValueError, "`axis`"),
        (_figure_error, _parameters, ("pixel_x",), ValueError, "`axis`"),
        (_figure_error, _parameters, ("pixel_x", "pixel_x"), ValueError, "`axis`"),
        (
            _figure_error,
            _parameters,
            ("pixel_x", "pixel_y", "channel"),
            ValueError,
            "`axis`",
        ),
        (
            na.FunctionArray(
                inputs=_pixels_default.x,
                outputs=_figure_error.outputs,
            ),
            _parameters,
            _axis,
            TypeError,
            "Cartesian2dVectorArray",
        ),
        (
            na.FunctionArray(
                inputs=_pixels(num_x=42, num_y=38),
                outputs=_figure_error.outputs,
            ),
            _parameters,
            _axis,
            ValueError,
            "centers",
        ),
        (
            na.FunctionArray(
                inputs=_pixels_default,
                outputs=_figure_error.outputs.value,
            ),
            _parameters,
            _axis,
            ValueError,
            "units",
        ),
        (
            na.FunctionArray(
                inputs=_pixels_default.value,
                outputs=_figure_error.outputs,
            ),
            optika.metrology.SlopeErrorParameters(4, 2),
            _axis,
            ValueError,
            "units",
        ),
        (
            na.FunctionArray(
                inputs=_pixels(num_x=1),
                outputs=_height(_pixels(num_x=1)),
            ),
            _parameters,
            _axis,
            ValueError,
            "two pixels",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(0.2 * u.mm, 2 * u.mm),
            _axis,
            ValueError,
            "`step_size`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(19 * u.mm, 2 * u.mm),
            _axis,
            ValueError,
            "`step_size`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(np.nan * u.mm, 2 * u.mm),
            _axis,
            ValueError,
            "`step_size`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(4 * u.mm, np.nan * u.mm),
            _axis,
            ValueError,
            "`kernel_size`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(4 * u.mm, -2 * u.mm),
            _axis,
            ValueError,
            "`kernel_size`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(
                step_size=na.ScalarArray([2, 4] * u.mm, axes="pixel_x"),
                kernel_size=2 * u.mm,
            ),
            _axis,
            ValueError,
            "`parameters`",
        ),
        (
            _figure_error,
            optika.metrology.SlopeErrorParameters(
                step_size=na.UncertainScalarArray(4 * u.mm, 4 * u.mm),
                kernel_size=2 * u.mm,
            ),
            _axis,
            TypeError,
            "`step_size` may not be uncertain",
        ),
        (
            na.FunctionArray(
                inputs=na.Cartesian2dVectorArray(
                    x=na.UncertainScalarArray(_pixels_default.x, _pixels_default.x),
                    y=_pixels_default.y,
                ),
                outputs=_figure_error.outputs,
            ),
            _parameters,
            _axis,
            TypeError,
            "positions of the pixels may not be uncertain",
        ),
    ],
)
def test_slope_error_invalid(
    figure_error: na.AbstractFunctionArray,
    parameters: optika.metrology.SlopeErrorParameters,
    axis: str | tuple[str, ...],
    error: type[Exception],
    match: str,
) -> None:
    """
    A str, a misnamed, single, repeated or third axis, inputs which are not
    vectors or not the centers of the pixels, heights and positions in units
    of different kinds, a single pixel, a step shorter than a pixel, as long
    as the map, or not finite, a kernel which is not finite or is negative,
    parameters which vary along the map, and uncertain parameters or
    positions are each rejected.
    """
    with pytest.raises(error, match=match):
        optika.metrology.slope_error(
            figure_error=figure_error,
            parameters=parameters,
            axis=axis,
        )
