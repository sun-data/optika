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

    half_x = int(np.rint((kernel_size / spacing_x).to(u.one).value)) // 2
    half_y = int(np.rint((kernel_size / spacing_y).to(u.one).value)) // 2
    smoothed = np.full(z.shape, np.nan) * z.unit
    for i in range(num_x):
        for j in range(num_y):
            box = z[
                max(0, i - half_x) : i + half_x + 1,
                max(0, j - half_y) : j + half_y + 1,
            ]
            if np.isfinite(box).any():
                smoothed[i, j] = np.nanmean(box)

    step_x = int(np.rint((step_size / spacing_x).to(u.one).value))
    step_y = int(np.rint((step_size / spacing_y).to(u.one).value))
    slope_x = (smoothed[step_x:, :] - smoothed[:-step_x, :]) / (step_x * spacing_x)
    slope_y = (smoothed[:, step_y:] - smoothed[:, :-step_y]) / (step_y * spacing_y)
    result = np.sqrt(np.nanmean(slope_x**2) + np.nanmean(slope_y**2))
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
    for each of several steps, is that of each map with each step.
    """
    pixels = [_pixels_default, _pixels(spacing=0.3 * u.mm, angle=30 * u.deg)]
    position = na.stack(pixels, axis="channel")
    height = na.stack(
        [_height(p, seed=seed) for seed, p in enumerate(pixels)],
        axis="channel",
    )
    step_size = na.ScalarArray([2, 4] * u.mm, axes="step")
    parameters = optika.metrology.SlopeErrorParameters(step_size, 1 * u.mm)
    result = optika.metrology.slope_error(
        figure_error=na.FunctionArray(position, height),
        parameters=parameters,
        axis=_axis,
    )
    assert result.shape == dict(channel=2, step=2)
    for index in na.ndindex(result.shape):
        expected = optika.metrology.slope_error(
            figure_error=na.FunctionArray(position[index], height[index]),
            parameters=optika.metrology.SlopeErrorParameters(
                step_size=step_size[index].ndarray,
                kernel_size=1 * u.mm,
            ),
            axis=_axis,
        )
        assert np.all(result[index] == expected)


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
    the centers of the pixels, a single pixel, a step shorter than half a
    pixel or as long as the map, parameters which vary along the map, and
    uncertain parameters or positions are each rejected.
    """
    with pytest.raises(error, match=match):
        optika.metrology.slope_error(
            figure_error=figure_error,
            parameters=parameters,
            axis=axis,
        )
