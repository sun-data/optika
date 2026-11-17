import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika.rays._tests.test_ray_vectors
from ._abc_test import AbstractTestAbstractSag, positions

_grid = na.Cartesian2dVectorLinearSpace(
    start=-60 * u.mm,
    stop=60 * u.mm,
    axis=na.Cartesian2dVectorArray("map_x", "map_y"),
    num=na.Cartesian2dVectorArray(31, 41),
).explicit
"""The positions at which the maps of the figure error are sampled."""


def _height(
    position: na.AbstractCartesian2dVectorArray,
) -> na.AbstractScalar:
    """
    An affine figure error, which multilinear interpolation reproduces
    exactly.

    Parameters
    ----------
    position
        The position at which to evaluate the figure error.
    """
    return (3 * position.x - 2 * position.y) * (u.nm / u.mm) + 5 * u.nm


_channel = na.ScalarArray(np.array([1.0, 2.0, 3.0]), axes="channel")
"""The scale of a separate map of the figure error for each channel."""

_offset = na.NormalUncertainScalarArray(0 * u.nm, width=1 * u.nm, seed=42)
"""An uncertain offset of a map of the figure error."""

_sag_flat = optika.sags.FigureErrorSag(
    figure_error=na.FunctionArray(_grid, _height(_grid)),
    axis=("map_x", "map_y"),
)
"""A map of the figure error of a flat surface."""

_sag_parabolic = optika.sags.FigureErrorSag(
    base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
    figure_error=na.FunctionArray(_grid, _height(_grid)),
    axis=["map_x", "map_y"],
)
"""A map whose axes are given as a list."""

_sag_channel = optika.sags.FigureErrorSag(
    base=optika.sags.SphericalSag(radius=500 * u.mm),
    figure_error=na.FunctionArray(_grid, _channel * _height(_grid)),
    axis=("map_x", "map_y"),
    transformation=na.transformations.Cartesian3dTranslation(x=5 * u.mm, z=10 * u.mm),
)
"""A separate map for each channel, on a transformed sag profile."""

_grid_channel = _grid + na.Cartesian2dVectorArray(_channel * u.mm, 0 * u.mm)
"""A separate grid for each channel."""

_sag_grids = optika.sags.FigureErrorSag(
    base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
    figure_error=na.FunctionArray(_grid_channel, _height(_grid_channel)),
    axis=("map_x", "map_y"),
)
"""A map sampled on a separate grid for each channel."""

_sag_uncertain = optika.sags.FigureErrorSag(
    base=optika.sags.ZernikeSag(
        base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
        coefficients=[0, 0, 0, 100, 0, 50] * u.nm,
        radius=50 * u.mm,
    ),
    figure_error=na.FunctionArray(_grid, _height(_grid) + _offset),
    axis=("map_x", "map_y"),
)
"""An uncertain map on top of a fit of its low-order terms."""

_unmeasured = (np.abs(_grid.x) < 10 * u.mm) & (np.abs(_grid.y) < 7 * u.mm)
"""A block of pixels in the middle of the maps which were not measured."""

_sag_unmeasured = optika.sags.FigureErrorSag(
    base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
    figure_error=na.FunctionArray(
        inputs=_grid,
        outputs=np.where(_unmeasured, np.nan, _height(_grid)),
    ),
    axis=("map_x", "map_y"),
)
"""A map with pixels which were not measured."""

_sag_none = optika.sags.FigureErrorSag(
    base=optika.sags.SphericalSag(radius=500 * u.mm),
)
"""No figure error."""


def _base(
    sag: optika.sags.FigureErrorSag,
) -> optika.sags.AbstractSag:
    """
    The base profile of a sag profile, found without
    :attr:`optika.sags.FigureErrorSag.base_`, which is under test.

    Parameters
    ----------
    sag
        The sag profile whose base profile to find.
    """
    if sag.base is None:
        return optika.sags.NoSag()
    return sag.base


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        _sag_flat,
        _sag_parabolic,
        _sag_channel,
        _sag_grids,
        _sag_uncertain,
        _sag_unmeasured,
        _sag_none,
    ],
)
class TestFigureErrorSag(
    AbstractTestAbstractSag,
):
    @pytest.mark.parametrize("position", positions)
    def test_normal(
        self,
        a: optika.sags.FigureErrorSag,
        position: na.AbstractCartesian3dVectorArray,
    ) -> None:
        """
        The figure error does not tilt the surface, so the normal vector is
        the one of the base profile.
        """
        super().test_normal(a, position)

        result = a.normal(position)

        if a.transformation is not None:
            position = a.transformation.inverse(position)
        result_expected = _base(a).normal(position)

        assert np.all(result == result_expected)

    @pytest.mark.parametrize("rays", optika.rays._tests.test_ray_vectors.rays)
    def test_intercept(
        self,
        a: optika.sags.FigureErrorSag,
        rays: optika.rays.RayVectorArray,
    ) -> None:
        """
        The figure error does not deflect rays, so they intercept the base
        profile exactly where they would without it.
        This replaces the abstract test, which requires the intercepts to lie
        on the surface which includes the figure error.
        """
        result = a.intercept(rays)

        transformation = a.transformation
        if transformation is not None:
            rays = transformation.inverse(rays)
        result_expected = _base(a).intercept(rays)
        if transformation is not None:
            result_expected = transformation(result_expected)

        assert np.all(result.position == result_expected.position)
        assert np.all(result.direction == result_expected.direction)


@pytest.mark.parametrize(
    argnames="sag,scale,offset",
    argvalues=[
        (_sag_flat, 1, 0 * u.nm),
        (_sag_parabolic, 1, 0 * u.nm),
        (_sag_channel, _channel, 0 * u.nm),
        (_sag_grids, na.ScalarArray(np.ones(3), axes="channel"), 0 * u.nm),
        (_sag_uncertain, 1, _offset),
        (_sag_none, 0, 0 * u.nm),
    ],
)
def test__call__figure_error(
    sag: optika.sags.FigureErrorSag,
    scale: float | na.AbstractScalar,
    offset: u.Quantity | na.AbstractScalar,
) -> None:
    """
    The height of the surface above the base profile is the figure error,
    evaluated in the coordinate system of the sag profile.
    """
    position = na.Cartesian3dVectorArray(
        x=na.random.uniform(-50, 50, shape_random=dict(point=11), seed=0) * u.mm,
        y=na.random.uniform(-50, 50, shape_random=dict(point=11), seed=1) * u.mm,
        z=0 * u.mm,
    )

    result = sag(position)

    if sag.transformation is not None:
        position = sag.transformation.inverse(position)
    result_expected = scale * _height(position.xy) + offset

    assert np.allclose(result - _base(sag)(position), result_expected)


@pytest.mark.parametrize(
    argnames="sag,axis",
    argvalues=[
        (_sag_flat, ("map_x", "map_y")),
        (_sag_parabolic, ("map_x", "map_y")),
        (_sag_channel, ("map_x", "map_y")),
        (_sag_grids, ("map_x", "map_y")),
        (_sag_uncertain, ("map_x", "map_y")),
        (_sag_none, ()),
        (dataclasses.replace(_sag_none, axis=("map_x", "map_y")), ()),
    ],
)
def test_axis_(
    sag: optika.sags.FigureErrorSag,
    axis: tuple[str, ...],
) -> None:
    """
    The axes along which the map is sampled are a tuple, and there are none
    without a map, even if they are named.
    """
    assert sag.axis_ == axis


@pytest.mark.parametrize(
    argnames="sag,error",
    argvalues=[
        (dataclasses.replace(_sag_flat, axis=None), ValueError),
        (dataclasses.replace(_sag_flat, axis="map_x"), TypeError),
        (dataclasses.replace(_sag_flat, axis=("map_X", "map_Y")), ValueError),
        (dataclasses.replace(_sag_flat, axis=("map_x",)), ValueError),
        (dataclasses.replace(_sag_flat, axis=("map_x", "map_x")), ValueError),
        (
            dataclasses.replace(_sag_grids, axis=("map_x", "map_y", "channel")),
            ValueError,
        ),
    ],
)
def test_axis_invalid(
    sag: optika.sags.FigureErrorSag,
    error: type[Exception],
) -> None:
    """
    The axes along which a map is sampled must be named, as two distinct
    axes of its inputs, and a mistake fails as soon as the shape of the sag
    profile is needed, instead of an axis of the map being reported as if it
    were a batch axis.
    """
    with pytest.raises(error, match="`axis`"):
        sag.shape
    with pytest.raises(error, match="`axis`"):
        sag(na.Cartesian3dVectorArray() * u.mm)


@pytest.mark.parametrize(
    argnames="sag,shape",
    argvalues=[
        (_sag_flat, dict()),
        (_sag_parabolic, dict()),
        (_sag_channel, dict(channel=3)),
        (_sag_grids, dict(channel=3)),
        (_sag_uncertain, dict()),
        (_sag_none, dict()),
    ],
)
def test_shape_figure_error(
    sag: optika.sags.FigureErrorSag,
    shape: dict[str, int],
) -> None:
    """
    The axes along which the map of the figure error is sampled are not
    axes of the sag profile, but its other axes are.
    """
    assert sag.shape == shape


def test_axis_inputs_invalid() -> None:
    """
    The inputs of a map must be the positions of its pixels, as vectors, and a
    map whose inputs are not fails as soon as the shape of the sag profile is
    needed, like a mistake in its axes.
    """
    sag = dataclasses.replace(
        _sag_flat,
        figure_error=na.FunctionArray(_grid.x, _height(_grid)),
    )
    with pytest.raises(TypeError, match="Cartesian2dVectorArray"):
        sag.shape


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test__call__unmeasured(value: float) -> None:
    """
    The figure error is taken to be zero at the pixels which were not
    measured, which are NaN or infinite, so the surface is the one whose map
    is zero there.
    """
    sag = dataclasses.replace(
        _sag_unmeasured,
        figure_error=na.FunctionArray(
            inputs=_grid,
            outputs=np.where(_unmeasured, value * u.nm, _height(_grid)),
        ),
    )
    sag_zero = dataclasses.replace(
        sag,
        figure_error=na.FunctionArray(
            inputs=_grid,
            outputs=np.where(_unmeasured, 0 * u.nm, _height(_grid)),
        ),
    )

    position = na.Cartesian3dVectorArray(
        x=na.random.uniform(-20, 20, shape_random=dict(point=101), seed=0) * u.mm,
        y=na.random.uniform(-20, 20, shape_random=dict(point=101), seed=1) * u.mm,
        z=0 * u.mm,
    )

    result = sag(position)

    assert np.all(np.isfinite(result))
    assert np.all(result == sag_zero(position))


def _ripple(
    position: na.AbstractCartesian2dVectorArray,
) -> na.AbstractScalar:
    """
    A figure error whose slope varies across the map, and is not symmetric
    about either axis, so that its slope error depends on which pixels are
    counted.

    Parameters
    ----------
    position
        The position at which to evaluate the figure error.
    """
    x = position.x / (7 * u.mm) * u.rad + 1 * u.rad
    y = position.y / (11 * u.mm) * u.rad + 2 * u.rad
    return 10 * u.nm * np.sin(x) * np.cos(y)


_ripple_unmeasured = np.where(_unmeasured, np.nan, _ripple(_grid))
"""The ripple, with pixels which were not measured."""

_parameters = optika.metrology.SlopeErrorParameters(
    step_size=8 * u.mm,
    kernel_size=4 * u.mm,
)
"""The step and the kernel of the slope error of the maps."""

_aperture = optika.apertures.CircularAperture(radius=40.5 * u.mm)
"""A clear aperture which leaves out the pixels near the edges of the maps."""

_tilt = 20 * u.deg
"""The angle by which a sag profile, or an aperture, is tilted about the :math:`y` axis."""

_sphere = optika.sags.SphericalSag(radius=200 * u.mm)
"""
A base profile curved enough that, on a tilted sag profile or behind a tilted
aperture, its height moves some of the pixels of the maps across the edge of
the aperture.
"""

_height_sphere = _sphere(na.Cartesian3dVectorArray(_grid.x, _grid.y, 0 * u.mm))
"""The height of :obj:`_sphere` at each pixel of the maps."""


def _inside(
    x: na.AbstractScalar,
) -> na.AbstractScalar:
    """
    Whether each pixel of the maps is inside :obj:`_aperture`, worked out
    directly instead of with the aperture.

    Parameters
    ----------
    x
        The :math:`x` coordinate of each pixel in the coordinates of the
        aperture.
        Its :math:`y` coordinate is the same as in the coordinates of the
        sag profile.
    """
    return np.sqrt(np.square(x) + np.square(_grid.y)) <= 40.5 * u.mm


@pytest.mark.parametrize(
    argnames="sag,height,aperture,height_expected",
    argvalues=[
        (_sag_flat, _ripple(_grid), None, _ripple(_grid)),
        (_sag_parabolic, _ripple_unmeasured, None, _ripple_unmeasured),
        (
            _sag_parabolic,
            _ripple_unmeasured,
            _aperture,
            np.where(_inside(_grid.x), _ripple_unmeasured, np.nan),
        ),
        (
            _sag_channel,
            _channel * _ripple(_grid),
            _aperture,
            np.where(_inside(_grid.x + 5 * u.mm), _channel * _ripple(_grid), np.nan),
        ),
        (
            optika.sags.FigureErrorSag(
                base=_sphere,
                axis=("map_x", "map_y"),
                transformation=na.transformations.Cartesian3dRotationY(_tilt),
            ),
            _ripple(_grid),
            _aperture,
            np.where(
                _inside(_grid.x * np.cos(_tilt) + _height_sphere * np.sin(_tilt)),
                _ripple(_grid),
                np.nan,
            ),
        ),
        (
            optika.sags.FigureErrorSag(
                base=_sphere,
                axis=("map_x", "map_y"),
            ),
            _ripple(_grid),
            optika.apertures.CircularAperture(
                radius=40.5 * u.mm,
                transformation=na.transformations.Cartesian3dRotationY(_tilt),
            ),
            np.where(
                _inside(_grid.x * np.cos(_tilt) - _height_sphere * np.sin(_tilt)),
                _ripple(_grid),
                np.nan,
            ),
        ),
        (
            _sag_flat,
            _ripple(_grid) + _offset,
            _aperture,
            np.where(_inside(_grid.x), _ripple(_grid) + _offset, np.nan),
        ),
    ],
)
def test_slope_error(
    sag: optika.sags.FigureErrorSag,
    height: na.AbstractScalar,
    aperture: None | optika.apertures.AbstractAperture,
    height_expected: na.AbstractScalar,
) -> None:
    """
    The slope error of a sag profile is the one of its map, worked out by
    :func:`optika.metrology.slope_error`, without the pixels which were not
    measured or are outside the aperture, if one is given.
    The aperture is in surface coordinates, so the transformation of a sag
    profile moves it across the map, and each pixel is placed on the base
    profile, so that a tilt of the sag profile or of the aperture also moves
    the pixel by the height of the base profile there.
    """
    sag = dataclasses.replace(
        sag,
        figure_error=na.FunctionArray(_grid, height),
        parameters_slope_error=_parameters,
    )

    result = sag.slope_error(aperture)

    expected = optika.metrology.slope_error(
        figure_error=na.FunctionArray(_grid, height_expected),
        parameters=_parameters,
        axis=("map_x", "map_y"),
    )

    assert result.shape == expected.shape
    if isinstance(expected, na.UncertainScalarArray):
        assert isinstance(result, na.UncertainScalarArray)
        assert np.all(result.nominal == expected.nominal)
        assert np.all(result.distribution == expected.distribution)
    else:
        assert np.all(result == expected)


@pytest.mark.parametrize(
    argnames="sag,aperture,error,match",
    argvalues=[
        (
            dataclasses.replace(_sag_none, parameters_slope_error=_parameters),
            None,
            ValueError,
            "`figure_error`",
        ),
        (_sag_flat, None, ValueError, "`parameters_slope_error`"),
        (
            dataclasses.replace(
                _sag_flat, axis=None, parameters_slope_error=_parameters
            ),
            None,
            ValueError,
            "`axis`",
        ),
        (
            dataclasses.replace(
                _sag_flat,
                figure_error=na.FunctionArray(_grid.x, _height(_grid)),
                parameters_slope_error=_parameters,
            ),
            _aperture,
            TypeError,
            "Cartesian2dVectorArray",
        ),
        (
            dataclasses.replace(_sag_flat, parameters_slope_error=_parameters),
            optika.apertures.CircularAperture(radius=1 * u.mm),
            ValueError,
            "no slope",
        ),
    ],
)
def test_slope_error_invalid(
    sag: optika.sags.FigureErrorSag,
    aperture: None | optika.apertures.AbstractAperture,
    error: type[Exception],
    match: str,
) -> None:
    """
    The slope error needs a map, whose inputs are the positions of its pixels
    as vectors, the step and the kernel, and the axes along which the map is
    sampled, and a mistake in the map is caught before it is cut to the
    aperture.
    An aperture which leaves no slope to take is an error too.
    """
    with pytest.raises(error, match=match):
        sag.slope_error(aperture)
