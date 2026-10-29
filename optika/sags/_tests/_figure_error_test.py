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
)
"""A map whose axes are left to default to every axis of its inputs."""

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
        (_sag_grids, ("map_x", "map_y")),
        (_sag_none, ()),
    ],
)
def test_axis_(
    sag: optika.sags.FigureErrorSag,
    axis: tuple[str, ...],
) -> None:
    """
    The axes along which the map is sampled default to every axis of its
    inputs, and there are none without a map.
    """
    assert sag.axis_ == axis


@pytest.mark.parametrize(
    argnames="sag,shape",
    argvalues=[
        (_sag_flat, dict()),
        (_sag_parabolic, dict()),
        (_sag_channel, dict(channel=3)),
        (_sag_grids, dict(channel=3)),
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
