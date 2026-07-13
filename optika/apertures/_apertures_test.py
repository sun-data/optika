import dataclasses
import pytest
import numpy as np
import matplotlib.axes
import astropy.units as u
import named_arrays as na
import optika
import optika.rays._tests.test_ray_vectors
from .._tests import test_mixins

active_parameterization = [
    True,
    False,
    na.linspace(-1, 1, "active", 3) >= 0,
]

inverted_parameterization = [
    False,
    True,
    na.linspace(-1, 1, "inverted", 4) >= 0,
]

transform_parameterization = [
    None,
    na.transformations.Cartesian3dRotationZ(30 * u.deg),
]


def _spanning_parameters(**parameters: list) -> list[dict]:
    """
    Generate a list of parameter dictionaries in which every value of every
    parameter appears at least once, by varying one parameter at a time from
    a base combination, instead of taking the full Cartesian product.
    """
    base = {name: values[0] for name, values in parameters.items()}
    result = [base]
    for name, values in parameters.items():
        for value in values[1:]:
            result.append(base | {name: value})
    return result


class AbstractTestAbstractAperture(
    test_mixins.AbstractTestDxfWritable,
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestPlottable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
):
    def test_samples_wire(self, a: optika.apertures.AbstractAperture):
        assert isinstance(a.samples_wire, int)
        assert a.samples_wire > 0

    def test_active(self, a: optika.apertures.AbstractAperture):
        assert isinstance(a.active, (bool, na.AbstractScalar))
        assert np.issubdtype(na.get_dtype(a.active), bool)

    def test_inverted(self, a: optika.apertures.AbstractAperture):
        assert isinstance(a.inverted, (bool, na.AbstractScalar))
        assert np.issubdtype(na.get_dtype(a.inverted), bool)

    def test__call__(self, a: optika.apertures.AbstractAperture):
        position = na.Cartesian3dVectorLinearSpace(
            start=a.bound_lower,
            stop=a.bound_upper,
            axis=na.Cartesian3dVectorArray("x", "y", "z"),
            num=na.Cartesian3dVectorArray(5, 5, 1),
        )

        result = a(position)
        assert np.issubdtype(na.get_dtype(result), bool)

        assert np.all(result[~na.as_named_array(a.active)])

        # the centroid of the wire must be inside an active, non-inverted
        # aperture (every aperture here is star-shaped about its centroid)
        if (a.active is True) and (a.inverted is False):
            centroid = a.wire().mean("wire")
            assert np.all(na.as_named_array(a(centroid)))

    @pytest.mark.parametrize("rays", optika.rays._tests.test_ray_vectors.rays)
    def test_clip_rays(
        self,
        a: optika.apertures.AbstractAperture,
        rays: optika.rays.RayVectorArray,
    ):
        result = a.clip_rays(rays)
        assert isinstance(result, optika.rays.RayVectorArray)
        assert result is not rays
        assert np.all(result.position == rays.position)
        assert np.all(result.direction == rays.direction)
        assert np.all(result.intensity == rays.intensity)
        assert np.all(result.attenuation == rays.attenuation)
        assert np.all(result.index_refraction == rays.index_refraction)
        assert result.unvignetted is not rays.unvignetted
        assert np.mean(result.unvignetted) <= np.mean(rays.unvignetted)

    def test_bound_lower(self, a: optika.apertures.AbstractAperture):
        result = a.bound_lower
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        # a densely-sampled wire approaches the true extent from the inside
        wire = a.wire(num=10001)
        for component in ("x", "y"):
            bound = getattr(result, component)
            edge = getattr(wire, component).min("wire")
            scale = np.abs(getattr(a.bound_upper, component) - bound)
            # the bound must enclose the wire ...
            assert np.all(bound <= edge + 1e-9 * scale)
            # ... and be tight: it must not undershoot the true extent
            assert np.all(np.abs(bound - edge) < 1e-6 * scale)

    def test_bound_upper(self, a: optika.apertures.AbstractAperture):
        result = a.bound_upper
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        assert np.all(result.x != a.bound_lower.x)
        assert np.all(result.y != a.bound_lower.y)
        # a densely-sampled wire approaches the true extent from the inside
        wire = a.wire(num=10001)
        for component in ("x", "y"):
            bound = getattr(result, component)
            edge = getattr(wire, component).max("wire")
            scale = np.abs(bound - getattr(a.bound_lower, component))
            # the bound must enclose the wire ...
            assert np.all(bound >= edge - 1e-9 * scale)
            # ... and be tight: it must not overshoot the true extent
            assert np.all(np.abs(bound - edge) < 1e-6 * scale)

    def test_wire(self, a: optika.apertures.AbstractAperture):
        wire = a.wire()
        assert isinstance(wire, na.AbstractCartesian3dVectorArray)
        assert "wire" in wire.shape
        assert wire.shape["wire"] == a.samples_wire

    class TestPlot(
        test_mixins.AbstractTestPlottable.TestPlot,
    ):
        def test_plot(
            self,
            a: optika.apertures.AbstractAperture,
            ax: None | matplotlib.axes.Axes | na.ScalarArray,
            transformation: None | na.transformations.AbstractTransformation,
        ):
            if na.unit_normalized(a.wire()).is_equivalent(u.mm):
                super().test_plot(
                    a=a,
                    ax=ax,
                    transformation=transformation,
                )


radius_parameterization = [
    0.5,
    100 * u.mm,
    na.linspace(100, 200, axis="radius", num=4) * u.mm,
    na.NormalUncertainScalarArray(100 * u.mm, width=10 * u.mm),
]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.CircularAperture(
            radius=p["radius"],
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            radius=radius_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestCircularAperture(
    AbstractTestAbstractAperture,
):
    def test_radius(self, a: optika.apertures.CircularAperture):
        assert isinstance(a.radius, (float, u.Quantity, na.AbstractScalar))
        assert np.all(a.radius >= 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.CircularSectorAperture(
            radius=p["radius"],
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            radius=radius_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestCircularSectorAperture(
    AbstractTestAbstractAperture,
):
    def test_radius(self, a: optika.apertures.CircularSectorAperture):
        assert isinstance(a.radius, (float, u.Quantity, na.AbstractScalar))
        assert np.all(a.radius >= 0)

    def test_angle_start(self, a: optika.apertures.CircularSectorAperture):
        assert isinstance(a.radius, (float, u.Quantity, na.AbstractScalar))
        assert np.all(a.radius >= 0)

    def test_angle_stop(self, a: optika.apertures.CircularSectorAperture):
        assert isinstance(a.radius, (float, u.Quantity, na.AbstractScalar))
        assert np.all(a.radius >= 0)


def test_circular_sector_wire_samples_radial_arms():
    # The sector boundary has two straight radial arms in addition to the arc;
    # the wire must sample them, not just the arc.
    aperture = optika.apertures.CircularSectorAperture(
        radius=125 * u.mm,
        angle_start=90 * u.deg,
        angle_stop=180 * u.deg,
    )
    wire = aperture.wire(num=21)
    radius = np.sqrt(wire.x**2 + wire.y**2)
    on_arm = (radius > 1 * u.mm) & (radius < (125 * u.mm - 1 * u.mm))
    assert np.any(on_arm)


class AbstractTestAbstractPolygonalAperture(
    AbstractTestAbstractAperture,
):
    def test_vertices(self, a: optika.apertures.AbstractAperture):
        if a.vertices is not None:
            assert isinstance(a.vertices, na.AbstractCartesian3dVectorArray)
            assert "vertex" in a.vertices.shape


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.EllipticalAperture(
            radius=na.Cartesian2dVectorArray(50, 100) * u.mm,
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestEllipticalAperture(
    AbstractTestAbstractAperture,
):
    def test_radius(self, a: optika.apertures.EllipticalAperture):
        assert isinstance(a.radius, na.AbstractCartesian2dVectorArray)
        assert np.all(a.radius >= 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.PolygonalAperture(
            vertices=na.Cartesian3dVectorArray(
                x=na.ScalarArray([-1, 1, 1, -1] * u.mm, axes="vertex"),
                y=na.ScalarArray([-1, -1, 1, 1] * u.mm, axes="vertex"),
                z=0 * u.mm,
            ),
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestPolygonalAperture(
    AbstractTestAbstractPolygonalAperture,
):
    pass


half_width_parameterization = [
    0.5,
    100 * u.mm,
    na.linspace(100, 200, axis="radius", num=4) * u.mm,
    na.NormalUncertainScalarArray(100 * u.mm, width=10 * u.mm),
    na.Cartesian2dVectorArray(100 * u.mm, 200 * u.mm),
]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.RectangularAperture(
            half_width=p["half_width"],
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            half_width=half_width_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestRectangularAperture(
    AbstractTestAbstractPolygonalAperture,
):
    def test_half_width(self, a: optika.apertures.RectangularAperture):
        types_valid = (
            float,
            u.Quantity,
            na.AbstractScalar,
            na.AbstractCartesian2dVectorArray,
        )
        assert isinstance(a.half_width, types_valid)
        assert np.all(a.half_width >= 0)

    def test_decentered(self, a: optika.apertures.RectangularAperture):
        """
        A rectangular aperture decentered by more than its width must accept
        the center of the translated rectangle and reject the center of the
        untranslated one.

        Regression test: ``RectangularAperture.__call__`` used to express the
        rectangle in terms of :attr:`bound_lower`/:attr:`bound_upper` while
        also inverse-transforming the position, applying the internal
        transformation twice.
        """
        if (a.active is not True) or (a.inverted is not False):
            return
        half_width = a.half_width
        if isinstance(half_width, na.AbstractCartesian2dVectorArray):
            half_width = half_width.x
        decenter = 3 * half_width
        zero = 0 * decenter
        b = dataclasses.replace(
            a,
            transformation=na.transformations.Cartesian3dTranslation(
                x=decenter,
                y=zero,
                z=zero,
            ),
        )
        point_inside = na.Cartesian3dVectorArray(x=decenter, y=zero, z=zero)
        point_outside = na.Cartesian3dVectorArray(x=zero, y=zero, z=zero)
        assert np.all(b(point_inside))
        assert not np.any(b(point_outside))


class AbstractTestAbstractRegularPolygonalAperture(
    AbstractTestAbstractPolygonalAperture,
):
    def test_radius(self, a: optika.apertures.AbstractRegularPolygonalAperture):
        assert isinstance(na.as_named_array(a.radius), na.AbstractScalar)

    def test_num_vertices(self, a: optika.apertures.AbstractRegularPolygonalAperture):
        assert isinstance(a.num_vertices, int)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.RegularPolygonalAperture(
            radius=p["radius"],
            num_vertices=6,
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            radius=radius_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestRegularPolygonalAperture(
    AbstractTestAbstractRegularPolygonalAperture,
):
    pass


class AbstractTestAbstractOctagonalAperture(
    AbstractTestAbstractRegularPolygonalAperture,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.OctagonalAperture(
            radius=p["radius"],
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            radius=radius_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestOctagonalAperture(
    AbstractTestAbstractOctagonalAperture,
):
    pass


class AbstractTestAbstractIsoscelesTrapezoidalAperture(
    AbstractTestAbstractPolygonalAperture,
):
    def test_x_left(self, a: optika.apertures.AbstractIsoscelesTrapezoidalAperture):
        assert isinstance(na.as_named_array(a.x_left), na.AbstractScalar)

    def test_x_right(self, a: optika.apertures.AbstractIsoscelesTrapezoidalAperture):
        assert isinstance(na.as_named_array(a.x_right), na.AbstractScalar)

    def test_angle(self, a: optika.apertures.AbstractIsoscelesTrapezoidalAperture):
        assert isinstance(na.as_named_array(a.angle), na.AbstractScalar)


x_left_parameterization = [
    25 * u.mm,
    na.linspace(25, 30, axis="x_left", num=2) * u.mm,
]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.IsoscelesTrapezoidalAperture(
            x_left=p["x_left"],
            x_right=50 * u.mm,
            angle=45 * u.deg,
            samples_wire=21,
            active=p["active"],
            inverted=p["inverted"],
            transformation=p["transformation"],
            kwargs_plot=p["kwargs_plot"],
        )
        for p in _spanning_parameters(
            x_left=x_left_parameterization,
            active=active_parameterization,
            inverted=inverted_parameterization,
            transformation=transform_parameterization,
            kwargs_plot=test_mixins.kwargs_plot_parameterization,
        )
    ],
)
class TestIsoscelesTrapezoidalAperture(
    AbstractTestAbstractIsoscelesTrapezoidalAperture,
):
    pass
