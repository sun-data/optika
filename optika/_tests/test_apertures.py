import pytest
import numpy as np
import matplotlib.axes
import astropy.units as u
import named_arrays as na
import optika
import optika.rays._tests.test_ray_vectors
from . import test_mixins


active_parameterization = [
    True,
    na.linspace(-1, 1, "active", 3) >= 0,
]

inverted_parameterization = [
    False,
    na.linspace(-1, 1, "inverted", 4) >= 0,
]

transform_parameterization = [
    None,
    na.transformations.Cartesian3dRotationZ(30 * u.deg),
]


class AbstractTestAbstractAperture(
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
        assert set(na.shape(position)).issubset(na.shape(result))
        assert np.any(result)

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
        assert isinstance(a.bound_lower, na.AbstractCartesian3dVectorArray)

    def test_bound_upper(self, a: optika.apertures.AbstractAperture):
        assert isinstance(a.bound_upper, na.AbstractCartesian3dVectorArray)
        assert np.all(a.bound_upper.x != a.bound_lower.x)
        assert np.all(a.bound_upper.y != a.bound_lower.y)

    def test_vertices(self, a: optika.apertures.AbstractAperture):
        if a.vertices is not None:
            assert isinstance(a.vertices, na.AbstractCartesian3dVectorArray)
            assert "vertex" in a.vertices.shape

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
            radius=radius,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for radius in radius_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            radius=radius,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for radius in radius_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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


class AbstractTestAbstractPolygonalAperture(
    AbstractTestAbstractAperture,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.apertures.EllipticalAperture(
            radius=radius,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for radius in [na.Cartesian2dVectorArray(50, 100) * u.mm]
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            half_width=half_width,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for half_width in half_width_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            radius=radius,
            num_vertices=6,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for radius in radius_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            radius=radius,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for radius in radius_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
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
            x_left=x_left,
            x_right=50 * u.mm,
            angle=45 * u.deg,
            samples_wire=21,
            active=active,
            inverted=inverted,
            transformation=transformation,
            kwargs_plot=kwargs_plot,
        )
        for x_left in x_left_parameterization
        for active in active_parameterization
        for inverted in inverted_parameterization
        for transformation in transform_parameterization
        for kwargs_plot in test_mixins.kwargs_plot_parameterization
    ],
)
class TestIsoscelesTrapezoidalAperture(
    AbstractTestAbstractIsoscelesTrapezoidalAperture,
):
    pass
