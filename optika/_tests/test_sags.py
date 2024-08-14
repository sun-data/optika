import numpy as np
import pytest
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika
from . import test_mixins


class AbstractTestAbstractSag(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
):
    def test_parameters_slope_error(self, a: optika.sags.AbstractSag):
        if a.parameters_slope_error is not None:
            assert isinstance(
                a.parameters_slope_error,
                optika.metrology.SlopeErrorParameters,
            )

    def test_parameters_roughness(self, a: optika.sags.AbstractSag):
        if a.parameters_roughness is not None:
            assert isinstance(
                a.parameters_roughness,
                optika.metrology.RoughnessParameters,
            )

    def test_parameters_microroughness(self, a: optika.sags.AbstractSag):
        if a.parameters_microroughness is not None:
            assert isinstance(
                a.parameters_microroughness,
                optika.metrology.RoughnessParameters,
            )

    @pytest.mark.parametrize(
        argnames="position",
        argvalues=[
            na.Cartesian3dVectorArray() * u.mm,
            na.Cartesian3dVectorLinearSpace(0, 1, axis="s", num=5) * u.mm,
            na.Cartesian3dVectorArray(
                x=na.ScalarLinearSpace(0, 1, axis="x", num=5) * u.mm,
                y=na.NormalUncertainScalarArray(
                    nominal=na.ScalarLinearSpace(-1, 0, axis="y", num=6) * u.mm,
                    width=0.1 * u.mm,
                ),
                z=0 * u.mm,
            ),
        ],
    )
    class TestFunctionsOfPosition:
        def test__call__(
            self,
            a: optika.sags.AbstractSag,
            position: na.AbstractCartesian3dVectorArray,
        ):
            result = a(position)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            if na.shape(result):
                assert set(na.shape(position)).issubset(na.shape(result))

        def test_normal(
            self,
            a: optika.sags.AbstractSag,
            position: na.AbstractCartesian3dVectorArray,
        ):
            result = a.normal(position)
            assert isinstance(result, na.AbstractCartesian3dVectorArray)
            if na.shape(result):
                assert set(na.shape(position)).issubset(na.shape(result))

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=500 * u.nm,
                position=na.Cartesian3dVectorArray(1, 2, 3) * u.mm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            optika.rays.RayVectorArray(
                wavelength=na.NormalUncertainScalarArray(500 * u.nm, width=5 * u.nm),
                position=na.Cartesian3dVectorArray(
                    x=na.NormalUncertainScalarArray(1 * u.mm, width=0.1 * u.mm),
                    y=na.NormalUncertainScalarArray(2 * u.mm, width=0.1 * u.mm),
                    z=na.NormalUncertainScalarArray(3 * u.mm, width=0.1 * u.mm),
                ),
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            optika.rays.RayVectorArray(
                wavelength=na.linspace(500, 600, axis="y", num=5) * u.nm,
                position=na.Cartesian3dVectorLinearSpace(0, 5, axis="y", num=5) * u.mm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    class TestFunctionsOfRays:
        def test_intercept(
            self,
            a: optika.sags.AbstractSag,
            rays: optika.rays.AbstractRayVectorArray,
        ):
            result = a.intercept(rays)
            assert isinstance(result, optika.rays.AbstractRayVectorArray)
            assert np.all(np.isfinite(result.position))
            assert np.allclose(a(result.position), result.position.z)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.NoSag(
            parameters_slope_error=optika.metrology.SlopeErrorParameters(
                kernel_size=2 * u.mm,
                step_size=4 * u.mm,
            ),
            parameters_roughness=optika.metrology.RoughnessParameters(
                period_min=2 * u.mm,
                period_max=4 * u.mm,
            ),
            parameters_microroughness=optika.metrology.RoughnessParameters(
                period_min=0.1 * u.mm,
                period_max=2 * u.mm,
            ),
        )
    ],
)
class TestNoSag(
    AbstractTestAbstractSag,
):
    pass


class AbstractTestAbstractSphericalSag(
    AbstractTestAbstractSag,
):
    def test_curvature(self, a: optika.sags.SphericalSag):
        assert isinstance(na.as_named_array(a.curvature), na.AbstractScalar)
        assert na.shape(a.curvature) == na.shape(a.radius)


def radius_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        100 * u.mm,
        na.ScalarLinearSpace(100, 1000, axis="radius", num=5) * u.mm,
    ]
    widths = [
        None,
        10 * u.mm,
    ]
    return [
        nominal if width is None else na.NormalUncertainScalarArray(nominal, width)
        for nominal in nominals
        for width in widths
    ]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.SphericalSag(
            radius=radius,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestSphericalSag(
    AbstractTestAbstractSphericalSag,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.CylindricalSag(
            radius=radius,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestCylindricalSag(
    AbstractTestAbstractSag,
):
    pass


class AbstractTestAbstractConicSag(
    AbstractTestAbstractSag,
):
    def test_radius(self, a: optika.sags):
        assert isinstance(na.as_named_array(a.radius), na.ScalarLike)
        assert na.unit_normalized(a.radius).is_equivalent(u.mm)

    def test_conic(self, a: optika.sags):
        assert isinstance(na.as_named_array(a.conic), na.ScalarLike)
        assert na.unit_normalized(a.conic).is_equivalent(u.dimensionless_unscaled)


def conic_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        0 * u.dimensionless_unscaled,
        na.ScalarLinearSpace(0, 1, axis="conic", num=4),
    ]
    widths = [
        None,
        0.1 * u.dimensionless_unscaled,
    ]
    return [
        nominal if width is None else na.NormalUncertainScalarArray(nominal, width)
        for nominal in nominals
        for width in widths
    ]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ConicSag(
            radius=radius,
            conic=conic,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for conic in conic_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestConicSag(
    AbstractTestAbstractConicSag,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ParabolicSag(
            focal_length=radius / 2,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestParabolicSag(
    AbstractTestAbstractConicSag,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ToroidalSag(
            radius=radius,
            radius_of_rotation=2 * radius_of_rotation,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for radius_of_rotation in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestToroidalSag(
    AbstractTestAbstractSphericalSag,
):
    pass
