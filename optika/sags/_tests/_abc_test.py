import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ..._tests import test_propagators
import optika.rays._tests.test_ray_vectors

positions = [
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
]


def radius_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        100 * u.mm,
        -100 * u.mm,
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


class AbstractTestAbstractSag(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
    test_propagators.AbstractTestAbstractRayPropagator,
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

    @pytest.mark.parametrize("position", positions)
    def test__call__(
        self,
        a: optika.sags.AbstractSag,
        position: na.AbstractCartesian3dVectorArray,
    ):
        result = a(position)
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.mm)

    @pytest.mark.parametrize("position", positions)
    def test_normal(
        self,
        a: optika.sags.AbstractSag,
        position: na.AbstractCartesian3dVectorArray,
    ):
        result = a.normal(position)
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result.z < 0)
        assert np.allclose(result.length, 1)

    @pytest.mark.parametrize("rays", optika.rays._tests.test_ray_vectors.rays)
    def test_intercept(
        self,
        a: optika.sags.AbstractSag,
        rays: optika.rays.RayVectorArray,
    ):
        result = a.intercept(rays)

        assert np.allclose(a(result.position), result.position.z)

        result_expected = optika.sags.AbstractSag.intercept(a, rays)

        assert np.allclose(result, result_expected)
