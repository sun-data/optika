import pytest
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


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
