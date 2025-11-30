import pytest
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


class AbstractTestAbstractSphericalSag(
    AbstractTestAbstractSag,
):
    def test_curvature(self, a: optika.sags.SphericalSag):
        assert isinstance(na.as_named_array(a.curvature), na.AbstractScalar)
        assert na.shape(a.curvature) == na.shape(a.radius)


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
