import pytest
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


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
