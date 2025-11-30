import pytest
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


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
    AbstractTestAbstractSag,
):
    pass
