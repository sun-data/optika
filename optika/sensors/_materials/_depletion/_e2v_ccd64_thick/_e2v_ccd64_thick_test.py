import pytest
import optika
from .._depletion_test import AbstractTestAbstractJanesickDepletionModel


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.E2VCCD64ThickDepletionModel(),
    ],
)
class TestE2VCCD64ThickDepletionModel(
    AbstractTestAbstractJanesickDepletionModel,
):
    pass
