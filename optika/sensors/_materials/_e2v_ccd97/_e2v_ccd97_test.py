import pytest
import optika
from .._materials_test import AbstractTestAbstractBackilluminatedCCDMaterial


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.E2VCCD97Material(),
    ],
)
class TestE2VCCD97Material(
    AbstractTestAbstractBackilluminatedCCDMaterial,
):
    pass
