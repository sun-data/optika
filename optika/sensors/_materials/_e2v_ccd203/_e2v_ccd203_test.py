import pytest
import optika
from .._materials_test import AbstractTestAbstractStern1994BackilluminatedCCDMaterial


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.E2VCCD203Material(),
    ],
)
class TestE2VCCD203Material(
    AbstractTestAbstractStern1994BackilluminatedCCDMaterial,
):
    pass
