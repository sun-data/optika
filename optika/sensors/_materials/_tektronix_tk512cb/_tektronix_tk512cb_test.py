import pytest
import optika
from .._materials_test import AbstractTestAbstractStern1994BackilluminatedCCDMaterial


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.TektronixTK512CBMaterial(),
    ],
)
class TestTektronixTK512CBMaterial(
    AbstractTestAbstractStern1994BackilluminatedCCDMaterial,
):
    pass
