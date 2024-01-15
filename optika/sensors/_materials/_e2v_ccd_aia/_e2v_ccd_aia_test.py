import pytest
import optika
from .._materials_test import AbstractTestAbstractStern1994BackilluminatedCCDMaterial


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.E2VCCDAIAMaterial(),
    ],
)
class TestE2VCCD9AIAMaterial(
    AbstractTestAbstractStern1994BackilluminatedCCDMaterial,
):
    pass
