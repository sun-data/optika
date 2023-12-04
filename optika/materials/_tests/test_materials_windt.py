import pytest
import optika
from .test_materials import AbstractTestAbstractMaterial


class AbstractTestAbstractWindtMaterial(
    AbstractTestAbstractMaterial,
):
    pass


@pytest.mark.parametrize("a", [optika.materials.Silicon()])
class TestSilicon(
    AbstractTestAbstractWindtMaterial,
):
    pass
