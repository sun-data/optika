import pytest
import astropy.units as u
import optika
from optika._tests import test_mixins


class AbstractTestAbstractMesh(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestShaped,
):
    def test_chemical(self, a: optika.materials.meshes.AbstractMesh):
        result = a.chemical
        assert isinstance(result, (str, optika.chemicals.AbstractChemical))

    def test_efficiency(self, a: optika.materials.meshes.AbstractMesh):
        result = a.efficiency
        assert result > 0

    def test_pitch(self, a: optika.materials.meshes.AbstractMesh):
        result = a.pitch
        assert result > (0 / u.mm)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.meshes.Mesh(
            chemical="Ni",
            efficiency=0.8,
            pitch=70 / u.mm,
        )
    ],
)
class TestMesh(
    AbstractTestAbstractMesh,
):
    pass
