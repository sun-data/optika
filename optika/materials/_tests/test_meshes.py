import pytest
import abc
import astropy.units as u
import optika


class AbstractTestAbstractMesh(
    abc.ABC,
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
