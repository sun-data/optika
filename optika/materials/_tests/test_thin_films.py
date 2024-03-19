import pytest
import astropy.units as u
import optika
from . import test_multilayers


class AbstractTestAbstractThinFilmFilter(
    test_multilayers.AbstractTestAbstractMultilayerFilm,
):

    def test_layer(self, a: optika.materials.AbstractThinFilmFilter):
        result = a.layer
        assert isinstance(result, optika.materials.AbstractLayer)

    def test_layer_oxide(self, a: optika.materials.AbstractThinFilmFilter):
        result = a.layer_oxide
        assert isinstance(result, optika.materials.AbstractLayer)

    def test_mesh(self, a: optika.materials.AbstractThinFilmFilter):
        result = a.mesh
        assert isinstance(result, optika.materials.meshes.AbstractMesh)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.ThinFilmFilter(
            layer=optika.materials.Layer(
                chemical="Al",
                thickness=100 * u.nm,
            ),
            layer_oxide=optika.materials.Layer(
                chemical="Al2O3",
                thickness=2 * u.nm,
            ),
            mesh=optika.materials.meshes.Mesh(
                chemical="Ni",
                efficiency=0.8,
                pitch=70 / u.mm,
            ),
        )
    ],
)
class TestThinFilmFilter(
    AbstractTestAbstractThinFilmFilter,
):
    pass
