import pytest
import astropy.units as u
import named_arrays as na
import optika
from . import test_mixins
from . import test_plotting
from . import test_propagators


surfaces = [
    optika.surfaces.Surface(),
    optika.surfaces.Surface(
        name="test_surface",
        sag=optika.sags.SphericalSag(radius=1000 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.RectangularAperture(half_width=10 * u.mm),
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
    ),
]


class AbstractTestAbstractSurface(
    test_plotting.AbstractTestPlottable,
    test_mixins.AbstractTestTransformable,
    test_propagators.AbstractTestAbstractRayPropagator,
):
    def test_name(self, a: optika.surfaces.AbstractSurface):
        if a.name is not None:
            assert isinstance(a.name, str)

    def test_sag(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.sag, optika.sags.AbstractSag)

    def test_material(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.material, optika.materials.AbstractMaterial)

    def test_aperture(self, a: optika.surfaces.AbstractSurface):
        if a.aperture is not None:
            assert isinstance(a.aperture, optika.apertures.AbstractAperture)

    def test_aperture_mechanical(self, a: optika.surfaces.AbstractSurface):
        if a.aperture_mechanical is not None:
            assert isinstance(a.aperture_mechanical, optika.apertures.AbstractAperture)

    def test_rulings(self, a: optika.surfaces.AbstractSurface):
        if a.rulings is not None:
            assert isinstance(a.rulings, optika.rulings.AbstractRulings)

    def test_is_field_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_field_stop, bool)

    def test_is_aperture_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_aperture_stop, bool)

    def test_is_spectral_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_spectral_stop, bool)

    def test_is_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_stop, bool)


@pytest.mark.parametrize("a", surfaces)
class TestSurface(
    AbstractTestAbstractSurface,
):
    pass