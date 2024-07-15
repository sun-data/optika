import pytest
import matplotlib.axes
import astropy.units as u
import named_arrays as na
import optika
from . import test_mixins
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
    optika.surfaces.Surface(
        name="test_surface",
        sag=optika.sags.SphericalSag(radius=1000 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.RectangularAperture(half_width=10 * u.mm),
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
        rulings=optika.rulings.Rulings(spacing=1 * u.um, diffraction_order=1),
    ),
]


class AbstractTestAbstractSurface(
    test_mixins.AbstractTestPlottable,
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
    test_propagators.AbstractTestAbstractLightPropagator,
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

    def test_is_pupil_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_pupil_stop, bool)

    def test_is_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_stop, bool)

    class TestPlot(
        test_mixins.AbstractTestPlottable.TestPlot,
    ):
        def test_plot(
            self,
            a: optika.surfaces.AbstractSurface,
            ax: None | matplotlib.axes.Axes | na.ScalarArray,
            transformation: None | na.transformations.AbstractTransformation,
        ):
            if a.aperture is not None:
                if na.unit_normalized(a.aperture.wire()).is_equivalent(u.mm):
                    super().test_plot(
                        a=a,
                        ax=ax,
                        transformation=transformation,
                    )


@pytest.mark.parametrize("a", surfaces)
class TestSurface(
    AbstractTestAbstractSurface,
):
    pass
