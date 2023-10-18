import pytest
import astropy.units as u
import named_arrays as na
import optika
from . import test_mixins


class AbstractTestAbstractSystem(
    test_mixins.AbstractTestPlottable,
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
):
    pass


class AbstractTestAbstractSequentialSystem(
    AbstractTestAbstractSystem,
):
    def test_object(self, a: optika.systems.AbstractSequentialSystem):
        if a.object is not None:
            assert isinstance(a.object, optika.surfaces.AbstractSurface)

    def test_surfaces(self, a: optika.systems.AbstractSequentialSystem):
        for surface in a.surfaces:
            assert isinstance(surface, optika.surfaces.AbstractSurface)

    def test_axis_surface(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.axis_surface, str)

    def test_surfaces_all(self, a: optika.systems.AbstractSequentialSystem):
        for surface in a.surfaces_all:
            assert isinstance(surface, optika.surfaces.AbstractSurface)

    def grid_input_normalized(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.grid_input_normalized, optika.vectors.ObjectVectorArray)

    def test_index_field_stop(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.index_field_stop, int)
        assert a.surfaces_all[a.index_field_stop].is_field_stop

    def test_index_pupil_stop(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.index_pupil_stop, int)
        assert a.surfaces_all[a.index_pupil_stop].is_pupil_stop

    def test_field_stop(self, a: optika.systems.AbstractSequentialSystem):
        assert a.field_stop.is_field_stop

    def test_pupil_stop(self, a: optika.systems.AbstractSequentialSystem):
        assert a.pupil_stop.is_pupil_stop

    def test_raytrace(self, a: optika.systems.AbstractSequentialSystem):
        raytrace = a.raytrace
        assert isinstance(raytrace, optika.rays.RayFunctionArray)
        assert isinstance(raytrace.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(raytrace.outputs, optika.rays.RayVectorArray)
        assert a.axis_surface in raytrace.shape

    def test_rayfunction(self, a: optika.systems.AbstractSequentialSystem):
        rayfunction = a.rayfunction
        assert isinstance(rayfunction, optika.rays.RayFunctionArray)
        assert isinstance(rayfunction.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(rayfunction.outputs, optika.rays.RayVectorArray)
        assert a.axis_surface not in rayfunction.shape


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.systems.SequentialSystem(
            surfaces=[
                optika.surfaces.Surface(
                    name="mirror",
                    sag=optika.sags.SphericalSag(-200 * u.mm),
                    material=optika.materials.Mirror(),
                    aperture=optika.apertures.CircularAperture(20 * u.mm),
                    is_pupil_stop=True,
                    transformation=na.transformations.Cartesian3dTranslation(
                        z=100 * u.mm
                    ),
                ),
                optika.surfaces.Surface(
                    name="detector",
                    aperture=optika.apertures.RectangularAperture(10 * u.mm),
                    is_field_stop=True,
                ),
            ],
            grid_input_normalized=optika.vectors.ObjectVectorArray(
                wavelength=500 * u.nm,
                field=na.Cartesian2dVectorLinearSpace(
                    start=0,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                    num=5,
                ),
                pupil=na.Cartesian2dVectorLinearSpace(
                    start=0,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                    num=5,
                ),
            ),
        )
    ],
)
class TestSequentialSystem(AbstractTestAbstractSequentialSystem):
    pass
