import pytest
import numpy as np
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

    def test_object_is_at_infinity(self, a: optika.systems.AbstractSequentialSystem):
        result = a.object_is_at_infinity
        assert isinstance(result, bool)

    def test_surfaces(self, a: optika.systems.AbstractSequentialSystem):
        for surface in a.surfaces:
            assert isinstance(surface, optika.surfaces.AbstractSurface)

    def test_sensor(self, a: optika.systems.AbstractSequentialSystem):
        if a.sensor is not None:
            assert isinstance(a.sensor, optika.sensors.AbstractImagingSensor)

    def test_axis_surface(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.axis_surface, str)

    def test_surfaces_all(self, a: optika.systems.AbstractSequentialSystem):
        for surface in a.surfaces_all:
            assert isinstance(surface, optika.surfaces.AbstractSurface)

    def test_grid_input(self, a: optika.systems.AbstractSequentialSystem):
        assert isinstance(a.grid_input, optika.vectors.ObjectVectorArray)

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

    def test_rayfunction_stops(self, a: optika.systems.AbstractSequentialSystem):
        result = a.rayfunction_stops
        assert isinstance(result, optika.rays.RayFunctionArray)
        assert isinstance(result.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(result.outputs, optika.rays.RayVectorArray)
        assert result.ndim >= 2

    def test_field_min(self, a: optika.systems.AbstractSequentialSystem):
        result = a.field_min
        assert isinstance(result, na.AbstractCartesian2dVectorArray)

    def test_field_max(self, a: optika.systems.AbstractSequentialSystem):
        result = a.field_max
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.all(result > a.field_min)

    def test_pupil_min(self, a: optika.systems.AbstractSequentialSystem):
        result = a.pupil_min
        assert isinstance(result, na.AbstractCartesian2dVectorArray)

    def test_pupil_max(self, a: optika.systems.AbstractSequentialSystem):
        result = a.pupil_max
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.all(result > a.pupil_min)

    @pytest.mark.parametrize(
        argnames="wavelength,field,pupil",
        argvalues=[
            (
                None,
                None,
                None,
            ),
            (
                500 * u.nm,
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                    num=11,
                ),
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                    num=11,
                ),
            ),
        ],
    )
    def test_raytrace(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
    ):
        raytrace = a.raytrace(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
        )
        assert isinstance(raytrace, optika.rays.RayFunctionArray)
        assert isinstance(raytrace.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(raytrace.outputs, optika.rays.RayVectorArray)
        assert a.axis_surface in raytrace.shape

    @pytest.mark.parametrize(
        argnames="wavelength,field,pupil",
        argvalues=[
            (
                None,
                None,
                None,
            ),
            (
                500 * u.nm,
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                    num=11,
                ),
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                    num=11,
                ),
            ),
        ],
    )
    def test_rayfunction(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
    ):
        raytrace = a.rayfunction(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
        )
        assert isinstance(raytrace, optika.rays.RayFunctionArray)
        assert isinstance(raytrace.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(raytrace.outputs, optika.rays.RayVectorArray)
        assert a.axis_surface not in raytrace.shape

    def test_rayfunction_default(self, a: optika.systems.AbstractSequentialSystem):
        rayfunction = a.rayfunction_default
        assert isinstance(rayfunction, optika.rays.RayFunctionArray)
        assert isinstance(rayfunction.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(rayfunction.outputs, optika.rays.RayVectorArray)
        assert a.axis_surface not in rayfunction.shape


_objects = [
    None,
    optika.surfaces.Surface(),
    optika.surfaces.Surface(
        aperture=optika.apertures.CircularAperture(10 * u.mm),
    ),
    optika.surfaces.Surface(
        aperture=optika.apertures.CircularAperture(0.1),
    ),
]

_surfaces = [
    optika.surfaces.Surface(
        name="mirror",
        sag=optika.sags.SphericalSag(-200 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.CircularAperture(20 * u.mm),
        is_pupil_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
    ),
]

_sensor = optika.sensors.IdealImagingSensor(
    name="sensor",
    width_pixel=15 * u.um,
    num_pixel=na.Cartesian2dVectorArray(2048, 1024),
    is_field_stop=True,
)

_grid_input = optika.vectors.ObjectVectorArray(
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
)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.systems.SequentialSystem(
            object=obj,
            surfaces=_surfaces,
            sensor=_sensor,
            grid_input=_grid_input,
        )
        for obj in _objects
    ],
)
class TestSequentialSystem(AbstractTestAbstractSequentialSystem):
    pass
