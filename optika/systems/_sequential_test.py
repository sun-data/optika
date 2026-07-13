import matplotlib.pyplot as plt
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins
from ._systems_test import AbstractTestAbstractSystem


class AbstractTestAbstractSequentialSystem(
    test_mixins.AbstractTestDxfWritable,
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

    def test_axis_wavelength(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_wavelength
        if result is not None:
            assert isinstance(result, str)

    def test_axis_field(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_field
        if result is not None:
            assert len(result) == 2
            for axis in result:
                assert isinstance(axis, str)

    def test_axis_pupil(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_pupil
        if result is not None:
            assert len(result) == 2
            for axis in result:
                assert isinstance(axis, str)

    def test_axis_wavelength_(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_wavelength_
        assert isinstance(result, tuple)
        assert len(result) <= 1
        for axis in result:
            assert isinstance(axis, str)
            assert axis in a.grid_input.wavelength.shape
        if a.axis_wavelength is not None:
            assert result == (a.axis_wavelength,)

    def test_axis_field_(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_field_
        assert len(result) == 2
        for axis in result:
            assert isinstance(axis, str)
            assert axis in a.grid_input.field.shape
        if a.axis_field is not None:
            assert result == a.axis_field
        assert not set(result) & set(a.axis_wavelength_)

    def test_axis_pupil_(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_pupil_
        assert len(result) == 2
        for axis in result:
            assert isinstance(axis, str)
            assert axis in a.grid_input.pupil.shape
        if a.axis_pupil is not None:
            assert result == a.axis_pupil
        assert not set(result) & set(a.axis_wavelength_)
        assert not set(result) & set(a.axis_field_)

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
        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.deg)
        else:
            assert na.unit(result).is_equivalent(u.m)

    def test_field_max(self, a: optika.systems.AbstractSequentialSystem):
        result = a.field_max
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.all(result > a.field_min)
        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.deg)
        else:
            assert na.unit(result).is_equivalent(u.m)

    def test_pupil_min(self, a: optika.systems.AbstractSequentialSystem):
        result = a.pupil_min
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.m)
        else:
            assert na.unit(result).is_equivalent(u.deg)

    def test_pupil_max(self, a: optika.systems.AbstractSequentialSystem):
        result = a.pupil_max
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.all(result > a.pupil_min)
        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.m)
        else:
            assert na.unit(result).is_equivalent(u.deg)

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
    @pytest.mark.parametrize("accumulate", [True, False])
    def test_raytrace(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
        accumulate: bool,
    ):
        raytrace = a.raytrace(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            accumulate=accumulate,
        )
        assert isinstance(raytrace, optika.rays.RayFunctionArray)
        assert isinstance(raytrace.inputs, optika.vectors.ObjectVectorArray)
        assert isinstance(raytrace.outputs, optika.rays.RayVectorArray)
        if accumulate:
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

    @pytest.mark.parametrize(
        argnames="wavelength,field,pupil",
        argvalues=[
            (
                None,
                None,
                None,
            ),
            (
                na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                    num=5,
                ),
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                    num=5,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("degree", [1, 2])
    def test_distortion(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
        degree: int,
    ):
        if wavelength is None and not a.axis_wavelength_:
            with pytest.raises(ValueError):
                a.distortion(
                    wavelength=wavelength,
                    field=field,
                    pupil=pupil,
                    degree=degree,
                )
            return
        result = a.distortion(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            degree=degree,
        )
        assert isinstance(result, optika.distortion.PolynomialDistortionModel)
        assert result.degree == degree

    @pytest.mark.parametrize(
        argnames="wavelength,field,pupil",
        argvalues=[
            (
                None,
                None,
                None,
            ),
            (
                na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                    num=5,
                ),
                na.Cartesian2dVectorLinearSpace(
                    start=-1,
                    stop=1,
                    axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                    num=5,
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("degree", [1, 2])
    def test_vignetting(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
        degree: int,
    ):
        if wavelength is None and not a.axis_wavelength_:
            with pytest.raises(ValueError):
                a.vignetting(
                    wavelength=wavelength,
                    field=field,
                    pupil=pupil,
                    degree=degree,
                )
            return
        result = a.vignetting(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            degree=degree,
        )
        assert isinstance(result, optika.radiometry.PolynomialVignettingModel)
        assert result.degree == degree
        assert np.all(result.illumination >= 0)
        mean = np.mean(
            result.illumination,
            axis=result.axis_field,
            where=result.where,
        )
        assert np.allclose(mean, 1)

    def test_spot_diagram(self, a: optika.systems.AbstractSequentialSystem):
        fig, axs = a.spot_diagram()
        assert isinstance(fig, plt.Figure)

        for ax in axs.ndarray.flat:
            assert isinstance(ax, plt.Axes)
            assert ax.has_data()


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

_transformations = [
    None,
    None,
    na.transformations.Cartesian3dTranslation(x=100 * u.mm),
    na.transformations.Cartesian3dRotationZ(23 * u.deg),
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

_sensor = optika.sensors.ImagingSensor(
    name="sensor",
    width_pixel=15 * u.um,
    axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
    timedelta_exposure=1 * u.s,
    num_pixel=na.Cartesian2dVectorArray(128, 128),
    transformation=na.transformations.Cartesian3dTranslation(z=1 * u.mm),
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

_grid_input_wavelength = optika.vectors.ObjectVectorArray(
    wavelength=na.linspace(
        start=500 * u.nm,
        stop=600 * u.nm,
        axis="wavelength",
        num=3,
    ),
    field=_grid_input.field,
    pupil=_grid_input.pupil,
)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.systems.SequentialSystem(
            object=obj,
            surfaces=_surfaces,
            sensor=_sensor,
            grid_input=_grid_input,
            transformation=transform,
        )
        for obj, transform in zip(_objects, _transformations)
    ]
    + [
        optika.systems.SequentialSystem(
            surfaces=_surfaces,
            sensor=_sensor,
            grid_input=_grid_input_wavelength,
        ),
        optika.systems.SequentialSystem(
            surfaces=_surfaces,
            sensor=_sensor,
            grid_input=_grid_input_wavelength,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            axis_pupil=("pupil_x", "pupil_y"),
        ),
    ],
)
class TestSequentialSystem(AbstractTestAbstractSequentialSystem):
    pass


def test__anchor_surface():
    first = optika.surfaces.Surface(name="first")
    last = optika.surfaces.Surface(name="last")
    mirror = optika.surfaces.Surface(
        name="mirror",
        material=optika.materials.Mirror(),
    )
    curved = optika.surfaces.Surface(
        name="curved",
        sag=optika.sags.SphericalSag(radius=-100 * u.mm),
    )
    grating = optika.surfaces.Surface(
        name="grating",
        rulings=optika.rulings.Rulings(spacing=1 * u.um, diffraction_order=1),
    )
    flat = optika.surfaces.Surface(name="flat")

    anchor = optika.systems.SequentialSystem._anchor_surface
    assert anchor([first, flat, mirror, last]) is mirror
    assert anchor([first, curved, last]) is curved
    assert anchor([first, grating, last]) is grating
    assert anchor([first, flat, last]) is last


# small enough that the image of the field fits on the sensor
_radius_field_newtonian = 0.05 * u.deg

_system_newtonian = optika.systems.SequentialSystem(
    object=optika.surfaces.Surface(
        name="source",
        aperture=optika.apertures.CircularAperture(
            radius=np.sin(_radius_field_newtonian),
        ),
        is_field_stop=True,
    ),
    surfaces=[
        optika.surfaces.Surface(
            name="primary",
            sag=optika.sags.SphericalSag(radius=-2000 * u.mm),
            material=optika.materials.Mirror(),
            aperture=optika.apertures.CircularAperture(radius=50 * u.mm),
            transformation=na.transformations.Cartesian3dTranslation(
                z=500 * u.mm,
            ),
        ),
        optika.surfaces.Surface(
            name="aperture",
            aperture=optika.apertures.CircularAperture(radius=10 * u.mm),
            transformation=na.transformations.Cartesian3dTranslation(
                z=250 * u.mm,
            ),
            is_pupil_stop=True,
        ),
    ],
    sensor=optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=15 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        timedelta_exposure=1 * u.s,
        num_pixel=na.Cartesian2dVectorArray(128, 128),
        transformation=na.transformations.Cartesian3dTranslation(
            z=-500 * u.mm,
        ),
    ),
    grid_input=_grid_input,
)


@pytest.mark.parametrize(argnames="a", argvalues=[_system_newtonian])
class TestSequentialSystemNewtonian(
    AbstractTestAbstractSequentialSystem,
):
    """
    A Newtonian-style telescope where the pupil stop is downstream of the
    primary mirror, so that the initial guess of the stop root-finding
    problem must be aimed at the center of the primary instead of directly
    at its own target on the pupil stop.
    """

    def test_field_max_matches_source_aperture(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        result = a.field_max
        assert np.abs(result.x - _radius_field_newtonian) < 1e-6 * u.deg
        assert np.abs(result.y - _radius_field_newtonian) < 1e-6 * u.deg


_radius_field_grazing = 0.25 * u.deg

_system_grazing = optika.systems.SequentialSystem(
    object=optika.surfaces.Surface(
        name="source",
        aperture=optika.apertures.CircularAperture(
            radius=np.sin(_radius_field_grazing),
        ),
        is_field_stop=True,
    ),
    surfaces=[
        optika.surfaces.Surface(
            name="paraboloid",
            sag=optika.sags.ParabolicSag(focal_length=-2000 * u.mm),
            material=optika.materials.Mirror(),
            aperture=optika.apertures.CircularAperture(radius=260 * u.mm),
            transformation=na.transformations.Cartesian3dTranslation(
                z=2500 * u.mm,
            ),
            is_pupil_stop=True,
        ),
        optika.surfaces.Surface(
            name="grating",
            rulings=optika.rulings.Rulings(
                spacing=10 * u.um,
                diffraction_order=1,
            ),
            aperture=optika.apertures.RectangularAperture(
                half_width=60 * u.mm,
            ),
            transformation=na.transformations.Cartesian3dTranslation(
                z=1000 * u.mm,
            ),
        ),
    ],
    sensor=optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=15 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        # short exposure so that the Poisson lam stays representable for the
        # large collecting area of the grazing primary
        timedelta_exposure=1 * u.us,
        num_pixel=na.Cartesian2dVectorArray(2048, 1024),
        # offset by the deflection of the first diffraction order,
        # (z_grating - z_sensor) * wavelength / spacing
        transformation=na.transformations.Cartesian3dTranslation(
            x=26 * u.mm,
            z=480 * u.mm,
        ),
    ),
    grid_input=_grid_input,
)


@pytest.mark.parametrize(argnames="a", argvalues=[_system_grazing])
class TestSequentialSystemGrazingSpectrograph(
    AbstractTestAbstractSequentialSystem,
):
    """
    A grazing-incidence spectrograph with a transmission grating, where the
    object surface (with an angular aperture) is the field stop. This guards
    against regressions in the object-as-field-stop code path of the stop
    root-finding problem.
    """

    def test_field_max_matches_source_aperture(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        result = a.field_max
        assert np.abs(result.x - _radius_field_grazing) < 1e-6 * u.deg
        assert np.abs(result.y - _radius_field_grazing) < 1e-6 * u.deg
