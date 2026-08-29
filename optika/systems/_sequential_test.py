import dataclasses
import matplotlib.lines
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins
from ._systems_test import AbstractTestAbstractSystem


class AbstractTestAbstractSequentialSystem(
    test_mixins.AbstractTestDxfWritable,
    test_mixins.AbstractTestPlottable,
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

    def test_axis_stops(self, a: optika.systems.AbstractSequentialSystem):
        result = a.axis_stops
        assert result == (a.axis_field_stop, a.axis_pupil_stop)

        # the axes a caller has to name to reduce either outline
        assert set(result).issubset(na.shape(a.field_boundary))
        assert set(result).issubset(na.shape(a.pupil_boundary))

    def test_field_boundary(self, a: optika.systems.AbstractSequentialSystem):
        result = a.field_boundary
        assert isinstance(result, na.AbstractCartesian2dVectorArray)

        # the outline of the field, along the edge of each stop
        assert set(a.axis_stops).issubset(na.shape(result))

        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.deg)
        else:
            assert na.unit(result).is_equivalent(u.m)

    def test_pupil_boundary(self, a: optika.systems.AbstractSequentialSystem):
        result = a.pupil_boundary
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert set(a.axis_stops).issubset(na.shape(result))

        if a.object_is_at_infinity:
            assert na.unit(result).is_equivalent(u.m)
        else:
            assert na.unit(result).is_equivalent(u.deg)

    def test_field_min(self, a: optika.systems.AbstractSequentialSystem):
        result = a.field_min
        assert isinstance(result, na.AbstractCartesian2dVectorArray)

        # the corner of the field is a reduction of its outline
        assert np.all(result == a.field_boundary.min(a.axis_stops))

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

    def test_rayfunction_efficiency(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        """
        Skipping the efficiency of each surface leaves the geometry of the
        rays untouched.
        """
        expected = a.rayfunction()
        result = a.rayfunction(efficiency=False)

        assert np.all(result.outputs.position == expected.outputs.position)
        assert np.all(result.outputs.direction == expected.outputs.direction)
        assert np.all(result.outputs.unvignetted == expected.outputs.unvignetted)

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
    def test_area_effective(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
    ):
        if wavelength is None and not a.axis_wavelength_:
            with pytest.raises(ValueError):
                a.area_effective(
                    wavelength=wavelength,
                    field=field,
                    pupil=pupil,
                )
            return
        result = a.area_effective(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
        )
        assert isinstance(result, optika.radiometry.InterpolatedEffectiveAreaModel)
        if a.object_is_at_infinity:
            assert na.unit(result.area).is_equivalent(u.cm**2)
        else:
            assert na.unit(result.area).is_equivalent(u.deg**2)
        assert np.all(result.area >= 0)

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
    def test_linearize(
        self,
        a: optika.systems.AbstractSequentialSystem,
        wavelength: None | u.Quantity | na.AbstractScalar,
        field: None | na.AbstractCartesian2dVectorArray,
        pupil: None | na.AbstractCartesian2dVectorArray,
    ):
        if wavelength is None and not a.axis_wavelength_:
            with pytest.raises(ValueError):
                a.linearize(wavelength=wavelength, field=field, pupil=pupil)
            return
        result = a.linearize(wavelength=wavelength, field=field, pupil=pupil)
        assert isinstance(result, optika.systems.LinearSystem)
        assert isinstance(result.distortion, optika.distortion.AbstractDistortionModel)
        assert isinstance(result.vignetting, optika.radiometry.AbstractVignettingModel)
        assert isinstance(
            result.area_effective, optika.radiometry.AbstractEffectiveAreaModel
        )
        assert result.sensor is a.sensor
        assert result.field_stop is None

        # `direction` has to be a scalar.  `expose` indexes it by the cell
        # centers of the *scene's* wavelength grid, which is unrelated to the
        # grid linearized here, so an array would either fail to broadcast or
        # silently pair up wavelengths which are not the same.
        assert na.shape(result.direction) == {}

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


@dataclasses.dataclass(eq=False, repr=False)
class _HalfMirror(optika.materials.Mirror):
    """A mirror which reflects half of the light which strikes it."""

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 0.5


def test_rayfunction_efficiency_skipped():
    """
    Skipping the efficiency of each surface leaves the intensity of the rays
    at its input value, instead of the throughput of the system.
    """
    system = optika.systems.SequentialSystem(
        surfaces=[dataclasses.replace(_surfaces[0], material=_HalfMirror())],
        sensor=_sensor,
        grid_input=_grid_input,
    )

    expected = system.rayfunction()
    result = system.rayfunction(efficiency=False)

    assert np.all(expected.outputs.intensity < 1)
    assert np.all(result.outputs.intensity == 1)
    assert np.all(result.outputs.position == expected.outputs.position)
    assert np.all(result.outputs.unvignetted == expected.outputs.unvignetted)


def test_area_effective_ignores_field_outside_the_field_of_view():
    """
    The effective area is averaged over the field of view, so sampling more
    of the field which lies outside it does not change the answer.

    This is what lets the model be multiplied by the vignetting model, which
    normalizes its illumination over that same set of field positions.
    """
    system = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input_wavelength,
    )

    field = na.Cartesian2dVectorLinearSpace(
        start=0,
        stop=1,
        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
        num=5,
    )

    # the same five samples along each axis, plus two which land far enough
    # outside the field stop that no ray through them reaches the sensor
    samples = np.array([-3, 0, 0.25, 0.5, 0.75, 1, 3])
    field_extended = na.Cartesian2dVectorArray(
        x=na.ScalarArray(samples, axes="field_x"),
        y=na.ScalarArray(samples, axes="field_y"),
    )

    result = system.area_effective(field=field)
    result_extended = system.area_effective(field=field_extended)

    # `area_effective` traces at randomly placed cell centers, so two calls
    # differ by a percent or so.  Averaging over the extra field positions
    # instead of ignoring them would halve the result, which this separates
    # comfortably.
    assert np.allclose(result_extended.area, result.area, rtol=0.05)


def _system_vignetted() -> optika.systems.SequentialSystem:
    """
    A system whose field stop is a circle rather than the sensor.

    The normalized field grid is the bounding box of the field of view, so a
    field stop shaped like the sensor fills it and nothing is vignetted.  A
    round one leaves the corners dark, which is what a system like ESIS
    actually looks like and what makes the vignetting model do any work.
    """
    surfaces = [
        optika.surfaces.Surface(
            name="mirror",
            sag=optika.sags.SphericalSag(-200 * u.mm),
            material=optika.materials.Mirror(),
            aperture=optika.apertures.CircularAperture(20 * u.mm),
            is_pupil_stop=True,
            transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
        ),
        optika.surfaces.Surface(
            name="field stop",
            aperture=optika.apertures.CircularAperture(0.96 * u.mm),
            is_field_stop=True,
            transformation=na.transformations.Cartesian3dTranslation(z=2 * u.mm),
        ),
    ]
    sensor = optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=15 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        timedelta_exposure=1 * u.s,
        num_pixel=na.Cartesian2dVectorArray(128, 128),
        transformation=na.transformations.Cartesian3dTranslation(z=1 * u.mm),
    )
    grid = optika.vectors.ObjectVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
        field=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=11,
        ),
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=11,
        ),
    )
    return optika.systems.SequentialSystem(
        surfaces=surfaces,
        sensor=sensor,
        grid_input=grid,
    )


def test_linearize_conserves_flux():
    """
    A flat field through the linearized system collects the same number of
    electrons as the same flat field traced through the system it came from.

    This is the end-to-end statement of what `linearize` is for, and the one
    thing that exercises the distortion, vignetting, and effective-area
    models against each other rather than one at a time.
    """
    system = _system_vignetted()

    # a uniform scene covering the inner half of the field of view, kept away
    # from the edge where the polynomial vignetting model cannot follow the
    # hard cutoff of the field stop
    center = (system.field_max + system.field_min) / 2
    half = (system.field_max - system.field_min) / 4
    num = 12
    scene = na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=4) * u.nm,
            position=na.Cartesian2dVectorArray(
                x=na.linspace(
                    center.x - half.x, center.x + half.x, axis="field_x", num=num + 1
                ),
                y=na.linspace(
                    center.y - half.y, center.y + half.y, axis="field_y", num=num + 1
                ),
            ),
        ),
        outputs=1e3
        * u.photon
        / u.s
        / u.cm**2
        / u.arcsec**2
        / u.nm
        * na.ScalarArray(np.ones((num, num)), axes=("field_x", "field_y")),
    )

    expected = system.image(scene, noise=False).outputs.sum()
    result = system.linearize(degree=2).image(scene, noise=False).outputs.sum()

    # the two discretize the problem differently, and `area_effective` traces
    # at randomly placed pupil cell centers, so they agree to a few percent
    # rather than exactly.  An average of the effective area taken over a
    # different set of field positions than the vignetting model is
    # normalized over would put this near 0.56, which the tolerance excludes.
    assert np.allclose(result, expected, rtol=0.15)


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


def test_plot_unit():
    """
    The whole system is drawn in the unit asked for, rays included.

    :func:`astropy.visualization.quantity_support` reconciles units on a 2D
    axes but not on a 3D one, where a part described in microns is drawn a
    thousand times larger than the millimeters around it.
    """

    def extent(unit: u.UnitBase) -> float:
        """The largest magnitude handed to matplotlib, surfaces and rays alike."""
        fig, ax = plt.subplots()
        try:
            _system_newtonian.plot(ax=ax, components=("z", "x"), unit=unit)
            x = np.concatenate(
                [
                    np.asarray(getattr(line.get_xdata(), "value", line.get_xdata()))
                    for line in ax.lines
                ]
            )
            return float(np.max(np.abs(x)))
        finally:
            plt.close(fig)

    assert extent(u.um) == pytest.approx(1000 * extent(u.mm))


def test_plot_kwargs_plot():
    """A system can carry the keywords its surfaces are to be drawn with."""
    color = "tab:red"
    system = dataclasses.replace(
        _system_newtonian,
        kwargs_plot=dict(color=color),
    )

    fig, ax = plt.subplots()
    system.plot(ax=ax, components=("z", "x"), plot_rays=False)
    colors = [line.get_color() for line in ax.lines]
    plt.close(fig)

    assert colors
    assert all(c == color for c in colors)


def test_field_stop_default():
    """
    With no surface marked as the field stop, the first one becomes it.

    A system needs a field stop to define its field of view, so rather than
    refuse to trace a system which does not name one, the object surface is
    taken to be it.
    """
    system = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=dataclasses.replace(_sensor, is_field_stop=False),
        grid_input=_grid_input,
    )

    assert system.index_field_stop == 0
    assert system.surfaces_all[0].is_field_stop


def test_index_pupil_stop_undefined():
    """
    A system which names no pupil stop cannot say where its pupil is.

    Unlike the field stop there is no sensible surface to fall back on, since
    the pupil is a property of the optics rather than of the frame.
    """
    system = optika.systems.SequentialSystem(
        surfaces=[dataclasses.replace(_surfaces[0], is_pupil_stop=False)],
        sensor=_sensor,
        grid_input=_grid_input,
    )

    with pytest.raises(ValueError, match="Pupil stop is not defined"):
        system.index_pupil_stop


_object_translated = optika.surfaces.Surface(
    aperture=optika.apertures.CircularAperture(10 * u.mm),
    transformation=na.transformations.Cartesian3dTranslation(z=-10 * u.mm),
)
"""An object surface placed away from the origin of the system."""


def test_object_transformation():
    """
    The rays start on the object surface wherever that surface has been put.

    The object carries its own transformation, like any other surface, and the
    rays are given in the coordinates of the system rather than of the object.
    """
    system = optika.systems.SequentialSystem(
        object=_object_translated,
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input,
    )

    raytrace = system.raytrace(axis="surface")
    z = raytrace.outputs.position.z[dict(surface=0)]

    assert np.all(z == _object_translated.transformation.z)


def test_stops_afocal():
    """
    A system whose image is at infinity is solved in angle at the far stop.

    The stop nearer the object is measured in millimeters and the one further
    away in direction cosines, so the rays are matched to the second stop by
    the direction they leave in rather than by where they land.
    """
    system = optika.systems.SequentialSystem(
        object=optika.surfaces.Surface(
            aperture=optika.apertures.CircularAperture(1 * u.mm),
            is_pupil_stop=True,
        ),
        surfaces=[
            optika.surfaces.Surface(
                name="exit",
                aperture=optika.apertures.CircularAperture(0.05),
                is_field_stop=True,
                transformation=na.transformations.Cartesian3dTranslation(
                    z=100 * u.mm,
                ),
            ),
        ],
        grid_input=_grid_input,
    )

    result = system.rayfunction_stops

    # the rays leave at every angle the far stop admits, and from everywhere
    # on the near one
    direction = np.max(np.abs(result.outputs.direction.x))
    position = np.max(np.abs(result.outputs.position.x))

    assert float(na.value(direction).ndarray) == pytest.approx(0.05)
    assert float(na.value(position.to(u.mm)).ndarray) == pytest.approx(1)


def test_plot_rays_3d_is_a_collection():
    """
    On a 3D axes the rays are drawn as collections, one per segment.

    A line is not sorted into a 3D scene at all: it keeps the zorder it was
    given, and at the default that is below every filled surface, so a beam
    disappears behind the first optic it crosses instead of reaching it.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    result = _system_newtonian.plot(
        ax=ax,
        components=("z", "x", "y"),
        plot_rays=True,
    )

    rays = [a for a in np.atleast_1d(result["rays"].ndarray).flat if a is not None]
    plt.close(fig)

    assert rays
    for artist in rays:
        assert isinstance(artist, mpl_toolkits.mplot3d.art3d.Line3DCollection)

    # a segment for every gap between surfaces
    axis = _system_newtonian.axis_surface
    assert na.shape(result["rays"])[axis] == len(_system_newtonian.surfaces_all) - 1


def test_plot_rays_2d_is_a_line():
    """On a 2D axes the rays are still drawn as ordinary lines."""
    fig, ax = plt.subplots()

    result = _system_newtonian.plot(
        ax=ax,
        components=("z", "x"),
        plot_rays=True,
    )

    rays = [a for a in np.atleast_1d(result["rays"].ndarray).flat if a is not None]
    plt.close(fig)

    assert rays
    for artist in rays:
        assert isinstance(artist, matplotlib.lines.Line2D)
