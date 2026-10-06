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

_motion_rigid = na.transformations.compose(
    na.transformations.Cartesian3dTranslation(
        x=7 * u.mm,
        y=-13 * u.mm,
        z=29 * u.mm,
    ),
    na.transformations.compose(
        na.transformations.Cartesian3dRotationZ(23 * u.deg),
        na.transformations.Cartesian3dRotationY(17 * u.deg),
    ),
)
"""
A rigid motion used to describe a system in a different frame.

The angles and distances are deliberately unrounded, so that a frame error
cannot hide behind a symmetry of the system it is applied to, and moderate,
so that no system is carried near the pole of the chart the launch solve
parametrizes a ray direction on.
"""


def _moved_rigidly(
    system: optika.systems.AbstractSequentialSystem,
    motion: na.transformations.AbstractTransformation,
) -> optika.systems.AbstractSequentialSystem:
    """
    Describe `system` in a frame moved by `motion`.

    Every surface moves together, so this is a relabeling of the same
    instrument rather than a different instrument, and every result of the
    system must survive it unchanged.
    :attr:`~optika.systems.AbstractSequentialSystem.transformation` already
    carries onto every surface except the object, so moving the object is all
    that is left to do.
    """
    obj = system.object
    if obj is not None:
        obj = dataclasses.replace(
            obj,
            transformation=na.transformations.compose(motion, obj.transformation),
        )
    return dataclasses.replace(
        system,
        object=obj,
        transformation=na.transformations.compose(motion, system.transformation),
    )


_grid_field_rigid = na.Cartesian2dVectorLinearSpace(
    start=-0.9,
    stop=0.9,
    axis=na.Cartesian2dVectorArray("_rigid_field_x", "_rigid_field_y"),
    num=4,
)
"""
The normalized field grid the rigid-motion tests trace.

It stops short of the edge of the field so that no ray lands exactly on the
boundary of an aperture, where the roundoff a rigid motion introduces would
decide which side of it the ray falls on.
"""

_grid_pupil_rigid = dataclasses.replace(
    _grid_field_rigid,
    axis=na.Cartesian2dVectorArray("_rigid_pupil_x", "_rigid_pupil_y"),
)
"""
The normalized pupil grid the rigid-motion tests trace.

The same grid as :obj:`_grid_field_rigid` on its own pair of axes, so that the
two sweep a four-dimensional grid instead of being broadcast against each
other.
"""


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

    def test_rayfunction_is_invariant_under_a_rigid_motion(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        """
        Moving every surface of a system together describes the same
        instrument in a different frame, so :meth:`rayfunction` must return
        what it did before.

        Specifically: the field and the pupil it was traced at, the position
        and the direction of every ray, and which rays were vignetted. Each is
        measured in a frame the system carries with it, the first two in the
        frame of the object and the rays in the frame of the sensor, so any of
        them measured in the global frame instead moves with the motion, and
        any solve anchored to the global frame finds a different answer. That
        exercises every frame the system works in at once.

        It is an invariance rather than a system built at an angle on purpose,
        since tilting one surface of a system changes the instrument and can
        quietly stop any light reaching the sensor, which no amount of broken
        frame handling would then make worse.

        What this does not cover: :meth:`raytrace`, whose output is global and
        so moves with the motion rather than staying put, though it is pinned
        indirectly, since the sensor frame these rays are in is the moved
        sensor's; and the three models fit from these rays, which
        :func:`test_models_are_invariant_under_a_rigid_motion` takes.
        """
        b = _moved_rigidly(a, _motion_rigid)

        kwargs = dict(
            field=_grid_field_rigid,
            pupil=_grid_pupil_rigid,
        )
        expected = a.rayfunction(**kwargs)
        result = b.rayfunction(**kwargs)

        # the field and the pupil are denormalized against the stops, which
        # the system solves for in the frame of its object surface, and the
        # rays are measured in the frame of the sensor
        for r, e in [
            (result.inputs.field, expected.inputs.field),
            (result.inputs.pupil, expected.inputs.pupil),
            (result.outputs.position, expected.outputs.position),
            (result.outputs.direction, expected.outputs.direction),
        ]:
            error = (r - e).length.max()
            assert error < 1e-6 * e.length.max()

        # no ray sits on the edge of an aperture, so the motion cannot move
        # one across it and every ray must be vignetted exactly as before
        assert np.all(result.outputs.unvignetted == expected.outputs.unvignetted)

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
    na.transformations.Cartesian3dTranslation(x=1 * u.mm),
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


def test_transformation_moves_optics_not_object():
    """
    The system transformation repositions the optics relative to a fixed
    object, so it changes the raytrace instead of being a rigid relabel of the
    whole system that leaves the image unmoved.
    """
    base = _system_newtonian
    shift = 10 * u.mm
    moved = dataclasses.replace(
        base,
        transformation=na.transformations.Cartesian3dTranslation(x=shift),
    )

    axis = base.axis_surface
    kwargs = dict(
        wavelength=500 * u.nm,
        field=na.Cartesian2dVectorArray(0, 0),
        pupil=na.Cartesian2dVectorArray(0, 0),
        accumulate=True,
    )
    sensor_x_base = base.raytrace(**kwargs).outputs.position[{axis: ~0}].x
    sensor_x_moved = moved.raytrace(**kwargs).outputs.position[{axis: ~0}].x

    # the image moves with the optics by the amount of the system translation
    assert not np.allclose(sensor_x_base.ndarray, sensor_x_moved.ndarray)
    assert np.allclose((sensor_x_moved - sensor_x_base).ndarray, shift)

    # the object surface is left in its own frame, untouched by the transform
    assert moved.surfaces_all[0].transformation == base.surfaces_all[0].transformation


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

    # `field` holds cell vertices, so these bound ten cells along each axis
    vertices = np.linspace(0, 1, num=11)
    field = na.Cartesian2dVectorArray(
        x=na.ScalarArray(vertices, axes="field_x"),
        y=na.ScalarArray(vertices, axes="field_y"),
    )

    # the same ten cells, plus four along each axis which lie beyond the
    # normalized field entirely, so no ray through them reaches the sensor
    vertices_extended = np.concatenate([vertices, [1.5, 2, 2.5, 3]])
    field_extended = na.Cartesian2dVectorArray(
        x=na.ScalarArray(vertices_extended, axes="field_x"),
        y=na.ScalarArray(vertices_extended, axes="field_y"),
    )

    result = system.area_effective(field=field)
    result_extended = system.area_effective(field=field_extended)

    # `area_effective` samples a random point inside every cell, so two calls
    # differ by about a percent at this resolution.  Averaging over the dark
    # cells instead of ignoring them would leave the result 49% low, which
    # this separates comfortably.
    assert np.allclose(result_extended.area, result.area, rtol=0.1)


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


def test_area_effective_without_any_illuminated_field():
    """
    A wavelength which no sampled field position admits has no effective
    area, rather than the undefined average of an empty set.

    Averaging with `where` over nothing gives `nan`, which would carry
    silently into every image the resulting model produces.
    """
    system = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input_wavelength,
    )

    # cells lying entirely beyond the normalized field, so nothing gets through
    vertices = np.array([5, 6, 7])
    field = na.Cartesian2dVectorArray(
        x=na.ScalarArray(vertices, axes="field_x"),
        y=na.ScalarArray(vertices, axes="field_y"),
    )

    result = system.area_effective(field=field)

    assert np.all(np.isfinite(result.area))
    assert np.all(result.area == 0 * result.area.unit)


def test_linearize_keeps_axes_it_is_not_centering():
    """
    Only the two field axes and the two pupil axes are collapsed from
    vertices to centers.

    A grid may carry axes of its own, such as one of ``system.shape``, and
    those have to survive.  Taking the axes to center from the shape of the
    grid rather than naming them would average such an axis into itself.
    """
    system = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input_wavelength,
    )

    num = 5
    wavelength = _grid_input_wavelength.wavelength
    num_wavelength = na.shape(wavelength)["wavelength"]

    # a field grid which drifts with wavelength, so that it carries an axis
    # which is not one of the two being centered
    drift = 1e-6 * na.linspace(0, 1, axis="wavelength", num=num_wavelength)
    field = na.Cartesian2dVectorArray(
        x=na.linspace(0, 1, axis="field_x", num=num) + drift,
        y=na.linspace(0, 1, axis="field_y", num=num) + drift,
    )
    pupil = na.Cartesian2dVectorArray(
        x=na.linspace(-1, 1, axis="pupil_x", num=num),
        y=na.linspace(-1, 1, axis="pupil_y", num=num),
    )

    result = system.linearize(
        wavelength=wavelength,
        field=field,
        pupil=pupil,
        degree=1,
    )

    # the scene the distortion is fit to is defined on the cell centers, so
    # each field axis loses exactly one sample and the wavelength axis, which
    # is not being centered, keeps all of its own
    shape = na.shape(result.distortion.coordinates_scene.position)
    assert shape["field_x"] == num - 1
    assert shape["field_y"] == num - 1
    assert shape["wavelength"] == num_wavelength


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


def _system_wolter() -> optika.systems.SequentialSystem:
    """
    A Wolter-I telescope sampled in polar pupil coordinates.

    Its entrance pupil is a thin annulus, the projection of a 200 mm shell at
    a quarter of a degree of graze, which covers 4% of its own box: in
    rectangular coordinates 24 rays in 25 miss the pupil stop, and in polar
    coordinates none do.
    """
    width_pixel = 10 * u.um
    focal_length = (width_pixel / (1 * u.arcsec)).to(
        u.mm,
        equivalencies=u.dimensionless_angles(),
    )
    radius_aperture = 72 * u.mm
    angle_graze = 0.25 * np.arctan(radius_aperture / focal_length)
    focus_parabola = (radius_aperture / np.tan(2 * angle_graze)).to(u.mm)
    f_parabola = (radius_aperture**2 / (4 * focus_parabola)).to(u.mm)
    halfwidth = (200 * u.mm * np.tan(angle_graze) / 2).to(u.mm)

    z_parabola_focus = focus_parabola - f_parabola
    z_center = (focal_length + z_parabola_focus) / 2
    c_hyp = (z_parabola_focus - focal_length) / 2
    z_intercept = 500 * u.mm
    radius_intercept = radius_aperture * (z_parabola_focus - z_intercept)
    radius_intercept = radius_intercept / z_parabola_focus
    d_far = np.sqrt(radius_intercept**2 + (z_intercept - z_parabola_focus) ** 2)
    d_near = np.sqrt(radius_intercept**2 + (z_intercept - focal_length) ** 2)
    a_hyp = np.abs(d_far - d_near) / 2
    e_hyp = (c_hyp / a_hyp).to_value(u.dimensionless_unscaled)

    return optika.systems.SequentialSystem(
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
                sag=optika.sags.ParabolicSag(focal_length=-f_parabola),
                material=optika.materials.Mirror(),
                aperture=optika.apertures.AnnularAperture(
                    radius_inner=radius_aperture - halfwidth,
                    radius_outer=radius_aperture + halfwidth,
                ),
                transformation=na.transformations.Cartesian3dTranslation(
                    z=focus_parabola,
                ),
                is_pupil_stop=True,
            ),
            optika.surfaces.Surface(
                name="hyperboloid",
                sag=optika.sags.ConicSag(
                    radius=-a_hyp * (e_hyp**2 - 1),
                    conic=-(e_hyp**2),
                ),
                material=optika.materials.Mirror(),
                aperture=optika.apertures.AnnularAperture(
                    radius_inner=radius_intercept - halfwidth,
                    radius_outer=radius_intercept + halfwidth,
                ),
                transformation=na.transformations.Cartesian3dTranslation(
                    z=z_center - a_hyp,
                ),
            ),
        ],
        sensor=optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=width_pixel,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            timedelta_exposure=1 * u.us,
            num_pixel=na.Cartesian2dVectorArray(1024, 1024),
            transformation=na.transformations.Cartesian3dTranslation(
                z=focal_length,
            ),
        ),
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=na.linspace(15, 20, axis="wavelength", num=2) * u.AA,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=7,
                centers=True,
            ),
        ),
        coordinates_pupil="polar",
    )


_system_wolter = _system_wolter()


@pytest.mark.parametrize(argnames="a", argvalues=[_system_wolter])
class TestSequentialSystemWolter(
    AbstractTestAbstractSequentialSystem,
):
    """
    A Wolter-I telescope with an annular entrance pupil, sampled in polar
    pupil coordinates.  This exercises the polar code path of the stop
    solve, the pupil fit, and the denormalization end to end.
    """

    def test_every_ray_inside_the_field_lands_on_the_pupil_stop(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        """
        The rays are drawn between the rings of the annulus, so the pupil
        stop vignettes none of the rays the field stop admits.
        """
        unvignetted = a.raytrace().outputs.unvignetted
        at_object = unvignetted[{a.axis_surface: 0}]
        at_stop = unvignetted[{a.axis_surface: a.index_pupil_stop}]
        assert np.all(at_stop == at_object)

    def test_rectangular_coordinates_miss_the_annulus(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        """
        The same telescope in rectangular coordinates loses most of its rays
        at the pupil stop, which is what polar coordinates are for.
        """
        b = dataclasses.replace(a, coordinates_pupil="rectangular")
        unvignetted = b.raytrace().outputs.unvignetted
        at_object = unvignetted[{b.axis_surface: 0}]
        at_stop = unvignetted[{b.axis_surface: b.index_pupil_stop}]
        assert at_stop.sum() < 0.1 * at_object.sum()

    def test_pupil_fit_reproduces_the_edge_of_the_pupil(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        """
        The polar fit is one quadratic in field per point along the edge of
        the pupil, and reproduces every point of the stop rays it was made
        from.
        """
        wavelength = a.grid_input.wavelength
        stops = a._calc_rayfunction_stops(wavelength)
        fit = a._calc_pupil_fit(wavelength, stops)
        assert isinstance(fit, na.PolynomialFitFunctionArray)

        field, pupil = a._field_and_pupil(stops.outputs)
        x = optika.vectors.SceneVectorArray(wavelength, a._field_of_stop_samples(field))
        error = (fit(x).outputs - pupil).length
        width = (pupil.max(a.axis_stops) - pupil.min(a.axis_stops)).length
        assert np.all(error < 1e-5 * width)


_radius_field_rotated = 2 * u.mm
_decenter_field_rotated = 3 * u.mm

_system_rotated_object = optika.systems.SequentialSystem(
    object=optika.surfaces.Surface(
        name="source",
        aperture=optika.apertures.CircularAperture(
            radius=_radius_field_rotated,
            transformation=na.transformations.Cartesian3dTranslation(
                x=_decenter_field_rotated,
            ),
        ),
        is_field_stop=True,
        transformation=na.transformations.Cartesian3dRotationY(180 * u.deg),
    ),
    surfaces=[
        optika.surfaces.Surface(
            name="mirror",
            sag=optika.sags.SphericalSag(radius=240 * u.mm),
            material=optika.materials.Mirror(),
            aperture=optika.apertures.CircularAperture(radius=15 * u.mm),
            is_pupil_stop=True,
            transformation=na.transformations.Cartesian3dTranslation(
                z=-200 * u.mm,
            ),
        ),
    ],
    sensor=optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=150 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        timedelta_exposure=1 * u.s,
        num_pixel=na.Cartesian2dVectorArray(128, 128),
        transformation=na.transformations.Cartesian3dTranslation(
            z=100 * u.mm,
        ),
    ),
    grid_input=_grid_input,
)


@pytest.mark.parametrize(argnames="a", argvalues=[_system_rotated_object])
class TestSequentialSystemRotatedObject(
    AbstractTestAbstractSequentialSystem,
):
    """
    A finite-conjugate relay whose object surface is rotated 180 degrees
    about :math:`y`, so its local coordinate frame differs from the global
    frame. This guards the object-local frame handling of the stop
    root-finding problem: the solved stop rays must be expressed in the
    object surface's local coordinates before their direction is flipped,
    since that is the frame in which the field and pupil coordinates of the
    input grid are interpreted.
    """

    def test_field_bounds_match_decentered_aperture(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        x_min = _decenter_field_rotated - _radius_field_rotated
        x_max = _decenter_field_rotated + _radius_field_rotated
        assert np.abs(a.field_min.x - x_min) < 1 * u.um
        assert np.abs(a.field_max.x - x_max) < 1 * u.um

    def test_rays_reach_the_sensor(
        self,
        a: optika.systems.AbstractSequentialSystem,
    ):
        # the square field/pupil grids overfill the circular apertures, so
        # the unvignetted fraction is well below 1 even for a healthy trace;
        # with a mirrored object frame it is exactly 0
        unvignetted = a.rayfunction_default.outputs.unvignetted
        assert unvignetted.mean().ndarray > 0.25


def _pupil_shared_box(
    a: optika.systems.AbstractSequentialSystem,
    grid: optika.vectors.ObjectVectorArray,
) -> na.AbstractCartesian2dVectorArray:
    """
    The pupil of `grid` denormalized onto the box shared by every field
    point, which is what the system falls back to without a per-field fit.
    """
    return a._denormalize_interval(grid.pupil, lower=a.pupil_min, upper=a.pupil_max)


def test_rayfunction_stops_leaves_out_the_center_it_was_solved_with():
    """
    :attr:`rayfunction_stops` is the outline of the field, so the rays through
    the center of the field stop, which are solved along with it for the sake
    of the pupil fit, are not among them.
    """
    a = _system_newtonian
    with_center = a._rayfunction_stops_with_center
    outline = a.rayfunction_stops
    axis_wire, axis_edge = a.axis_pupil_stop, a.axis_field_stop

    assert na.shape(with_center)[axis_edge] == na.shape(outline)[axis_edge] + 1

    # and it is the same solve, not a second one
    kept = with_center.outputs[{axis_edge: slice(None, -1)}]
    assert np.all(kept.position == outline.outputs.position)

    # the extra sample sits inside the outline of the field
    field, _ = a._field_and_pupil(with_center.outputs)
    center = field[{axis_edge: -1}].mean(axis_wire)
    edge = field[{axis_edge: slice(None, -1)}]
    assert np.all(center.x > edge.x.min())
    assert np.all(center.x < edge.x.max())
    assert np.all(center.y > edge.y.min())
    assert np.all(center.y < edge.y.max())


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[_system_newtonian, _system_grazing],
)
def test_pupil_fit_resolves_the_entrance_pupil_per_field(
    a: optika.systems.AbstractSequentialSystem,
):
    """
    The entrance pupil is denormalized per field point, from a fit which
    reproduces the pupil of every point along the edge of the field, and which
    is narrower at the center of the field than the box shared by every field
    point, since the pupil of these systems walks across the field.
    """
    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    fit_min, fit_max = a._calc_pupil_fit(wavelength, stops)

    # the field and pupil along the edge of both stops, in whichever of angle
    # and position each is measured in for this object
    field = a.field_boundary
    pupil = a.pupil_boundary

    # the axis along the edge of the pupil stop, whichever stop axis the pupil
    # grid was swept over
    axis_wire = a.axis_pupil_stop
    axis_edge = a.axis_field_stop

    width = pupil.max(a.axis_stops) - pupil.min(a.axis_stops)
    tolerance = 1e-5 * width

    # the fit reproduces the pupil at every point along the edge of the field
    field_edge = field.mean(axis_wire)
    x = optika.vectors.SceneVectorArray(wavelength, field_edge)
    error_min = np.abs(fit_min(x).outputs - pupil.min(axis_wire))
    error_max = np.abs(fit_max(x).outputs - pupil.max(axis_wire))
    assert np.all(error_min.x < tolerance.x)
    assert np.all(error_min.y < tolerance.y)
    assert np.all(error_max.x < tolerance.x)
    assert np.all(error_max.y < tolerance.y)

    # the pupil at the center of the field is narrower than the box shared by
    # every field point
    x_center = optika.vectors.SceneVectorArray(wavelength, field_edge.mean(axis_edge))
    width_center = fit_max(x_center).outputs - fit_min(x_center).outputs
    assert np.all(width_center.x < width.x)
    assert np.all(width_center.y < width.y)


def test_pupil_fit_is_made_against_an_uncertain_field():
    """
    The fit is made against the field as it actually is, uncertainty and all,
    so it reproduces the pupil of every sample of the distribution rather than
    only of the nominal value.

    An uncertain field stop is what makes the field uncertain here. The optics
    are untouched, so every sample sees the same quadratic relation between
    the pupil and the field, sampled at a different set of field points. A fit
    made against the nominal field alone would reproduce the nominal pupil and
    miss each sample by roughly the width of the distribution.
    """
    base = _system_newtonian
    radius = base.object.aperture.radius
    a = dataclasses.replace(
        base,
        object=dataclasses.replace(
            base.object,
            aperture=dataclasses.replace(
                base.object.aperture,
                radius=na.NormalUncertainScalarArray(
                    nominal=radius,
                    width=0.05 * radius,
                    num_distribution=5,
                    seed=42,
                ),
            ),
        ),
    )

    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    fit_min, fit_max = a._calc_pupil_fit(wavelength, stops)

    field = a.field_boundary
    pupil = a.pupil_boundary

    assert isinstance(field.x, na.AbstractUncertainScalarArray)
    assert isinstance(pupil.x, na.AbstractUncertainScalarArray)

    axis_wire = a.axis_pupil_stop

    width = pupil.max(a.axis_stops) - pupil.min(a.axis_stops)
    tolerance = 1e-5 * width

    x = optika.vectors.SceneVectorArray(wavelength, field.mean(axis_wire))
    error_min = np.abs(fit_min(x).outputs - pupil.min(axis_wire))
    error_max = np.abs(fit_max(x).outputs - pupil.max(axis_wire))

    # `np.all` leaves the distribution alone, since it is a batch axis, so
    # these hold sample by sample and not merely on the nominal value
    assert np.all(error_min.x < tolerance.x)
    assert np.all(error_min.y < tolerance.y)
    assert np.all(error_max.x < tolerance.x)
    assert np.all(error_max.y < tolerance.y)


def test_pupil_of_the_center_of_the_field_is_measured_on_a_translated_object():
    """
    The rays which find the pupil at the center of the field lie on the object
    surface itself, even when it is translated along the axis, since the
    entrance pupil is measured on the object.

    They are solved along with the rays which graze both stops and carried
    back to the object with them, so this holds for the same reason it holds
    for the outline of the field.
    """
    base = _system_newtonian
    shift = -500 * u.mm
    a = dataclasses.replace(
        base,
        object=dataclasses.replace(
            base.object,
            transformation=na.transformations.Cartesian3dTranslation(z=shift),
        ),
    )

    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    axis_edge = a.axis_field_stop
    center = stops.outputs[{axis_edge: -1}]

    # these come back in the object's own coordinates, as the stop rays do, so
    # carry them into the world's to say where the object actually is
    position = a.object.transformation(center.position)

    assert np.allclose(position.z, shift)


def test_models_are_invariant_under_a_rigid_motion():
    """
    The three models fit from a raytrace survive a rigid motion of the whole
    system, in the same way and for the same reason the rays do.

    Fit on one system rather than on all of them, since these need a
    wavelength grid on an axis of its own which not every system in this
    module has, and each one costs a raytrace of its own.

    The grid stops short of the edge of the field, for a sharper reason than
    :func:`AbstractTestAbstractSequentialSystem.test_rayfunction_is_invariant_under_a_rigid_motion`
    has. Both models fit through a mask built from ``unvignetted``, so a ray
    which the roundoff of a motion carries across an aperture edge does not
    merely move, it joins or leaves the fitted set, and a vignetted ray landed
    wherever it landed. With a grid running to the edge that moves the
    distortion model by tens of pixels; an interior grid holds it to roundoff.
    That is a fact about sampling on aperture edges rather than about frames,
    and the reason :obj:`_grid_field_rigid` stops short as well.
    """
    a = _system_grazing
    b = _moved_rigidly(a, _motion_rigid)

    wavelength = na.linspace(500, 600, axis="_rigid_wavelength", num=3) * u.nm
    kwargs = dict(
        wavelength=wavelength,
        field=_grid_field_rigid,
        pupil=_grid_pupil_rigid,
        degree=1,
    )

    distortion_a = a.distortion(**kwargs)
    distortion_b = b.distortion(**kwargs)
    sensor_a = distortion_a.coordinates_sensor
    sensor_b = distortion_b.coordinates_sensor
    assert (sensor_b - sensor_a).length.max() < 1e-6 * sensor_a.length.max()

    vignetting_a = a.vignetting(**kwargs)
    vignetting_b = b.vignetting(**kwargs)
    illumination_a = vignetting_a.illumination
    illumination_b = vignetting_b.illumination
    assert np.abs(illumination_b - illumination_a).max() < 1e-6

    # the sampling of this one is random, so hold it to a seed
    area_a = a.area_effective(wavelength=wavelength, seed=42).area
    area_b = b.area_effective(wavelength=wavelength, seed=42).area
    assert np.abs(area_b - area_a).max() < 1e-6 * np.abs(area_a).max()


def test_area_effective_gives_each_ray_the_pupil_of_its_own_field_point(
    monkeypatch,
):
    """
    Every ray the effective area traces is drawn inside the entrance pupil
    of the field position it was drawn at, and weighted by the area of its
    pupil cell there.

    The entrance pupil walks across the field, so the pupil at the center of
    a field cell is not the pupil of a ray drawn near the cell's edge. One box
    for the whole cell either clips the pupils near the edges, if it is the
    center's, or wastes rays outside their own pupil, if it is the union of
    the pupils at the vertices. On FURST the pupil walks eight times its own
    width across one cell of the default grid, and a union would waste nine
    rays in ten.
    """
    a = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input_wavelength,
    )
    axis_pupil = ("_pupil_x", "_pupil_y")
    shape_pupil = na.shape(a._pupil_vertices_default)
    num_cells = (shape_pupil["_pupil_x"] - 1) * (shape_pupil["_pupil_y"] - 1)

    # a pupil a fifth of the shared box wide which walks across seven tenths
    # of it, so the box moves by a third of its width per cell of the default
    # field grid, and stays inside the shared box at every field position
    lo, hi = a.pupil_min, a.pupil_max
    span = hi - lo
    center = (lo + hi) / 2
    field_center = (a.field_min + a.field_max) / 2
    field_half = (a.field_max - a.field_min) / 2

    def walk(sign):
        def fit(x: optika.vectors.SceneVectorArray) -> na.FunctionArray:
            normalized = (x.field - field_center) / field_half
            box_center = center + 0.35 * span * normalized
            return na.FunctionArray(x, box_center + sign * 0.1 * span)

        return fit

    # `pupil_fit` is a cached property, so this is what `_denormalize_grid` reads
    a.__dict__["pupil_fit"] = (walk(-1), walk(+1))

    captured = {}
    rayfunction = optika.systems.AbstractSequentialSystem.rayfunction

    def capture(self, **kwargs):
        captured.update(kwargs)
        return rayfunction(self, **kwargs)

    monkeypatch.setattr(optika.systems.AbstractSequentialSystem, "rayfunction", capture)
    result = a.area_effective(seed=42)
    assert np.all(np.isfinite(result.area))

    field = captured["field"]
    pupil = captured["pupil"]
    area = captured["intensity"]

    # the box at each ray's own field position, from the same calibration
    corners = a._denormalize_grid_from_rays(
        grid=optika.vectors.ObjectVectorArray(
            wavelength=captured["wavelength"],
            field=field,
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray(*axis_pupil),
                num=2,
            ),
        ),
        rayfunction_stops=a.rayfunction_stops,
        pupil_fit=a.pupil_fit,
        normalized_field=False,
    ).pupil
    box_lo = corners.min(axis_pupil)
    box_hi = corners.max(axis_pupil)

    # the walk is real: the boxes are not all the same
    assert box_lo.x.ptp() > 0.5 * span.x.min()

    # every ray lies inside its own pupil
    assert np.all((box_lo.x <= pupil.x) & (pupil.x <= box_hi.x))
    assert np.all((box_lo.y <= pupil.y) & (pupil.y <= box_hi.y))

    # and is weighted by its share of that pupil and no other box
    box = box_hi - box_lo
    expected = box.x * box.y / num_cells
    assert np.all(np.abs(area - expected) <= 1e-9 * expected)


def test_a_physical_pupil_at_a_new_wavelength_skips_the_pupil_fit(monkeypatch):
    """
    A grid whose pupil is already physical has no use for the entrance pupil
    fit, so at a wavelength the caches were not built for only the stops are
    solved.
    """
    a = dataclasses.replace(_system_newtonian)
    wavelength = 1.01 * a.grid_input.wavelength

    # take the calibration away altogether, so that anything asking for it
    # fails by name rather than through a stub whose body must never run
    monkeypatch.delattr(optika.systems.AbstractSequentialSystem, "_calc_pupil_fit")

    # a grid which does need it now cannot be served, which is the control
    with pytest.raises(AttributeError, match="_calc_pupil_fit"):
        a._stops_and_pupil_fit(wavelength, normalized_pupil=True)

    stops, fit = a._stops_and_pupil_fit(wavelength, normalized_pupil=False)
    assert fit is None

    # the outline alone, as many samples as the outline the system exposes
    axis_edge = a.axis_field_stop
    assert na.shape(stops)[axis_edge] == na.shape(a.rayfunction_stops)[axis_edge]

    # and the same holds all the way through a raytrace
    grid = a.grid_input
    pupil = _pupil_shared_box(a, grid)
    rays = a.raytrace(
        wavelength=wavelength,
        field=grid.field,
        pupil=pupil,
        normalized_pupil=False,
        accumulate=False,
    )
    assert np.any(rays.outputs.unvignetted)


@pytest.mark.parametrize("coordinates_pupil", ["rectangular", "polar"])
def test_vignetting_weights_each_field_point_by_the_size_of_its_pupil(
    coordinates_pupil: str,
):
    """
    A field point which collects the same fraction of a pupil twice as wide
    collects four times the light, and the vignetting model says so.

    :meth:`area_effective` weights by the area of each pupil cell and averages
    the product over the field, so the two models multiply together to give
    the light collected at one field point only if this one carries how large
    that field point's pupil is.  Once the pupil is resolved per field point
    that is no longer shared, which is what makes the weight necessary.

    Every fixture in this module has a pupil whose area is the same across its
    field, so the rays are built here rather than traced.
    """
    a = dataclasses.replace(_system_newtonian, coordinates_pupil=coordinates_pupil)

    axis_wavelength = ("_vw",)
    axis_field = ("_vfx", "_vfy")
    axis_pupil = ("_vpx", "_vpy")

    # the pupil of the second column of the field is twice as wide
    width = na.ScalarArray(np.array([1.0, 2.0]), axes=("_vfx",))
    pupil = (
        na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray(*axis_pupil),
            num=3,
        )
        * width
        * u.mm
    )

    inputs = optika.vectors.ObjectVectorArray(
        wavelength=na.linspace(500, 600, axis=axis_wavelength[0], num=2) * u.nm,
        field=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray(*axis_field),
            num=2,
        )
        * u.deg,
        pupil=pupil,
    )
    shape = na.shape_broadcasted(inputs.wavelength, inputs.field, inputs.pupil)
    rays = optika.rays.RayFunctionArray(
        inputs=inputs,
        outputs=optika.rays.RayVectorArray(
            unvignetted=na.broadcast_to(na.ScalarArray(np.array(True)), shape),
        ),
    )

    model = a._fit_vignetting(
        rays=rays,
        axis_wavelength=axis_wavelength,
        axis_field=axis_field,
        axis_pupil=axis_pupil,
        degree=1,
    )

    # every ray survives, so the whole difference is the size of the pupil
    illumination = model.illumination
    narrow = illumination[{"_vfx": 0}].mean()
    wide = illumination[{"_vfx": 1}].mean()
    assert np.allclose((wide / narrow).ndarray, 4)


def test_pupil_fit_is_anchored_at_the_center_of_the_field_at_every_sampling():
    """
    The sample the pupil fit is anchored on sits at the center of the field
    however finely the stops are sampled, and the fit there is the same
    between samplings which resolve the pupil's extent.

    The center is the middle of the field stop's wire, which is exact at every
    sampling. The extent of the pupil found there is read off a polygon with
    `samples_pupil_stop` vertices, like every other sample of the fit, and a
    polygon too coarse to reach the pupil's extrema falls short at the center
    exactly as it does along the edge: an 11-point wire, whose ten distinct
    vertices sit 36 degrees apart and never at 90, misses a circle's extent
    in y. The two samplings compared below both reach it.
    """
    a = _system_newtonian
    wavelength = a.grid_input.wavelength
    zero = optika.vectors.SceneVectorArray(
        wavelength,
        na.Cartesian2dVectorArray(0, 0) * na.unit(a.field_boundary.x),
    )

    fits = {}
    for samples in [11, 21, 41]:
        stops = a._calc_rayfunction_stops(
            wavelength_input=wavelength,
            samples_field_stop=samples,
            samples_pupil_stop=samples,
        )

        # the anchor is at the center of the field at every sampling
        field, _ = a._field_and_pupil(stops.outputs)
        center = field[{a.axis_field_stop: -1}].mean(a.axis_pupil_stop)
        assert np.abs(center.x) < 1e-12 * na.unit(center.x)
        assert np.abs(center.y) < 1e-12 * na.unit(center.y)

        if samples > 11:
            fit_min, fit_max = a._calc_pupil_fit(wavelength, stops)
            fits[samples] = fit_min(zero).outputs, fit_max(zero).outputs

    lo_coarse, hi_coarse = fits[21]
    lo_fine, hi_fine = fits[41]
    width = (hi_fine - lo_fine).length
    assert (lo_coarse - lo_fine).length < 1e-9 * width
    assert (hi_coarse - hi_fine).length < 1e-9 * width


def test_pupil_calibration_does_not_swallow_an_unrelated_error(monkeypatch):
    """
    Only a design matrix which cannot be inverted falls back to the shared
    box.

    A :class:`ValueError` from anywhere else in the fit is a system built
    wrong or a mistake in the fit itself, and turning it into a pupil which
    silently stops being resolved per field point would hide it.
    """
    a = dataclasses.replace(_system_newtonian)

    def fail(*args, **kwargs):
        raise ValueError("this is not a singular matrix")

    monkeypatch.setattr(na.PolynomialFitFunctionArray, "from_degree", fail)

    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    with pytest.raises(ValueError, match="not a singular matrix"):
        a._calc_pupil_fit(wavelength, stops)


def test_pupil_denormalization_falls_back_when_a_corner_is_not_a_number():
    """
    A pupil corner which is not a number falls back to the box shared by every
    field point, in the same way a collapsed one does.

    Testing the healthy case and negating it, rather than testing for the
    broken one, is what makes this hold: a comparison against a NaN is false
    either way round, so a guard written the other way would pass the NaN
    through and turn every ray at that field point into a NaN silently.
    """
    a = dataclasses.replace(_system_newtonian)

    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    fit_min, fit_max = a._calc_pupil_fit(wavelength, stops)

    # poison the fit so that it evaluates to NaN at every field point
    nan = np.nan * na.unit(fit_min.outputs.x)
    fit_min = dataclasses.replace(
        fit_min,
        outputs=na.Cartesian2dVectorArray(x=nan, y=nan),
    )
    # `pupil_fit` is a cached property, so this is what `_denormalize_grid` reads
    a.__dict__["pupil_fit"] = (fit_min, fit_max)

    grid = a.grid_input
    result = a._denormalize_grid(grid)

    assert not np.any(np.isnan(result.pupil.x.ndarray))
    assert not np.any(np.isnan(result.pupil.y.ndarray))

    # the fallback is the shared box, exactly as when the fit is singular
    expected = _pupil_shared_box(a, grid)
    assert np.allclose(result.pupil.x, expected.x)
    assert np.allclose(result.pupil.y, expected.y)


def test_pupil_denormalization_falls_back_when_the_fit_is_singular():
    """
    A field stop whose outline is degenerate in one component leaves the
    least-squares fit of the pupil with a singular design matrix.  That is the
    one way the calibration can fail on its own, since the rays through the
    center of the field are solved along with the stops and fail only with
    them, and it falls back to the box shared by every field point rather than
    failing a raytrace which the shared box would have carried out.

    The fit solves for its coefficients lazily, so forcing the solve is what
    keeps this failure inside the fallback.
    """
    base = _system_newtonian
    a = dataclasses.replace(
        base,
        object=dataclasses.replace(
            base.object,
            aperture=optika.apertures.RectangularAperture(
                half_width=na.Cartesian2dVectorArray(
                    x=np.sin(0.05 * u.deg),
                    y=0 * u.dimensionless_unscaled,
                ),
            ),
        ),
    )

    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)
    assert a._calc_pupil_fit(wavelength, stops) is None

    # the pupil is the shared box, as it was before the fit existed
    grid = a.grid_input
    result = a._denormalize_grid(grid)
    expected = _pupil_shared_box(a, grid)
    assert np.allclose(result.pupil.x, expected.x)
    assert np.allclose(result.pupil.y, expected.y)

    rays = a.raytrace(accumulate=False)
    assert np.any(rays.outputs.unvignetted)


def test_stops_and_pupil_are_solved_once_at_the_default_wavelength(monkeypatch):
    """
    Denormalizing a grid needs the stop rays and the entrance-pupil fit, and
    both depend on nothing but the wavelength.  A caller working at the
    system's own input wavelengths must pay for each of them once, however
    many times it traces.
    """
    a = dataclasses.replace(_system_newtonian)

    calls = dict(stops=0, fit=0)

    solve_stops = type(a)._calc_rayfunction_stops
    solve_fit = type(a)._calc_pupil_fit

    def count_stops(self, *args, **kwargs):
        calls["stops"] += 1
        return solve_stops(self, *args, **kwargs)

    def count_fit(self, *args, **kwargs):
        calls["fit"] += 1
        return solve_fit(self, *args, **kwargs)

    monkeypatch.setattr(type(a), "_calc_rayfunction_stops", count_stops)
    monkeypatch.setattr(type(a), "_calc_pupil_fit", count_fit)

    grid = a.grid_input
    for _ in range(3):
        a.raytrace(field=grid.field, pupil=grid.pupil, accumulate=False)

    assert calls["stops"] == 1
    assert calls["fit"] == 1

    # a wavelength the caches were not solved at is solved from scratch
    a.raytrace(
        wavelength=grid.wavelength + 1 * u.nm,
        field=grid.field,
        pupil=grid.pupil,
        accumulate=False,
    )

    assert calls["stops"] == 2
    assert calls["fit"] == 2


def test_solve_rays_launches_from_a_translated_surface():
    """
    The rays the stop solver finds are launched from the surface it is given,
    in global coordinates, even when that surface is translated along the
    axis.  The solver fixes them in the local coordinates of that surface, so
    it must carry every trial, and the result, into global coordinates, or a
    translated surface launches them from the wrong depth and a ray with any
    angle to the axis misses the surfaces it should strike.
    """
    base = _system_newtonian
    shift = -500 * u.mm
    a = dataclasses.replace(
        base,
        object=dataclasses.replace(
            base.object,
            transformation=na.transformations.Cartesian3dTranslation(z=shift),
        ),
    )

    surfaces = a.surfaces_all
    subsystem = surfaces[: a.index_pupil_stop + 1]
    obj = subsystem[0]
    pupil_stop = subsystem[~0]

    grid_first = obj.aperture.wire(num=5)
    grid_first = na.Cartesian2dVectorArray(grid_first.x, grid_first.y)
    grid_last = pupil_stop.aperture.wire(num=5)
    grid_last = na.Cartesian2dVectorArray(grid_last.x, grid_last.y)

    rays = a._solve_rays(subsystem, grid_first, grid_last, a.grid_input.wavelength)

    assert np.allclose(rays.position.z, shift)


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


def test_area_effective_is_reproducible_when_seeded():
    """
    A seed fixes the sampling, and so fixes the answer.

    The field and the pupil are sampled at a point drawn inside each cell,
    which keeps the quadrature from aliasing against an edge but leaves the
    result a little different on every call. Anything which must be
    reproduced, such as a figure in an article, needs to be able to ask for
    the same sample twice.
    """
    system = optika.systems.SequentialSystem(
        surfaces=_surfaces,
        sensor=_sensor,
        grid_input=_grid_input_wavelength,
    )
    wavelength = _grid_input_wavelength.wavelength

    a = system.area_effective(wavelength=wavelength, seed=42)(wavelength)
    b = system.area_effective(wavelength=wavelength, seed=42)(wavelength)
    c = system.area_effective(wavelength=wavelength, seed=43)(wavelength)

    assert np.all(a == b)
    assert np.any(a != c)


def _system_with_launch_surface(transformation, fold: bool = False):
    """
    A system whose pupil stop comes first and carries a physical aperture, so
    that the solver's free variable is the launch direction, with that stop
    placed by the given transformation.

    With `fold`, a 45 degree mirror ahead of the stop turns the beam onto the
    world's :math:`x` axis, and the surfaces after the stop are placed along
    that axis instead.
    """
    turn = na.transformations.Cartesian3dRotationY(90 * u.deg)

    mirror = optika.surfaces.Surface(
        name="mirror",
        sag=optika.sags.SphericalSag(radius=-600 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.CircularAperture(60 * u.mm),
        transformation=(
            (na.transformations.Cartesian3dTranslation(x=400 * u.mm) @ turn)
            if fold
            else na.transformations.Cartesian3dTranslation(z=300 * u.mm)
        ),
    )

    stop = optika.surfaces.Surface(
        name="stop",
        aperture=optika.apertures.CircularAperture(15 * u.mm),
        transformation=transformation,
        is_pupil_stop=True,
    )

    fold_mirror = optika.surfaces.Surface(
        name="fold",
        material=optika.materials.Mirror(),
        aperture=optika.apertures.CircularAperture(40 * u.mm),
        transformation=na.transformations.Cartesian3dRotationY(-45 * u.deg),
    )

    return optika.systems.SequentialSystem(
        object=optika.surfaces.Surface(
            name="object",
            aperture=optika.apertures.CircularAperture(np.sin(0.5 * u.deg)),
            transformation=na.transformations.Cartesian3dTranslation(z=-400 * u.mm),
        ),
        surfaces=([fold_mirror] if fold else []) + [stop, mirror],
        sensor=optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=10 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(256, 256),
            transformation=(
                (na.transformations.Cartesian3dTranslation(x=100 * u.mm) @ turn)
                if fold
                else None
            ),
            is_field_stop=True,
        ),
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=500 * u.nm,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=3,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=3,
            ),
        ),
    )


def test_stops_solve_with_a_folded_beam():
    """
    A 45 degree fold ahead of the pupil stop, so that the beam reaching it runs
    along the world's :math:`x` axis rather than its :math:`z` axis.

    The launch rays are solved in the local coordinates of the surface they are
    launched from, where that beam runs along the surface's own normal.
    Solving in world coordinates instead puts the parametrization of the free
    direction exactly on its pole here, since the beam is then perpendicular to
    the axis the two free components are measured against.
    """
    a = _system_with_launch_surface(
        na.transformations.Cartesian3dTranslation(x=150 * u.mm)
        @ na.transformations.Cartesian3dRotationY(90 * u.deg),
        fold=True,
    )
    result = a._calc_rayfunction_stops(500 * u.nm)
    assert np.all(np.isfinite(result.outputs.position.length))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the free direction is charted on the launch surface's transverse "
        "plane, which is singular for a beam grazing that surface (#227)"
    ),
)
def test_stops_solve_when_the_launch_surface_is_nearly_edge_on():
    """
    A launch surface turned nearly edge-on to the beam, as a grazing-incidence
    optic modeled as a tilted segment would be.
    """
    a = _system_with_launch_surface(
        na.transformations.Cartesian3dRotationY(88 * u.deg),
    )
    result = a._calc_rayfunction_stops(500 * u.nm)

    # unreachable until the solve above stops raising, which is the point
    assert np.all(np.isfinite(result.outputs.position.length))  # pragma: nocover


# the field stop is decentered so that a mirrored field frame is observable:
# rays aimed using global-frame field bounds miss the aperture entirely


def test_polar_pupil_of_a_stop_without_a_hole_fills_it():
    """
    A round pupil stop with no hole is swept out from its center, so polar
    coordinates land every ray the field stop admits on it, where
    rectangular coordinates lose the corners of the box around it.
    """
    grid = optika.vectors.ObjectVectorArray(
        wavelength=_grid_input.wavelength,
        field=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
            centers=True,
        ),
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=11,
            centers=True,
        ),
    )
    survival = dict()
    for coordinates in ("rectangular", "polar"):
        a = dataclasses.replace(
            _system_grazing,
            grid_input=grid,
            coordinates_pupil=coordinates,
        )
        unvignetted = a.raytrace().outputs.unvignetted
        at_object = unvignetted[{a.axis_surface: 0}]
        at_stop = unvignetted[{a.axis_surface: a.index_pupil_stop}]
        survival[coordinates] = at_stop.sum() / at_object.sum()

    assert survival["polar"] == 1
    assert survival["rectangular"] < 0.9


def test_polar_pupil_without_a_fit_takes_the_average_edge():
    """
    Where the polar fit is singular, every field point takes the edge of the
    pupil averaged along the edge of the field, which lies inside the shared
    pupil.
    """
    a = _system_wolter
    wavelength = a.grid_input.wavelength
    stops = a._calc_rayfunction_stops(wavelength)

    result = a._denormalize_grid_from_rays(
        grid=a.grid_input,
        rayfunction_stops=a._without_center(stops),
        pupil_fit=None,
    )
    pupil = result.pupil

    assert np.all(pupil.x >= a.pupil_min.x)
    assert np.all(pupil.y >= a.pupil_min.y)
    assert np.all(pupil.x <= a.pupil_max.x)
    assert np.all(pupil.y <= a.pupil_max.y)

    # the same edge at every field point
    center = pupil[dict(field_x=0, field_y=0)]
    assert np.all(pupil.x == center.x)
    assert np.all(pupil.y == center.y)


def test_denormalize_polar_interpolates_the_rings_in_polar_coordinates():
    """
    Between the samples of a ring, the map interpolates the radius and the
    azimuth about the center of the ring rather than the position along a
    chord, so a round ring stays round, including across the branch cut of
    the azimuth.
    """
    axis = "_edge"
    azimuth = na.linspace(0, 360, axis=axis, num=9) * u.deg

    def ring(radius: u.Quantity) -> na.Cartesian2dVectorArray:
        return na.Cartesian2dVectorArray(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
        )

    edge = np.concatenate([ring(2 * u.mm), ring(1 * u.mm)], axis=axis)

    # the first point of the outer ring, the last of the inner, the middle
    # of the annulus opposite the first point, a point halfway between the
    # first two samples of the outer ring, and one halfway between the
    # samples on either side of the branch cut
    normalized = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([-1, 1, 0, -0.875, 0.125]), axes=("_n",)),
        y=na.ScalarArray(np.array([1, -1, 0, 1, 0]), axes=("_n",)),
    )

    result = optika.systems.AbstractSequentialSystem._denormalize_polar(
        normalized=normalized,
        edge=edge,
        axis=axis,
    )

    radius = na.ScalarArray(np.array([2, 1, 1.5, 2, 1.5]) * u.mm, axes=("_n",))
    angle = na.ScalarArray(np.array([0, 0, 180, 22.5, 202.5]) * u.deg, axes=("_n",))

    assert np.allclose(result.x, radius * np.cos(angle))
    assert np.allclose(result.y, radius * np.sin(angle))


def test_coordinates_pupil_must_be_rectangular_or_polar():
    a = dataclasses.replace(_system_newtonian, coordinates_pupil="cylindrical")
    with pytest.raises(ValueError, match="coordinates_pupil"):
        a.rayfunction_stops


def _scene_normalized(
    wavelength: na.AbstractScalar,
    num: int = 11,
) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
    """A uniform-ish scene on a normalized field, for :meth:`image`."""
    return na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=+1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=num,
            ),
        ),
        outputs=na.random.uniform(
            low=0 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
            high=100 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
            shape_random=dict(field_x=num - 1, field_y=num - 1),
        ),
    )


def test_image_in_polar_coordinates_with_a_finite_object():
    """
    With the object at a finite distance the pupil is angular and the extent
    of a field cell is an area rather than a solid angle; the polar image
    path handles both.
    """
    a = dataclasses.replace(_system_rotated_object, coordinates_pupil="polar")
    assert not a.object_is_at_infinity

    scene = _scene_normalized(
        wavelength=na.linspace(530, 531, axis="wavelength", num=3) * u.nm,
    )
    result = a.image(scene, noise=False)
    assert result.outputs.sum() != 0 * u.electron


def test_image_in_polar_coordinates_with_grids_which_share_axes():
    """
    The field vertices may carry the wavelength axis and the pupil vertices
    the field axes; each is reduced to its cell centers along the axes it
    shares before the rays are drawn in it.
    """
    a = _system_wolter
    num = 11

    scene = _scene_normalized(
        wavelength=na.linspace(15, 20, axis="wavelength", num=3) * u.AA,
        num=num,
    )
    scene.inputs.position = scene.inputs.position * na.ScalarArray(
        np.ones(3),
        axes=("wavelength",),
    )
    pupil = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=+1,
        axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
        num=3,
    ) * na.ScalarArray(np.ones(num), axes=("field_x",))

    result = a.image(scene, pupil=pupil, noise=False)
    assert "wavelength" not in result.outputs.shape
    assert result.outputs.sum() != 0 * u.electron


@pytest.mark.parametrize("axis_pupil", [("pupil_x", "pupil_y"), ("pupil_y", "pupil_x")])
def test_area_of_polar_cells_does_not_depend_on_the_order_of_the_axes(
    axis_pupil: tuple[str, str],
):
    """
    One cell spanning the whole ring has the area of the annulus, whichever
    order its two axes are named in: the cells are cut along the axis the
    azimuth varies along, not the first one given.
    """
    a = _system_wolter
    wavelength = a.grid_input.wavelength
    stops, fit = a._stops_and_pupil_fit(wavelength)
    field, pupil = a._field_and_pupil(stops.outputs)

    grid = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=+1,
        axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
        num=2,
    )
    area = a._area_pupil_cells(
        wavelength=wavelength,
        field=a.field_boundary.mean(a.axis_stops),
        pupil=grid,
        axis_pupil=axis_pupil,
        rayfunction_stops=stops,
        pupil_fit=fit,
        normalized_pupil=True,
    )

    # the area of the annulus, from the stop rays at the center of the field
    edge = pupil[{a.axis_field_stop: ~0}]
    num = edge.shape[a.axis_pupil_stop] // 2
    axis = a.axis_pupil_stop

    def shoelace(ring: na.AbstractCartesian2dVectorArray) -> na.AbstractScalar:
        ring = ring[{axis: slice(None, -1)}]
        other = np.roll(ring, -1, axis=axis)
        return np.abs((ring.x * other.y - other.x * ring.y).sum(axis)) / 2

    expected = shoelace(edge[{axis: slice(None, num)}])
    expected = expected - shoelace(edge[{axis: slice(num, None)}])

    assert np.allclose(area.sum(axis_pupil), expected, rtol=1e-2)


def test_inferred_axes_follow_the_order_of_the_grid():
    """
    The field and pupil axes inferred from a grid come in the order the grid
    carries them, so that nothing drawn along them changes from one process
    to the next with the hash seed.
    """
    a = _system_newtonian
    axis_wavelength = ("wavelength",)

    for axes in (("_a", "_b"), ("_b", "_a")):
        field = na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=+1,
            axis=na.Cartesian2dVectorArray(*axes),
            num=3,
        )
        pupil = na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=+1,
            axis=na.Cartesian2dVectorArray(*reversed(axes)),
            num=3,
        )
        axis_field = a._normalize_axis_field(
            axis_field=None,
            axis_wavelength=axis_wavelength,
            field=field,
        )
        assert axis_field == tuple(field.shape)
        axis_pupil = a._normalize_axis_pupil(
            axis_pupil=None,
            axis_field=("_c", "_d"),
            axis_wavelength=axis_wavelength,
            pupil=pupil,
        )
        assert axis_pupil == tuple(pupil.shape)
