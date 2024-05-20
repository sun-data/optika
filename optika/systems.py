"""
Optical systems consisting of multiple optical surfaces.
"""

from __future__ import annotations
from typing import Sequence, Callable, Any, ClassVar
import abc
import dataclasses
import functools
import astropy.units as u
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import named_arrays as na
import optika

__all__ = [
    "AbstractSystem",
    "AbstractSequentialSystem",
    "SequentialSystem",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSystem(
    optika.mixins.Plottable,
    optika.mixins.Printable,
    optika.mixins.Transformable,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSequentialSystem(
    AbstractSystem,
):
    @property
    @abc.abstractmethod
    def object(self) -> None | optika.surfaces.AbstractSurface:
        """
        The external object being imaged or illuminated by this system.

        If :obj:`None`, the external object is assumed to be at infinity.
        """

    @property
    def object_is_at_infinity(self) -> bool:
        obj = self.object
        if obj is not None:
            aperture_obj = obj.aperture
            if aperture_obj is not None:
                unit_obj = na.unit_normalized(aperture_obj.bound_lower)
                if unit_obj.is_equivalent(u.mm):
                    result = False
                elif unit_obj.is_equivalent(u.dimensionless_unscaled):
                    result = True
                else:  # pragma: nocover
                    raise ValueError(
                        f"Unrecognized unit for object aperture, {unit_obj}"
                    )
            else:
                result = False
        else:
            result = True
        return result

    @property
    @abc.abstractmethod
    def surfaces(self) -> Sequence[optika.surfaces.AbstractSurface]:
        """
        a sequence of surfaces representing this optical system.

        At least one of these surfaces needs to be marked as the pupil surface,
        and if the object surface is not marked as the field stop, one of these
        surfaces needs to be marked as the field stop.
        """

    @property
    @abc.abstractmethod
    def sensor(self) -> None | optika.sensors.AbstractImagingSensor:
        """
        The imaging sensor that measures the light captured by this system.

        This is the last surface in the optical system.
        """

    @property
    @abc.abstractmethod
    def axis_surface(self) -> str:
        """
        The name of the logical axis representing the sequence of surfaces.
        """

    @property
    def surfaces_all(self) -> list[optika.surfaces.AbstractSurface]:
        """
        Concatenate :attr:`object` with :attr:`surfaces` into a single list of
        surfaces.
        """
        obj = self.object
        if obj is not None:
            result = [obj]
        else:
            result = []

        result += list(self.surfaces)

        sensor = self.sensor
        if sensor is not None:
            result += [sensor]

        if not any(s.is_field_stop for s in result):
            result[0] = dataclasses.replace(result[0], is_field_stop=True)
        return result

    @property
    @abc.abstractmethod
    def grid_input(self) -> optika.vectors.ObjectVectorArray:
        """
        The input grid to sample with rays.

        This grid is simultaneously projected onto both the field stop and the
        pupil stop.

        Positions on the stop can be specified in either absolute or normalized
        coordinates. Using normalized coordinates allows for injecting different
        grid types (cylindrical, stratified random, etc.) without specifying the
        scale of the stop surface.

        If positions are specified in absolute units, they are measured in the
        coordinate system of the corresponding stop surface.
        """

    @property
    def _indices_field_stop(self) -> list[int]:
        return [i for i, s in enumerate(self.surfaces_all) if s.is_field_stop]

    @property
    def _indices_pupil_stop(self) -> list[int]:
        return [i for i, s in enumerate(self.surfaces_all) if s.is_pupil_stop]

    @property
    def index_field_stop(self) -> int:
        """
        The index of the field stop in :attr:`surfaces_all`.
        """
        indices = self._indices_field_stop
        if not indices:
            raise ValueError(
                "Field stop is not defined for this system."
                "Set `is_field_stop=True` for at least one surface in this system."
            )
        return indices[~0]

    @property
    def index_pupil_stop(self) -> int:
        """
        The index of the pupil stop in :attr:`surfaces_all`.
        """
        indices = self._indices_pupil_stop
        if not indices:
            raise ValueError(
                "Pupil stop is not defined for this system."
                "Set `is_pupil_stop=True` for at least one surface in this "
                "system."
            )
        return indices[~0]

    @property
    def field_stop(self) -> optika.surfaces.AbstractSurface:
        """
        The field stop surface.
        """
        return self.surfaces_all[self.index_field_stop]

    @property
    def pupil_stop(self) -> optika.surfaces.AbstractSurface:
        """
        The pupil stop surface.
        """
        return self.surfaces_all[self.index_pupil_stop]

    @classmethod
    def _ray_error(
        cls,
        a: na.Cartesian2dVectorArray,
        rays: optika.rays.RayVectorArray,
        subsystem: list[optika.surfaces.AbstractSurface],
        grid_last: na.Cartesian2dVectorArray,
        component_variable: str,
        component_target: str,
        zfunc: Callable[[na.Cartesian2dVectorArray], na.ScalarLike],
    ):
        rays_component_variable = getattr(rays, component_variable)
        rays_component_variable.x = a.x
        rays_component_variable.y = a.y
        rays_component_variable.z = zfunc(a)

        rays = optika.propagators.propagate_rays(
            propagators=subsystem[1:],
            rays=rays,
        )

        transformation_last = subsystem[~0].transformation
        if transformation_last is not None:
            rays = transformation_last.inverse(rays)

        rays_component_target = getattr(rays, component_target)
        grid_last_trial = na.Cartesian2dVectorArray(
            x=rays_component_target.x,
            y=rays_component_target.y,
        )
        result = grid_last_trial - grid_last
        return result

    def _calc_rayfunction_stops_only(
        self,
        wavelength_input: na.ScalarLike,
        axis_pupil_stop: str,
        axis_field_stop: str,
        samples_pupil_stop: int = 101,
        samples_field_stop: int = 101,
    ) -> optika.rays.RayFunctionArray:
        result = optika.rays.RayFunctionArray(
            inputs=optika.vectors.ObjectVectorArray(
                wavelength=wavelength_input,
            ),
            outputs=optika.rays.RayVectorArray(
                wavelength=wavelength_input,
            ),
        )

        surfaces = self.surfaces_all

        indices_pupil_stop = self._indices_pupil_stop
        indices_field_stop = self._indices_field_stop

        if not indices_pupil_stop:
            raise ValueError("pupil not defined")
        if not indices_field_stop:
            raise ValueError("field stop not defined")

        while indices_pupil_stop and indices_field_stop:
            index_pupil_stop = indices_pupil_stop[0]
            index_field_stop = indices_field_stop[0]

            index_first = min(index_pupil_stop, index_field_stop)
            index_last = max(index_pupil_stop, index_field_stop)

            subsystem = surfaces[index_first : index_last + 1]

            surface_first = subsystem[0]
            surface_last = subsystem[~0]

            aperture_first = surface_first.aperture
            aperture_last = surface_last.aperture

            grid_first = np.moveaxis(
                a=aperture_first.wire(num=samples_pupil_stop),
                source="wire",
                destination=axis_pupil_stop,
            )
            grid_first = na.Cartesian2dVectorArray(grid_first.x, grid_first.y)

            grid_last = np.moveaxis(
                a=aperture_last.wire(num=samples_field_stop),
                source="wire",
                destination=axis_field_stop,
            )
            grid_last = na.Cartesian2dVectorArray(grid_last.x, grid_last.y)

            if surface_first.is_pupil_stop:
                indices_pupil_stop.pop(0)
                result.inputs.pupil = grid_first
                result.inputs.field = grid_last
            else:
                indices_field_stop.pop(0)
                result.inputs.field = grid_first
                result.inputs.pupil = grid_last

            if na.unit(grid_first).is_equivalent(u.mm):
                result.outputs.position = na.Cartesian3dVectorArray(
                    x=grid_first.x,
                    y=grid_first.y,
                    z=surface_first.sag(grid_first),
                )
                result.outputs.direction = na.Cartesian3dVectorArray(0, 0, 1)
                component_variable = "direction"

                def zfunc(xy: na.AbstractCartesian3dVectorArray):
                    return np.sqrt(1 - np.square(xy.length))

            elif na.unit(grid_first).is_equivalent(u.dimensionless_unscaled):
                result.outputs.direction = na.Cartesian3dVectorArray(
                    x=grid_first.x,
                    y=grid_first.y,
                    z=np.sqrt(1 - np.square(grid_first.length)),
                )
                result.outputs.position = na.Cartesian3dVectorArray() * u.mm
                component_variable = "position"

                def zfunc(xy: na.AbstractCartesian3dVectorArray):
                    return surface_first.sag(xy)

            else:
                raise ValueError(f"unrecognized input grid unit, {na.unit(grid_first)}")

            if surface_first.transformation is not None:
                result.outputs = surface_first.transformation(result.outputs)

            if na.unit(grid_last).is_equivalent(u.mm):
                component_target = "position"
            elif na.unit(grid_last).is_equivalent(u.dimensionless_unscaled):
                component_target = "direction"
            else:
                raise ValueError(f"unrecognized output grid unit, {na.unit(grid_last)}")

            variables = getattr(result.outputs, component_variable)

            root = na.optimize.root_newton(
                function=functools.partial(
                    self._ray_error,
                    rays=result.outputs,
                    subsystem=subsystem,
                    grid_last=grid_last,
                    component_variable=component_variable,
                    component_target=component_target,
                    zfunc=zfunc,
                ),
                guess=na.Cartesian2dVectorArray(
                    x=variables.x,
                    y=variables.y,
                ),
            )

            variables.x = root.x
            variables.y = root.y
            variables.z = zfunc(root)

        return result

    def _calc_rayfunction_stops(
        self,
        wavelength_input: na.ScalarLike,
        axis_pupil_stop: str,
        axis_field_stop: str,
        samples_pupil_stop: int = 101,
        samples_field_stop: int = 101,
    ) -> optika.rays.RayFunctionArray:
        surfaces = self.surfaces_all

        index_pupil_stop = self.index_pupil_stop
        index_field_stop = self.index_field_stop

        index_stop = min(index_pupil_stop, index_field_stop)

        subsystem = surfaces[index_stop::-1]

        rays_stop = self._calc_rayfunction_stops_only(
            wavelength_input=wavelength_input,
            axis_pupil_stop=axis_pupil_stop,
            axis_field_stop=axis_field_stop,
            samples_pupil_stop=samples_pupil_stop,
            samples_field_stop=samples_field_stop,
        )

        result = rays_stop.copy_shallow()
        result.outputs = optika.propagators.propagate_rays(
            propagators=subsystem,
            rays=rays_stop.outputs,
        )

        if self.transformation is not None:
            result.outputs = self.transformation(result.outputs)

        return result

    _axis_pupil_stop: ClassVar[str] = "_stop_pupil"
    _axis_field_stop: ClassVar[str] = "_stop_field"

    @functools.cached_property
    def rayfunction_stops(self) -> optika.rays.RayFunctionArray:
        """
        A rayfunction defined on the input surface of the optical system,
        which is designed to exactly strike the borders of both the field
        stop and the pupil stop.
        """
        return self._calc_rayfunction_stops(
            wavelength_input=self.grid_input.wavelength,
            axis_pupil_stop=self._axis_pupil_stop,
            axis_field_stop=self._axis_field_stop,
            samples_pupil_stop=21,
            samples_field_stop=21,
        )

    @property
    def field_min(self) -> na.AbstractCartesian2dVectorArray:
        """
        The lower left corner of this optical system's field of view.
        """
        axis = (self._axis_field_stop, self._axis_pupil_stop)
        if self.object_is_at_infinity:
            return self.rayfunction_stops.outputs.direction.xy.min(axis)
        else:
            return self.rayfunction_stops.outputs.position.xy.min(axis)

    @property
    def field_max(self) -> na.AbstractCartesian2dVectorArray:
        """
        The upper right corner of this optical system's field of view.
        """
        axis = (self._axis_field_stop, self._axis_pupil_stop)
        if self.object_is_at_infinity:
            return self.rayfunction_stops.outputs.direction.xy.max(axis)
        else:
            return self.rayfunction_stops.outputs.position.xy.max(axis)

    @property
    def pupil_min(self) -> na.AbstractCartesian2dVectorArray:
        """
        The lower left corner of this optical system's entrance pupil in
        physical units.
        """
        axis = (self._axis_field_stop, self._axis_pupil_stop)
        if self.object_is_at_infinity:
            return self.rayfunction_stops.outputs.position.xy.min(axis)
        else:
            return self.rayfunction_stops.outputs.direction.xy.min(axis)

    @property
    def pupil_max(self):
        """
        The upper right corner of this optical system's entrance pupil in
        physical units.
        """
        axis = (self._axis_field_stop, self._axis_pupil_stop)
        if self.object_is_at_infinity:
            return self.rayfunction_stops.outputs.position.xy.max(axis)
        else:
            return self.rayfunction_stops.outputs.direction.xy.max(axis)

    def _calc_rayfunction_input(
        self,
        grid_input: optika.vectors.ObjectVectorArray,
    ) -> optika.rays.RayFunctionArray:

        rayfunction_stops = self.rayfunction_stops

        object_is_at_infinity = self.object_is_at_infinity

        result = rayfunction_stops.copy_shallow()

        if object_is_at_infinity:
            field = rayfunction_stops.outputs.direction.xy
            pupil = rayfunction_stops.outputs.position.xy
        else:
            field = rayfunction_stops.outputs.position.xy
            pupil = rayfunction_stops.outputs.direction.xy

        axis_pupil = tuple(result.inputs.pupil.shape)
        axis_field = tuple(result.inputs.field.shape)

        field = (field.min(axis_pupil) + field.max(axis_pupil)) / 2
        pupil = (pupil.min(axis_field) + pupil.max(axis_field)) / 2

        min_field = field.min(axis=axis_field)
        min_pupil = pupil.min(axis=axis_pupil)

        ptp_field = field.ptp(axis=axis_field)
        ptp_pupil = pupil.ptp(axis=axis_pupil)

        result.inputs = grid_input.copy_shallow()
        result.inputs.field = ptp_field * (result.inputs.field + 1) / 2 + min_field
        result.inputs.pupil = ptp_pupil * (result.inputs.pupil + 1) / 2 + min_pupil

        if object_is_at_infinity:
            position = result.inputs.pupil
            direction = result.inputs.field
        else:
            position = result.inputs.field
            direction = result.inputs.pupil

        rays = optika.rays.RayVectorArray(
            wavelength=grid_input.wavelength,
            position=na.Cartesian3dVectorArray(
                x=position.x,
                y=position.y,
                z=0 * position.x.unit,
            ),
            direction=na.Cartesian3dVectorArray(
                x=direction.x,
                y=direction.y,
                z=np.sqrt(1 - np.square(direction.length)),
            ),
        )

        obj = self.object
        if obj is not None:
            if obj.transformation is not None:
                rays = obj.transformation(rays)

        result.outputs = rays

        return result

    @functools.cached_property
    def _rayfunction_input(self) -> optika.rays.RayFunctionArray:
        return self._calc_rayfunction_input(
            grid_input=self.grid_input,
        )

    def raytrace(
        self,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
        axis: None | str = None,
    ) -> optika.rays.RayFunctionArray:
        """
        Given the wavelength, field position, and pupil position of some input
        rays, trace those rays through the system and return the result,
        including all intermediate rays.

        Parameters
        ----------
        wavelength
            The wavelengths of the input rays.
            If :obj:`None` (the default), ``self.grid_input.wavelength``
            will be used.
        field
            The field positions of the input rays, in either normalized or physical units.
            If :obj:`None` (the default), ``self.grid_input.field``
            will be used.
        pupil
            The pupil positions of the input rays, in either normalized or physical units.
            If :obj:`None` (the default), ``self.grid_input.pupil``
            will be used.
        axis
            The axis along which the rays are accumulated.
            If :obj:`None` (the default), :attr:`axis_surface` will be used.

        See Also
        --------
        rayfunction : Similar to `raytrace` except it only returns the rays at the last surface.
        """

        if axis is None:
            axis = self.axis_surface

        if (wavelength is None) and (field is None) and (pupil is None):
            result = self._rayfunction_input.copy_shallow()
        else:
            grid_input = self.grid_input.copy_shallow()
            if wavelength is not None:
                grid_input.wavelength = wavelength
            if field is not None:
                grid_input.field = field
            if pupil is not None:
                grid_input.pupil = pupil
            result = self._calc_rayfunction_input(grid_input)

        rays = result.outputs
        if self.transformation is not None:
            rays = self.transformation.inverse(rays)
        result.outputs = optika.propagators.accumulate_rays(
            propagators=self.surfaces_all,
            rays=rays,
            axis=axis,
        )
        return result

    def rayfunction(
        self,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
    ) -> optika.rays.RayFunctionArray:
        """
        Given the wavelength, field position, and pupil position of some input
        rays, trace those rays through the system and return the resulting
        rays at the last surface.

        Parameters
        ----------
        wavelength
            The wavelengths of the input rays.
            If :obj:`None` (the default), ``self.grid_input.wavelength``
            will be used.
        field
            The field positions of the input rays, in either normalized or physical units.
            If :obj:`None` (the default), ``self.grid_input.field``
            will be used.
        pupil
            The pupil positions of the input rays, in either normalized or physical units.
            If :obj:`None` (the default), ``self.grid_input.pupil``
            will be used.

        See Also
        --------
        raytrace : Similar to `rayfunction` except it computes all the intermediate rays.
        """

        axis = "_dummy"
        raytrace = self.raytrace(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            axis=axis,
        )
        return raytrace[{axis: ~0}]

    @functools.cached_property
    def rayfunction_default(self) -> optika.rays.RayFunctionArray:
        """
        Computes the rays at the last surface in the system as a function of
        input wavelength and position using :attr:`grid_input`.
        """
        return self.rayfunction()

    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        plot_rays: bool = True,
        plot_rays_vignetted: bool = False,
        kwargs_rays: None | dict[str, Any] = None,
        **kwargs,
    ) -> na.AbstractScalar | dict[str, na.AbstractScalar]:
        """
        Plot the surfaces of the system and the default raytrace.

        Parameters
        ----------
        ax
            The matplotlib axes on which to plot the system
        transformation
            Any additional transformation to apply to the system before plotting.
        components
            The vector components to plot if `ax` is 2-dimensional.
        plot_rays
            Boolean flag indicating whether to plot the rays.
        plot_rays_vignetted
            Boolean flag indicating whether to plot the vignetted rays.
        kwargs_rays
            Any additional keyword arguments to use when plotting the rays.
        kwargs
            Any additional keyword arguments to use when plotting the surfaces.
        """

        surfaces = self.surfaces_all
        transformation_self = self.transformation
        kwargs_plot = self.kwargs_plot

        if transformation is not None:
            if transformation_self is not None:
                transformation = transformation @ transformation_self
        else:
            transformation = transformation_self

        if kwargs_plot is not None:
            kwargs = kwargs | kwargs_plot

        result = dict()

        result["surfaces"] = []
        for surface in surfaces:
            result["surfaces"].append(
                surface.plot(
                    ax=ax,
                    transformation=transformation,
                    components=components,
                    **kwargs,
                )
            )

        if plot_rays:
            raytrace = self.raytrace()

            if kwargs_rays is None:
                kwargs_rays = dict()

            where = True
            if not plot_rays_vignetted:
                where = raytrace.outputs.unvignetted[{self.axis_surface: ~0}]

            result["rays"] = na.plt.plot(
                raytrace.outputs.position,
                ax=ax,
                axis=self.axis_surface,
                where=where,
                transformation=transformation,
                components=components,
                **kwargs_rays,
            )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class SequentialSystem(
    AbstractSequentialSystem,
):
    """
    A sequential optical system is a composition of a sequence of
    :class:`optika.surfaces.AbstractSurface` instances and a default
    grid to sample them with.

    Examples
    --------

    Here is an example of a simple Newtonian telescope, with a parabolic
    primary mirror, a 45 degree fold mirror, and a detector.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # store the z-coordinate of the primary  and fold mirror
        # so that we can determine the detector position
        primary_mirror_z = 300 * u.mm
        fold_mirror_z = 50 * u.mm

        # define the parabolic primary mirror
        primary_mirror = optika.surfaces.Surface(
            name="mirror",
            sag=optika.sags.ParabolicSag(-300 * u.mm),
            aperture=optika.apertures.RectangularAperture(40 * u.mm),
            material=optika.materials.Mirror(),
            is_pupil_stop=True,
            transformation=na.transformations.Cartesian3dTranslation(
                z=primary_mirror_z,
            ),
        )

        # define the flat fold mirror which directs light
        # to the detector surface
        fold_mirror = optika.surfaces.Surface(
            name="fold_mirror",
            aperture=optika.apertures.CircularAperture(15 * u.mm),
            material=optika.materials.Mirror(),
            transformation=na.transformations.TransformationList([
                na.transformations.Cartesian3dRotationX(45 * u.deg),
                na.transformations.Cartesian3dTranslation(
                    z=fold_mirror_z,
                ),
            ]),
        )

        # define the central obscuration, the back face
        # of the fold mirror which blocks some of the
        # incoming light
        obscuration = optika.surfaces.Surface(
            name="obscuration",
            aperture=optika.apertures.CircularAperture(
                radius=fold_mirror.aperture.radius,
                inverted=True,
            ),
            transformation=fold_mirror.transformation,
        )

        # define the imaging sensor surface
        sensor = optika.sensors.IdealImagingSensor(
            name="sensor",
            width_pixel=5 * u.um,
            num_pixel=na.Cartesian2dVectorArray(1024, 2048),
            transformation=na.transformations.TransformationList([
                na.transformations.Cartesian3dRotationX(90 * u.deg),
                na.transformations.Cartesian3dTranslation(
                    y=primary_mirror.sag.focal_length + (primary_mirror_z - fold_mirror_z),
                    z=fold_mirror_z,
                ),
            ]),
            is_field_stop=True,
        )

        # define the grid of normalized field coordinates,
        # which are in the range +/-0.99 because the range +/-1
        # would be clipped by the edge of the detector
        field = 0.99 * na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
        )

        # define the grid of normalized pupil coordinates
        # in a similar fashion to the normalized field
        # coordinates
        pupil = 0.99 * na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=5,
        )

        # define the optical system using the surfaces
        # and the normalized field/pupil coordinates
        system = optika.systems.SequentialSystem(
            surfaces=[
                obscuration,
                primary_mirror,
                fold_mirror,
            ],
            sensor=sensor,
            grid_input=optika.vectors.ObjectVectorArray(
                wavelength=500 * u.nm,
                field=field,
                pupil=pupil,
            ),
        )

        # plot the system
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            system.plot(
                ax=ax,
                components=("z", "y"),
                kwargs_rays=dict(
                    color="tab:blue",
                ),
                color="black",
                zorder=10,
            )
    """

    surfaces: Sequence[optika.surfaces.AbstractSurface] = dataclasses.MISSING
    """
    a sequence of surfaces representing this optical system.

    At least one of these surfaces needs to be marked as the pupil surface,
    and if the object surface is not marked as the field stop, one of these
    surfaces needs to be marked as the field stop.
    """

    object: optika.surfaces.AbstractSurface = None
    """
    The external object being imaged or illuminated by this system.

    If :obj:`None`, the external object is assumed to be at infinity.
    """

    sensor: optika.sensors.AbstractImagingSensor = None
    """
    The imaging sensor that measures the light captured by this system.

    This is the last surface in the optical system.
    """

    axis_surface: str = "surface"
    """
    The name of the logical axis representing the sequence of surfaces.
    """

    grid_input: optika.vectors.ObjectVectorArray = None
    """
    The input grid to sample with rays.

    This grid is simultaneously projected onto both the field stop and the
    pupil stop.

    Positions on the stop can be specified in either absolute or normalized
    coordinates. Using normalized coordinates allows for injecting different
    grid types (cylindrical, stratified random, etc.) without specifying the
    scale of the stop surface.

    If positions are specified in absolute units, they are measured in the
    coordinate system of the corresponding stop surface.
    """

    transformation: None | na.transformations.AbstractTransformation = None
    """
    A optional coordinate transformation to apply to the entire optical system.
    """

    kwargs_plot: None | dict[str, Any] = None
    """
    Additional keyword arguments used by default in :meth:`plot`.
    """
