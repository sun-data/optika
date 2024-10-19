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
    optika.mixins.Shaped,
):
    """
    An interface describing an optical system.

    Could potentially be sequential or non-sequential.
    """

    @abc.abstractmethod
    def __call__(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        **kwargs: Any,
    ) -> na.SpectralPositionalVectorArray:
        """
        Forward model of the optical system.
        Maps the given spectral radiance of a scene to detector counts.

        Parameters
        ----------
        scene
            The spectral radiance of the scene as a function of wavelength
            and field position
        kwargs
            Additional keyword arguments used by subclass implementations
            of this method.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSequentialSystem(
    AbstractSystem,
):
    """
    An interface describing a sequential optical system.

    A sequential optical system is a system where the ordering of the optical
    surfaces is known in advance.
    """

    @property
    @abc.abstractmethod
    def object(self) -> optika.surfaces.AbstractSurface:
        """
        The external object being imaged or illuminated by this system.

        If :obj:`None`, the external object is assumed to be at infinity.
        """

    @property
    def object_is_at_infinity(self) -> bool:
        """
        A boolean flag indicating if the object is at infinity.

        If :attr:`object` doesn't have an aperture,
        it is assumed that the object is at infinity.
        """
        aperture_obj = self.object.aperture
        if aperture_obj is not None:
            unit_obj = na.unit_normalized(aperture_obj.bound_lower)
            if unit_obj.is_equivalent(u.mm):
                result = False
            elif unit_obj.is_equivalent(u.dimensionless_unscaled):
                result = True
            else:  # pragma: nocover
                raise ValueError(f"Unrecognized unit for object aperture, {unit_obj}")
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

        obj = subsystem[~0]
        rays = result.outputs
        if obj.transformation is not None:
            rays = obj.transformation.inverse(rays)

        where = rays.direction @ obj.sag.normal(rays.position) > 0
        result.outputs.direction[where] = -result.outputs.direction[where]

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
            angles = optika.angles(self.rayfunction_stops.outputs.direction)
            return angles.min(axis)
        else:
            return self.rayfunction_stops.outputs.position.xy.min(axis)

    @property
    def field_max(self) -> na.AbstractCartesian2dVectorArray:
        """
        The upper right corner of this optical system's field of view.
        """
        axis = (self._axis_field_stop, self._axis_pupil_stop)
        if self.object_is_at_infinity:
            angles = optika.angles(self.rayfunction_stops.outputs.direction)
            return angles.max(axis)
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
            angles = optika.angles(self.rayfunction_stops.outputs.direction)
            return angles.min(axis)

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
            angles = optika.angles(self.rayfunction_stops.outputs.direction)
            return angles.max(axis)

    def _denormalize_grid(
        self,
        grid: optika.vectors.ObjectVectorArray,
        normalized_field: bool = True,
        normalized_pupil: bool = True,
    ) -> optika.vectors.ObjectVectorArray:

        if (not normalized_field) and (not normalized_pupil):
            return grid

        axis_field = self._axis_field_stop
        axis_pupil = self._axis_pupil_stop

        rayfunction_stops = self._calc_rayfunction_stops(
            wavelength_input=grid.wavelength,
            axis_pupil_stop=axis_pupil,
            axis_field_stop=axis_field,
            samples_pupil_stop=21,
            samples_field_stop=21,
        )

        result = grid.copy_shallow()

        object_is_at_infinity = self.object_is_at_infinity
        if object_is_at_infinity:
            field = rayfunction_stops.outputs.direction.xy
            pupil = rayfunction_stops.outputs.position.xy
        else:
            field = rayfunction_stops.outputs.position.xy
            pupil = rayfunction_stops.outputs.direction.xy

        if normalized_field:
            min_field = field.min(axis=(axis_field, axis_pupil))
            ptp_field = field.ptp(axis=(axis_field, axis_pupil))
            result.field = ptp_field * (result.field + 1) / 2 + min_field

            if object_is_at_infinity:
                direction = na.Cartesian3dVectorArray(
                    x=result.field.x,
                    y=result.field.y,
                    z=np.sqrt(
                        1 - np.square(result.field.x) - np.square(result.field.y)
                    ),
                )
                result.field = optika.angles(direction)

        if normalized_pupil:
            min_pupil = pupil.min(axis=(axis_field, axis_pupil))
            ptp_pupil = pupil.ptp(axis=(axis_field, axis_pupil))
            result.pupil = ptp_pupil * (result.pupil + 1) / 2 + min_pupil

            if not object_is_at_infinity:
                direction = na.Cartesian3dVectorArray(
                    x=result.pupil.x,
                    y=result.pupil.y,
                    z=np.sqrt(
                        1 - np.square(result.pupil.x) - np.square(result.pupil.y)
                    ),
                )
                result.pupil = optika.angles(direction)

        return result

    def _calc_rayfunction_input(
        self,
        grid_input: optika.vectors.ObjectVectorArray,
        normalized_field: bool = True,
        normalized_pupil: bool = True,
    ) -> optika.rays.RayFunctionArray:

        grid_input = self._denormalize_grid(
            grid=grid_input,
            normalized_field=normalized_field,
            normalized_pupil=normalized_pupil,
        )

        if self.object_is_at_infinity:
            position = grid_input.pupil
            direction = optika.direction(grid_input.field)
        else:
            position = grid_input.field
            direction = optika.direction(grid_input.pupil)

        rays = optika.rays.RayVectorArray(
            wavelength=grid_input.wavelength,
            position=na.Cartesian3dVectorArray(
                x=position.x,
                y=position.y,
                z=0 * position.x.unit,
            ),
            direction=direction,
        )

        obj = self.object
        if obj is not None:
            if obj.transformation is not None:
                rays = obj.transformation(rays)

        result = optika.rays.RayFunctionArray(inputs=grid_input, outputs=rays)

        return result

    @functools.cached_property
    def _rayfunction_input(self) -> optika.rays.RayFunctionArray:
        return self._calc_rayfunction_input(
            grid_input=self.grid_input,
        )

    def raytrace(
        self,
        intensity: None | float | u.Quantity | na.AbstractScalar = None,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
        axis: None | str = None,
        normalized_field: bool = True,
        normalized_pupil: bool = True,
    ) -> optika.rays.RayFunctionArray:
        """
        Given the wavelength, field position, and pupil position of some input
        rays, trace those rays through the system and return the result in
        global coordinates, including all intermediate rays.

        Parameters
        ----------
        intensity
            The energy density of the input rays.
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
        normalized_field
            A boolean flag indicating whether the `field` parameter is given
            in normalized or physical units.
        normalized_pupil
            A boolean flag indicating whether the `pupil` parameter is given
            in normalized or physical units.

        See Also
        --------
        rayfunction : Similar to `raytrace` except it only returns the rays at
            the last surface in local coordinates.
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
            result = self._calc_rayfunction_input(
                grid_input,
                normalized_field=normalized_field,
                normalized_pupil=normalized_pupil,
            )

        rays = result.outputs
        if intensity is not None:
            rays.intensity = intensity
        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        surfaces = self.surfaces_all

        result.outputs = optika.propagators.accumulate_rays(
            propagators=surfaces,
            rays=rays,
            axis=axis,
        )
        return result

    def rayfunction(
        self,
        intensity: None | float | u.Quantity | na.AbstractScalar = None,
        wavelength: None | u.Quantity | na.AbstractScalar = None,
        field: None | na.AbstractCartesian2dVectorArray = None,
        pupil: None | na.AbstractCartesian2dVectorArray = None,
        normalized_field: bool = True,
        normalized_pupil: bool = True,
    ) -> optika.rays.RayFunctionArray:
        """
        Given the wavelength, field position, and pupil position of some input
        rays, trace those rays through the system and return the resulting
        rays in local coordinates at the last surface.

        Parameters
        ----------
        intensity
            The energy density of the input rays.
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
        normalized_field
            A boolean flag indicating whether the `field` parameter is given
            in normalized or physical units.
        normalized_pupil
            A boolean flag indicating whether the `pupil` parameter is given
            in normalized or physical units.

        See Also
        --------
        raytrace : Similar to `rayfunction` except it computes all the
            intermediate rays, and it returns results in global coordinates.
        """

        axis = "_dummy"
        raytrace = self.raytrace(
            intensity=intensity,
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            axis=axis,
            normalized_field=normalized_field,
            normalized_pupil=normalized_pupil,
        )
        rayfunction = raytrace[{axis: ~0}]
        rays = rayfunction.outputs

        if self.sensor.transformation is not None:
            rays = self.sensor.transformation.inverse(rays)

        rayfunction.outputs = rays

        return rayfunction

    @functools.cached_property
    def rayfunction_default(self) -> optika.rays.RayFunctionArray:
        """
        Computes the rays in local coordinates at the last surface in the system
        as a function of input wavelength and position using :attr:`grid_input`.

        This property is cached to increase performance.
        If :attr:`grid_input` is updated, the cache must be cleared with
        ``del system.rayfunction_default`` before calling property.
        """
        return self.rayfunction()

    @classmethod
    def _avg_left_right(cls, a: na.AbstractArray, axis: str) -> na.AbstractArray:
        a_left = a[{axis: slice(None, ~0)}]
        a_right = a[{axis: slice(1, None)}]
        return (a_left + a_right) / 2

    @classmethod
    def _lerp(
        cls,
        i: float | na.AbstractScalar,
        a0: float | na.AbstractScalar,
        a1: float | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return a0 * (1 - i) + a1 * i

    @classmethod
    def _nlerp(
        cls,
        i: dict[str, na.AbstractScalar],
        a: na.AbstractArray,
    ) -> na.AbstractArray:

        axis = next(iter(i))

        i_new = {ax: i[ax] for ax in i if ax != axis}

        a0 = a[{axis: slice(None, ~0)}]
        a1 = a[{axis: slice(1, None)}]

        if i_new:
            a0 = cls._nlerp(i_new, a0)
            a1 = cls._nlerp(i_new, a1)

        return cls._lerp(i[axis], a0, a1)

    def _rayfunction_from_vertices(
        self,
        radiance: na.AbstractScalar,
        wavelength: na.AbstractScalar,
        field: na.AbstractCartesian2dVectorArray,
        pupil: na.AbstractCartesian2dVectorArray,
        axis_wavelength: str,
        axis_field: tuple[str, str],
        axis_pupil: tuple[str, str],
        normalized_field: bool = True,
        normalized_pupil: bool = True,
    ) -> optika.rays.RayFunctionArray:
        """
        Similar to :meth:`rayfunction` except that the wavelength, field, and
        pupil positions are specified on cell vertices instead of cell centers.

        This function uses the vertices to compute both the area and centroid of
        each cell before calling :meth:`rayfunction` and returning the result.

        Parameters
        ----------
        radiance
            The <spectral radiance https://en.wikipedia.org/wiki/Spectral_radiance>_
            of each field position.
            This will be converted into a flux before being passed into
            :meth:`rayfunction`.
        wavelength
            The vertices of the wavelength grid, unless `area_wavelength` is
            specified.
        field
            The vertices of the field grid (in either normalized or physical units),
            unless `area_field` is specified.
        pupil
            The vertices of the pupil grid (in either normalized or physical units),
            unless `area_pupil` is specified.
        axis_wavelength
            The logical axis corresponding to changing wavelength.
            This axis should only be present in the `wavelength` and `pupil`
            parameters.
        axis_field
            The two logical axes corresponding to changing field positions.
            These axes should only be present in the `field` and `pupil`
            parameters.
        axis_pupil
            The two logical axes corresponding to changing pupil positions.
            These axes should only be present in the `pupil` parameter.
        area_wavelength
            An optional parameter specifying the width of each wavelength cell.
            If :obj:`None` (the default), the `wavelength` parameter is assumed
            to be specified on cell vertices, otherwise the `wavelength` parameter
            is assumed to be specified on cell centers.
        area_field
            An optional parameter specifying the area of each field position.
            If :obj:`None` (the default), the `field` parameter is assumed
            to be specified on cell vertices, otherwise the `field` parameter
            is assumed to be specified on cell centers.
        area_pupil
            An optional parameter specifying the area of each pupil position.
            If :obj:`None` (the default), the `pupil` parameter is assumed
            to be specified on cell vertices, otherwise the `pupil` parameter
            is assumed to be specified on cell centers.
        normalized_field
            A boolean flag indicating if the `field` parameter is in normalized
            units.
        normalized_pupil
            A boolean flag indicating if the `pupil` parameter is in normalized
            units.
        """

        grid = optika.vectors.ObjectVectorArray(
            wavelength=wavelength,
            field=field,
            pupil=pupil,
        )

        grid = self._denormalize_grid(
            grid=grid,
            normalized_field=normalized_field,
            normalized_pupil=normalized_pupil,
        )

        wavelength = grid.wavelength
        field = grid.field
        pupil = grid.pupil

        shape_self = self.shape
        shape_wavelength = wavelength.shape
        shape_field = field.shape
        shape_pupil = pupil.shape

        shape = na.broadcast_shapes(
            shape_self,
            shape_wavelength,
            shape_field,
            shape_pupil,
        )

        if axis_wavelength not in shape_wavelength:
            raise ValueError(  # pragma: nocover
                f"{axis_wavelength=} must be in {shape_wavelength=}",
            )
        if not set(axis_field).issubset(shape_field):
            raise ValueError(  # pragma: nocover
                f"{axis_field=} must be a subset of {shape_field=}",
            )
        if set(axis_field).intersection(shape_wavelength):
            raise ValueError(  # pragma: nocover
                f"{axis_field=} must not intersect {shape_wavelength=}"
            )
        if not set(axis_pupil).issubset(shape_pupil):
            raise ValueError(  # pragma: nocover
                f"{axis_pupil=} must be a subset of {shape_pupil=}",
            )
        if set(axis_pupil).intersection(shape_wavelength | shape_field):
            raise ValueError(  # pragma: nocover
                f"{axis_pupil=} must not intersect {shape_wavelength=} or {shape_field=}"
            )

        axis_field_x, axis_field_y = axis_field
        axis_pupil_x, axis_pupil_y = axis_pupil

        area_wavelength = wavelength.volume_cell(axis_wavelength)

        shape_field = na.broadcast_shapes(
            shape_wavelength,
            shape_field,
        )
        field = field.broadcast_to(shape_field)

        shape_pupil = na.broadcast_shapes(shape_wavelength, shape_field, shape_pupil)
        pupil = pupil.broadcast_to(shape_pupil)

        if self.object_is_at_infinity:
            area_field = optika.direction(field).solid_angle_cell(axis_field)
            area_pupil = pupil.volume_cell(axis_pupil)
        else:
            area_field = field.volume_cell(axis_field)
            area_pupil = optika.direction(pupil).solid_angle_cell(axis_pupil)

        area_field = self._avg_left_right(area_field, axis_wavelength)

        area_pupil = self._avg_left_right(area_pupil, axis_wavelength)
        area_pupil = self._avg_left_right(area_pupil, axis_field_x)
        area_pupil = self._avg_left_right(area_pupil, axis_field_y)

        axis = (axis_wavelength,) + axis_field + axis_pupil
        shape_centers = {ax: shape[ax] - 1 if ax in axis else shape[ax] for ax in shape}

        wavelength = self._nlerp(
            i={axis_wavelength: na.random.uniform(0, 1, shape_random=shape_centers)},
            a=wavelength,
        )

        field = self._nlerp(
            i={
                axis_wavelength: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_field_x: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_field_y: na.random.uniform(0, 1, shape_random=shape_centers),
            },
            a=field,
        )

        pupil = self._nlerp(
            i={
                axis_wavelength: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_field_x: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_field_y: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_pupil_x: na.random.uniform(0, 1, shape_random=shape_centers),
                axis_pupil_y: na.random.uniform(0, 1, shape_random=shape_centers),
            },
            a=pupil,
        )

        flux = radiance * area_wavelength * area_field * area_pupil

        return self.rayfunction(
            intensity=flux,
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            normalized_field=False,
            normalized_pupil=False,
        )

    def __call__(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        grid_pupil: None | na.AbstractCartesian2dVectorArray = None,
        **kwargs,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:

        shape = self.shape

        scene = scene.explicit

        wavelength = scene.inputs.wavelength
        field = scene.inputs.position

        pupil = grid_pupil
        if pupil is None:
            pupil = self.grid_input.pupil

        unit_field = na.unit_normalized(field)
        unit_pupil = na.unit_normalized(pupil)

        normalized_field = unit_field.is_equivalent(u.dimensionless_unscaled)
        normalized_pupil = unit_pupil.is_equivalent(u.dimensionless_unscaled)

        shape_wavelength = na.broadcast_shapes(shape, wavelength.shape)
        shape_field = na.broadcast_shapes(shape, field.shape)
        shape_pupil = na.broadcast_shapes(shape, pupil.shape)

        shape_wavelength = {
            axis: shape_wavelength[axis]
            for axis in shape_wavelength
            if axis not in shape
        }
        shape_field = {
            axis: shape_field[axis]
            for axis in shape_field
            if axis not in shape | shape_wavelength
        }
        shape_pupil = {
            axis: shape_pupil[axis]
            for axis in shape_pupil
            if axis not in shape | shape_wavelength | shape_field
        }

        (axis_wavelength,) = tuple(shape_wavelength)
        axis_field = tuple(shape_field)
        axis_pupil = tuple(shape_pupil)

        rayfunction = self._rayfunction_from_vertices(
            radiance=scene.outputs,
            wavelength=wavelength,
            field=field,
            pupil=pupil,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            axis_pupil=axis_pupil,
            normalized_field=normalized_field,
            normalized_pupil=normalized_pupil,
        )

        return self.sensor.readout(
            rays=rayfunction.outputs,
            axis=tuple(shape_field | shape_pupil),
        )

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

        import dataclasses
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # store the coordinates of the primary mirror, fold mirror,
        # and sensor, so we can determine the focal length of the
        # primary mirror.
        primary_mirror_z = 200 * u.mm
        fold_mirror_z = 50 * u.mm
        sensor_x = 50 * u.mm

        # Define the front aperture surface.
        front = optika.surfaces.Surface(
            name="front",
        )

        # Define the parabolic primary mirror.
        primary_mirror = optika.surfaces.Surface(
            name="mirror",
            sag=optika.sags.ParabolicSag(
                focal_length=-(primary_mirror_z - fold_mirror_z + sensor_x),
            ),
            aperture=optika.apertures.RectangularAperture(40 * u.mm),
            material=optika.materials.Mirror(),
            is_pupil_stop=True,
            transformation=na.transformations.Cartesian3dTranslation(
                z=primary_mirror_z,
            ),
        )

        # Define the flat fold mirror which directs light
        # to the detector surface.
        fold_mirror = optika.surfaces.Surface(
            name="fold_mirror",
            aperture=optika.apertures.RectangularAperture(25 * u.mm),
            material=optika.materials.Mirror(),
            transformation=na.transformations.TransformationList([
                na.transformations.Cartesian3dRotationY((90 + 45) * u.deg),
                na.transformations.Cartesian3dTranslation(
                    z=fold_mirror_z,
                ),
            ]),
        )

        # Define the central obscuration, the back face
        # of the fold mirror which blocks some of the
        # incoming light.
        obscuration = optika.surfaces.Surface(
            name="obscuration",
            aperture=dataclasses.replace(fold_mirror.aperture, inverted=True),
            transformation=fold_mirror.transformation,
        )

        # define the imaging sensor surface
        sensor = optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=20 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(128, 128),
            timedelta_exposure=1 * u.s,
            transformation=na.transformations.TransformationList([
                na.transformations.Cartesian3dRotationY(-90 * u.deg),
                na.transformations.Cartesian3dTranslation(
                    x=-sensor_x,
                    z=fold_mirror_z,
                ),
            ]),
            is_field_stop=True,
        )

        # Define the grid of normalized field coordinates,
        field = na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
            centers=True,
        )

        # Define the grid of normalized pupil coordinates
        # in a similar fashion to the normalized field
        # coordinates
        pupil = na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=5,
            centers=True,
        )

        # define the optical system using the surfaces
        # and the normalized field/pupil coordinates
        system = optika.systems.SequentialSystem(
            surfaces=[
                front,
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
            fig, ax = plt.subplots(constrained_layout=True)
            ax.set_aspect("equal")
            system.plot(
                ax=ax,
                components=("z", "x"),
                kwargs_rays=dict(
                    color="tab:blue",
                ),
                color="black",
                zorder=10,
            )

    |

    Using this model, we can simulate an image of an airforce target

    .. jupyter-execute::

        # Define the number of points to sample
        num_field = 2 * system.sensor.num_pixel

        # Define the scene as an airforce target.
        # Note how the coordinates (inputs) are defined on
        # cell vertices and the values (outputs) are
        # defined on cell centers.
        scene = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=na.linspace(530, 531, axis="wavelength", num=2) * u.nm,
                position=na.Cartesian2dVectorLinearSpace(
                    start=system.field_min,
                    stop=system.field_max,
                    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                    num=num_field + 1,
                ),
            ),
            outputs=optika.targets.airforce(
                axis_x="field_x",
                axis_y="field_y",
                num_x=num_field.x,
                num_y=num_field.y,
            ) * 100 * u.photon / u.s / u.m ** 2 / u.arcsec ** 2 / u.nm,
        )

        # Simulate an image of the scene using the optical system
        image = system(scene)

        # Plot the original scene and the simulated image
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                figsize=(8, 5),
                constrained_layout=True,
            )
            mappable_scene = na.plt.pcolormesh(
                scene.inputs.position,
                C=scene.outputs.value,
                ax=ax[0],
            )
            mappable_image = na.plt.pcolormesh(
                image.inputs.position.mean("wavelength"),
                C=image.outputs.value.sum("wavelength"),
                ax=ax[1],
            )
            cbar_0 = fig.colorbar(
                mappable=mappable_scene.ndarray.item(),
                ax=ax[0],
                location="top",
            )
            cbar_0.set_label(f"radiance ({scene.outputs.unit:latex_inline})")
            cbar_1 = fig.colorbar(
                mappable=mappable_image.ndarray.item(),
                ax=ax[1],
                location="top",
            )
            cbar_1.set_label(f"measured charge ({image.outputs.unit:latex_inline})")
            ax[0].set_aspect("equal")
            ax[1].set_aspect("equal")

    The result is flipped vertically and horizontally due to the layout
    of the optical system.
    The noise on the image is from the stratified random sampling used to
    generate the grid of rays traced through the system, there is no
    additional noise sources, such as photon shot noise in this simulation.
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

    If :obj:`None`, the object is assumed to be an empty surface at the origin.
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

    def __post_init__(self):
        if self.object is None:
            self.object = optika.surfaces.Surface()

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            *[optika.shape(surface) for surface in self.surfaces],
            optika.shape(self.object),
            optika.shape(self.sensor),
            optika.shape(self.transformation),
        )
