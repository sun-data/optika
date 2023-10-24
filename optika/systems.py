from __future__ import annotations
from typing import Sequence, TypeVar, Generic, Callable, Any
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
        if not any(s.is_field_stop for s in result):
            result[0] = dataclasses.replace(result[0], is_field_stop=True)
        return result

    @property
    @abc.abstractmethod
    def grid_input_normalized(self) -> optika.vectors.ObjectVectorArray:
        """
        The input grid to sample with rays.

        This grid is projected onto the 3 possible types of stops: field, pupil,
        and spectral.

        Positions on the stop can be specified in either absolute or normalized
        coordinates. Using normalized coordinates allows for injecting different
        grid types (cylindrical, stratified random, etc.) without specifying the
        scale of the stop surface.

        If positions are specified in absolute units, they are measured in the
        coordinate system of the corresponding stop surface
        """

    @property
    def _indices_field_stop(self) -> list[int]:
        return [i for i, s in enumerate(self.surfaces_all) if s.is_field_stop]

    @property
    def _indices_pupil_stop(self) -> list[int]:
        return [i for i, s in enumerate(self.surfaces_all) if s.is_pupil_stop]

    @property
    def index_field_stop(self) -> int:
        indices = self._indices_field_stop
        if not indices:
            raise ValueError(
                "Field stop is not defined for this system."
                "Set `is_field_stop=True` for at least one surface in this system."
            )
        return indices[~0]

    @property
    def index_pupil_stop(self) -> int:
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
        return self.surfaces_all[self.index_field_stop]

    @property
    def pupil_stop(self) -> optika.surfaces.AbstractSurface:
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

    def _calc_rayfunction_input_stops_only(
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
                zfunc = lambda xy: np.sqrt(1 - np.square(xy.length))
            elif na.unit(grid_first).is_equivalent(u.dimensionless_unscaled):
                result.outputs.direction = na.Cartesian3dVectorArray(
                    x=grid_first.x,
                    y=grid_first.y,
                    z=np.sqrt(1 - np.square(grid_first.length)),
                )
                result.outputs.position = na.Cartesian3dVectorArray() * u.mm
                component_variable = "position"
                zfunc = lambda xy: surface_first.sag(xy)
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

    def _calc_rayfunction_input_stops(
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

        rays_stop = self._calc_rayfunction_input_stops_only(
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

    def _calc_rayfunction_input(
        self,
        grid_input: optika.vectors.ObjectVectorArray,
    ) -> optika.rays.RayFunctionArray:
        rayfunction_inputs_stops = self._calc_rayfunction_input_stops(
            wavelength_input=grid_input.wavelength,
            axis_pupil_stop="wire_pupil",
            axis_field_stop="wire_field_stop",
            samples_pupil_stop=21,
            samples_field_stop=21,
        )

        # return rayfunction_inputs_stops

        obj = self.object
        if obj is not None:
            aperture_obj = obj.aperture
            if aperture_obj is not None:
                unit_obj = na.unit_normalized(aperture_obj.bound_lower)
                if unit_obj.is_equivalent(u.mm):
                    object_at_infinity = False
                else:
                    object_at_infinity = True
            else:
                object_at_infinity = False
        else:
            object_at_infinity = True

        result = rayfunction_inputs_stops.copy_shallow()

        if object_at_infinity:
            field = rayfunction_inputs_stops.outputs.direction.xy
            pupil = rayfunction_inputs_stops.outputs.position.xy
        else:
            field = rayfunction_inputs_stops.outputs.position.xy
            pupil = rayfunction_inputs_stops.outputs.direction.xy

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

        if object_at_infinity:
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

        if obj is not None:
            if obj.transformation is not None:
                rays = obj.transformation(rays)

        result.outputs = rays

        return result

    @functools.cached_property
    def _rayfunction_input(self) -> optika.rays.RayFunctionArray:
        return self._calc_rayfunction_input(
            grid_input=self.grid_input_normalized,
        )

    def _calc_raytrace(
        self,
        rayfunction_input: optika.rays.RayFunctionArray,
        axis: str,
    ) -> optika.rays.RayFunctionArray:
        rays = rayfunction_input.outputs
        if self.transformation is not None:
            rays = self.transformation.inverse(rays)
        result = rayfunction_input.copy_shallow()
        result.outputs = optika.propagators.accumulate_rays(
            propagators=self.surfaces_all,
            rays=rays,
            axis=axis,
        )
        return result

    @property
    def raytrace(self) -> optika.rays.RayFunctionArray:
        return self._calc_raytrace(
            rayfunction_input=self._rayfunction_input,
            axis=self.axis_surface,
        )

    def _calc_rayfunction(
        self,
        rayfunction_input: optika.rays.RayFunctionArray,
        axis: str,
    ) -> optika.rays.RayFunctionArray:
        raytrace = self._calc_raytrace(
            rayfunction_input=rayfunction_input,
            axis=axis,
        )
        return raytrace[{axis: ~0}]

    @functools.cached_property
    def rayfunction(self) -> optika.rays.RayFunctionArray:
        return self._calc_rayfunction(
            rayfunction_input=self._rayfunction_input,
            axis=self.axis_surface,
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
            raytrace = self.raytrace

            if kwargs_rays is None:
                kwargs_rays = dict()

            where = True
            if not plot_rays_vignetted:
                intensity = raytrace.outputs[dict(surface=~0)].intensity
                where = intensity != 0

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

        # define the detector surface
        detector = optika.surfaces.Surface(
            name="detector",
            aperture=optika.apertures.RectangularAperture(5 * u.mm),
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
                detector,
            ],
            grid_input_normalized=optika.vectors.ObjectVectorArray(
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
    object: optika.surfaces.AbstractSurface = None
    axis_surface: str = "surface"
    grid_input_normalized: optika.vectors.ObjectVectorArray = None
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict[str, Any] = None
