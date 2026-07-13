import abc
import dataclasses
import functools
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
from ezdxf.addons.r12writer import R12FastStreamWriter

__all__ = [
    "AbstractAperture",
    "CircularAperture",
    "CircularSectorAperture",
    "EllipticalAperture",
    "AbstractPolygonalAperture",
    "PolygonalAperture",
    "RectangularAperture",
    "AbstractRegularPolygonalAperture",
    "RegularPolygonalAperture",
    "AbstractOctagonalAperture",
    "OctagonalAperture",
    "AbstractIsoscelesTrapezoidalAperture",
    "IsoscelesTrapezoidalAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractAperture(
    optika.mixins.DxfWritable,
    optika.mixins.Printable,
    optika.mixins.Plottable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    """
    An interface describing a generalized aperture.
    """

    samples_wire: int = dataclasses.field(default=101, kw_only=True)
    """The default number of samples used for :meth:`wire`."""

    active: bool | na.AbstractScalar = dataclasses.field(default=True, kw_only=True)
    """Whether the aperture is active and can clip rays."""

    inverted: bool | na.AbstractScalar = dataclasses.field(default=False, kw_only=True)
    """
    Whether this object is being used as an aperture or obscuration.
    
    If :obj:`True`, the interior of the aperture allows light to passthrough.
    If :obj:`False`, the exterior of the aperture allows light to pass through.
    """

    transformation: None | na.transformations.AbstractTransformation = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The transformation between the local surface coordinates and the aperture."""

    kwargs_plot: None | dict = dataclasses.field(default=None, kw_only=True)
    """
    Extra keyword arguments that will be used in the call to
    :func:`named_arrays.plt.plot` within the :meth:`plot` method.
    """

    @abc.abstractmethod
    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Check if a given point is inside the aperture.

        Parameters
        ----------
        position
            Points in surface coordinates.
        """

    def clip_rays(self, rays: optika.rays.RayVectorArray):
        """
        Given a set of input rays in surface coordinates,
        update the :attr:`~optika.rays.RayVectorArray.unvignetted` to be
        :obj:`False` if the ray is blocked by the aperture.

        Parameters
        ----------
        rays
            The input rays to clip.
        """
        unit = na.unit_normalized(self.bound_lower)
        if unit.is_equivalent(u.mm):
            mask = self(rays.position)
        elif unit.is_equivalent(u.dimensionless_unscaled):
            mask = self(rays.direction)
        else:
            raise ValueError(f"aperture with unit {unit} is not supported")
        rays = rays.copy_shallow()
        rays.unvignetted = rays.unvignetted & mask
        return rays

    @property
    @abc.abstractmethod
    def bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        """
        The lower-left corner of the aperture's rectangular footprint
        in surface coordinates.
        """

    @property
    @abc.abstractmethod
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        """
        The upper-right corner of the aperture's rectangular footprint
        in surface coordinates.
        """

    @abc.abstractmethod
    def wire(self, num: None | int = None) -> na.AbstractCartesian3dVectorArray:
        """
        A sequence of points representing this aperture in surface coordinates.

        Parameters
        ----------
        num
            The total number of samples that will be used to represent this
            wire.
        """

    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        sag: None | optika.sags.AbstractSag = None,
        **kwargs,
    ) -> None | na.ScalarArray[npt.NDArray[None | matplotlib.lines.Line2D]]:
        if ax is None:
            ax = plt.gca()
        ax = na.as_named_array(ax)

        wire = self.wire().explicit

        if not wire.length.unit_normalized.is_equivalent(u.mm):
            return None

        if sag is not None:
            wire.z = sag(wire)

        kwargs_plot = self.kwargs_plot
        if kwargs_plot is None:
            kwargs_plot = dict()

        kwargs = kwargs_plot | kwargs

        return na.plt.plot(
            wire,
            ax=ax,
            axis="wire",
            transformation=transformation,
            components=components,
            **kwargs,
        )

    def _write_to_dxf(
        self,
        dxf: R12FastStreamWriter,
        unit: u.Unit,
        transformation: None | na.transformations.AbstractTransformation = None,
        sag: None | optika.sags.AbstractSag = None,
        **kwargs,
    ) -> None:

        super()._write_to_dxf(
            dxf=dxf,
            unit=unit,
            transformation=transformation,
        )

        wire = self.wire()

        wire = wire.broadcast_to(wire.shape)

        unit_wire = na.unit_normalized(wire)
        if not unit_wire.is_equivalent(unit):
            return

        if sag is not None:
            wire.z = sag(wire)

        if transformation is not None:
            wire = transformation(wire)

        wire = na.nominal(wire.broadcasted)

        x = na.as_named_array(wire.x)
        y = na.as_named_array(wire.y)
        z = na.as_named_array(wire.z)

        for index in wire.ndindex(axis_ignored="wire"):

            vertices = np.stack(
                arrays=[
                    x[index].ndarray,
                    y[index].ndarray,
                    z[index].ndarray,
                ],
                axis=~0,
            )

            vertices = vertices.to_value(unit)

            dxf.add_polyline(
                vertices=vertices,
            )


@dataclasses.dataclass(eq=False, repr=False)
class CircularAperture(
    AbstractAperture,
):
    """
    A circular aperture or obscuration

    Examples
    --------

    Plot a single circular aperture

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        aperture = optika.apertures.CircularAperture(50 * u.mm)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")

    |

    Plot an array of circular apertures, similar to the configuration of the
    `Giant Magellan Telescope <https://en.wikipedia.org/wiki/Giant_Magellan_Telescope>`_.

    .. jupyter-execute::

        diameter = 8.417 * u.m
        radius = diameter / 2

        angle = na.linspace(0, 360, axis="segment", num=6, endpoint=False) * u.deg

        displacement = na.Cartesian3dVectorArray(
            x=diameter * np.cos(angle),
            y=diameter * np.sin(angle),
        )
        displacement = np.concatenate([
            na.Cartesian3dVectorArray().add_axes("segment") * u.mm,
            displacement
        ], axis="segment")

        aperture = optika.apertures.CircularAperture(
            radius=radius,
            transformation=na.transformations.Translation(displacement),
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The radius of the aperture."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        active = self.active
        inverted = self.inverted
        if self.transformation is not None:
            position = self.transformation.inverse(position)

        shape = na.shape_broadcasted(radius, active, inverted, position)

        radius = na.broadcast_to(radius, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask = position.xy.length <= radius

        mask[inverted] = ~mask[inverted]
        mask[~active] = True

        return mask

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result = result - self.radius
        return result

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result = result + self.radius
        return result

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire
        az = na.linspace(
            start=0 * u.deg,
            stop=360 * u.deg,
            axis="wire",
            num=num,
        )
        unit_radius = na.unit(self.radius)
        result = na.Cartesian3dVectorArray(
            x=self.radius * np.cos(az),
            y=self.radius * np.sin(az),
            z=0 * unit_radius if unit_radius is not None else 0,
        )
        if self.transformation is not None:
            result = self.transformation(result)
        return result


@dataclasses.dataclass(eq=False, repr=False)
class CircularSectorAperture(
    AbstractAperture,
):
    """
    A `circular sector <https://en.wikipedia.org/wiki/Circular_sector>`_
    aperture.

    Examples
    --------

    Plot a single circular aperture sector

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define a circular aperture sector
        aperture = optika.apertures.CircularSectorAperture(
            radius=50 * u.mm,
            angle_start=-11 * u.deg,
            angle_stop=40 * u.deg,
        )

        # Define points to sample the aperture with
        points = na.Cartesian3dVectorLinearSpace(
            start=aperture.bound_lower,
            stop=aperture.bound_upper,
            axis=na.Cartesian3dVectorArray("x", "y", "z"),
            num=na.Cartesian3dVectorArray(11, 11, 1),
        )

        # Compute which points are inside the aperture
        where = aperture(points)

        # Plot the circular aperture sector
        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")
            na.plt.scatter(
                points.x,
                points.y,
                c=where.astype(float)
            )
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the cirucular sector.
    """

    angle_start: u.Quantity | na.AbstractScalar = 0 * u.deg
    r"""
    The starting angle of the circular sector.
    Must be between :math:`-2 \pi` and :math:`+2 \pi` radians.
    """

    angle_stop: u.Quantity | na.AbstractScalar = 180 * u.deg
    r"""
    The ending angle of the circular sector.
    Must be between :math:`-2 \pi` and :math:`+2 \pi` radians and 
    counterclockwise from `angle_start`.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.angle_start),
            optika.shape(self.angle_stop),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        angle_start = self.angle_start
        angle_stop = self.angle_stop
        active = self.active
        inverted = self.inverted
        if self.transformation is not None:
            position = self.transformation.inverse(position)

        shape = na.shape_broadcasted(
            radius,
            angle_start,
            angle_stop,
            active,
            inverted,
            position,
        )

        radius = na.broadcast_to(radius, shape)
        angle_start = na.broadcast_to(angle_start, shape)
        angle_stop = na.broadcast_to(angle_stop, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask_radius = position.xy.length <= radius

        angle = np.arctan2(position.y, position.x)
        angle_positive = angle % (+2 * np.pi * u.rad)
        angle_negative = angle % (-2 * np.pi * u.rad)
        mask_positive = (angle_start < angle_positive) & (angle_positive < angle_stop)
        mask_negative = (angle_start < angle_negative) & (angle_negative < angle_stop)
        mask_angle = mask_positive | mask_negative

        mask = mask_radius & mask_angle

        mask[inverted] = ~mask[inverted]
        mask[~active] = True

        return mask

    def _bound_extrema(
        self,
    ) -> tuple[na.Cartesian3dVectorArray, na.Cartesian3dVectorArray]:
        """
        Compute the axis-aligned bounding box of this aperture analytically.

        The extremum of the sector along a given world axis is attained
        either at the apex, at one of the two endpoints of the arc, or at an
        interior point of the arc where the boundary is tangent to the world
        axis, if that point lies within the angular range of the sector.
        """
        radius = self.radius
        angle_start = self.angle_start
        angle_stop = self.angle_stop

        zero = 0 * radius
        apex = na.Cartesian3dVectorArray(x=zero, y=zero, z=zero)
        axis_a = na.Cartesian3dVectorArray(x=radius, y=zero, z=zero)
        axis_b = na.Cartesian3dVectorArray(x=zero, y=radius, z=zero)
        if self.transformation is not None:
            apex = self.transformation(apex)
            axis_a = self.transformation(axis_a) - apex
            axis_b = self.transformation(axis_b) - apex

        span = (angle_stop - angle_start) % (360 * u.deg)

        result_lower = na.Cartesian3dVectorArray()
        result_upper = na.Cartesian3dVectorArray()
        for c in ("x", "y", "z"):
            center = getattr(apex, c)
            coeff_a = getattr(axis_a, c)
            coeff_b = getattr(axis_b, c)

            point_start = center + coeff_a * np.cos(angle_start)
            point_start = point_start + coeff_b * np.sin(angle_start)
            point_stop = center + coeff_a * np.cos(angle_stop)
            point_stop = point_stop + coeff_b * np.sin(angle_stop)

            candidates = [center, point_start, point_stop]

            # interior extrema of the arc along this axis, kept only if they
            # lie within the angular range of the sector
            angle_critical = np.arctan2(coeff_b, coeff_a)
            if na.unit(angle_critical) is None:
                angle_critical = angle_critical * u.rad
            for angle in (angle_critical, angle_critical + 180 * u.deg):
                point_angle = center + coeff_a * np.cos(angle)
                point_angle = point_angle + coeff_b * np.sin(angle)
                where = ((angle - angle_start) % (360 * u.deg)) <= span
                candidates.append(np.where(where, point_angle, point_start))

            setattr(result_lower, c, functools.reduce(np.minimum, candidates))
            setattr(result_upper, c, functools.reduce(np.maximum, candidates))

        return result_lower, result_upper

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        lower, upper = self._bound_extrema()
        return lower

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        lower, upper = self._bound_extrema()
        return upper

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire

        unit_radius = na.unit(self.radius)
        z = 0 * unit_radius if unit_radius is not None else 0

        # The boundary of a circular sector has three segments: the two straight
        # radial arms (from the vertex out to the arc) and the arc itself.
        # Distribute the points evenly across all three segments -- like the
        # polygonal apertures -- so the entire boundary is sampled.  Sampling
        # only the arc would leave the radial arms unsampled.
        num_segments = 3
        num_per_segment = num / num_segments
        segments = []
        num_cumulative = 0
        for s in range(num_segments):
            num_s = int((s + 1) * num_per_segment - num_cumulative)
            num_cumulative += num_s
            t = na.linspace(
                start=0,
                stop=1,
                axis="wire",
                num=num_s,
                endpoint=num_cumulative == num,
            )
            if s == 0:  # radial arm from the vertex out to the start of the arc
                radius = self.radius * t
                angle = self.angle_start
            elif s == 1:  # the arc, from angle_start to angle_stop
                radius = self.radius
                angle = self.angle_start + (self.angle_stop - self.angle_start) * t
            else:  # radial arm from the end of the arc back to the vertex
                radius = self.radius * (1 - t)
                angle = self.angle_stop
            segments.append(
                na.Cartesian3dVectorArray(
                    x=radius * np.cos(angle),
                    y=radius * np.sin(angle),
                    z=z,
                )
            )

        result = na.concatenate(segments, axis="wire")
        if self.transformation is not None:
            result = self.transformation(result)
        return result


@dataclasses.dataclass(eq=False, repr=False)
class EllipticalAperture(
    AbstractAperture,
):
    """
    An elliptical aperture or obscuration

    Examples
    --------

    Plot a single elliptical aperture

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        aperture = optika.apertures.EllipticalAperture(
            na.Cartesian2dVectorArray(100, 50) * u.mm,
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")
    """

    radius: na.AbstractCartesian2dVectorArray = 0 * u.mm
    """The semi major/minor axes of the elliptical aperture."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        radius = self.radius
        active = self.active
        inverted = self.inverted
        if self.transformation is not None:
            position = self.transformation.inverse(position)

        shape = na.shape_broadcasted(radius, active, inverted, position)

        radius = na.broadcast_to(radius, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask = np.square(position.x / radius.x) + np.square(position.y / radius.y) <= 1

        mask[inverted] = ~mask[inverted]
        mask[~active] = True

        return mask

    def _bound_center_half(
        self,
    ) -> tuple[na.Cartesian3dVectorArray, na.Cartesian3dVectorArray]:
        """
        The center and per-component half-extent of the axis-aligned bounding
        box, computed analytically so the bound is exact even when
        :attr:`transformation` rotates the ellipse.

        A point on the ellipse boundary is
        :math:`p(t) = c + a\\,\\hat{e}_a \\cos t + b\\,\\hat{e}_b \\sin t`,
        where :math:`c` is the center and :math:`\\hat{e}_a, \\hat{e}_b` are the
        images of the local axes under the transformation.  The extent along
        any world component is then
        :math:`\\sqrt{(a\\,\\hat{e}_a)^2 + (b\\,\\hat{e}_b)^2}`, since
        :math:`\\max_t (A \\cos t + B \\sin t) = \\sqrt{A^2 + B^2}`.
        """
        radius = self.radius
        center = na.Cartesian3dVectorArray() << radius.x.unit
        axis_a = na.Cartesian3dVectorArray(x=radius.x) << radius.x.unit
        axis_b = na.Cartesian3dVectorArray(y=radius.y) << radius.y.unit

        if self.transformation is not None:
            center = self.transformation(center)
            axis_a = self.transformation(axis_a) - center
            axis_b = self.transformation(axis_b) - center

        half = np.sqrt(np.square(axis_a) + np.square(axis_b))
        return center, half

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        center, half = self._bound_center_half()
        return center - half

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        center, half = self._bound_center_half()
        return center + half

    @property
    def vertices(self) -> None:
        return None

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire
        az = na.linspace(
            start=0 * u.deg,
            stop=360 * u.deg,
            axis="wire",
            num=num,
        )
        unit_radius = na.unit(self.radius)
        result = na.Cartesian3dVectorArray(
            x=self.radius.x * np.cos(az),
            y=self.radius.y * np.sin(az),
            z=0 * unit_radius if unit_radius is not None else 0,
        )
        if self.transformation is not None:
            result = self.transformation(result)
        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolygonalAperture(
    AbstractAperture,
):
    """
    An interface describing a generalized polygonal aperture.
    """

    @property
    @abc.abstractmethod
    def vertices(self) -> None | na.AbstractCartesian3dVectorArray:
        """
        The vertices of the polygon in local coordinates.
        """

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        vertices = self.vertices
        active = self.active
        inverted = self.inverted

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        if np.any(active):
            result = na.geometry.point_in_polygon(
                x=position.x,
                y=position.y,
                vertices_x=vertices.x,
                vertices_y=vertices.y,
                axis="vertex",
            )

            if np.any(inverted):
                if np.all(inverted):
                    result = ~result
                else:
                    shape_inverted = na.shape_broadcasted(result, inverted)
                    if shape_inverted != result.shape:
                        result = na.broadcast_to(result, shape_inverted).copy()
                    result[inverted] = ~result[inverted]

            if not np.all(active):
                shape_active = na.shape_broadcasted(result, active)
                if shape_active != result.shape:
                    result = na.broadcast_to(result, shape_active).copy()
                result[~active] = True

        else:
            result = na.ScalarArray(True)

        return result

    @property
    def bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        vertices = self.vertices
        if self.transformation is not None:
            vertices = self.transformation(vertices)
        return vertices.min(axis="vertex")

    @property
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        vertices = self.vertices
        if self.transformation is not None:
            vertices = self.transformation(vertices)
        return vertices.max(axis="vertex")

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire
        vertices = self.vertices.broadcasted
        num_vertices = vertices.shape["vertex"]
        num_sides = num_vertices
        num_per_side = num / num_sides
        index_right = na.arange(0, num_vertices, axis="vertex") + 1
        index_right = index_right % num_vertices
        index_right = dict(vertex=index_right)
        vertices_left = vertices
        vertices_right = vertices[index_right]
        wire = []
        num_cumulative = 0
        for v in range(num_vertices):
            num_v = int((v + 1) * num_per_side - num_cumulative)
            num_cumulative += num_v

            if num_cumulative == num:
                endpoint = True
            else:
                endpoint = False

            t = na.linspace(
                start=0,
                stop=1,
                axis="wire",
                num=num_v,
                endpoint=endpoint,
            )
            vertex_left = vertices_left[dict(vertex=v)]
            vertex_right = vertices_right[dict(vertex=v)]
            diff = vertex_right - vertex_left
            wire_v = vertex_left + diff * t
            wire.append(wire_v)

        wire = na.concatenate(wire, axis="wire")

        if self.transformation is not None:
            wire = self.transformation(wire)

        return wire


@dataclasses.dataclass(eq=False, repr=False)
class PolygonalAperture(
    AbstractPolygonalAperture,
):
    """A polygonal aperture or obstruction."""

    vertices: na.Cartesian3dVectorArray = 0 * u.mm
    """The vertices of the polygon in local coordinates."""

    @property
    def shape(self) -> dict[str, int]:
        shape = optika.shape(self.vertices)
        shape.pop("vertex")
        return na.broadcast_shapes(
            shape,
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )


@dataclasses.dataclass(eq=False, repr=False)
class RectangularAperture(
    AbstractPolygonalAperture,
):
    """
    A rectangular aperture or obscuration

    Examples
    --------

    Create a square aperture by setting :attr:`half_width` to a scalar value.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        half_width = 50 * u.mm

        aperture = optika.apertures.RectangularAperture(half_width)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")

    |

    Create a rectangular aperture by setting :attr:`half_width` to an
    instance of :class:`named_arrays.AbstractCartesian2dVectorArray`.

    .. jupyter-execute::

        half_width = na.Cartesian2dVectorArray(100, 50) * u.mm

        aperture = optika.apertures.RectangularAperture(half_width)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(color="black")

    |

    Create a grid of rectangular apertures using the :attr:`transformation`
    parameter and the :class:`optika.transforms.Translation` transformation.

    .. jupyter-execute::

        pitch = 2 * half_width + 10 * u.mm

        displacement = na.Cartesian3dVectorArray(
            x=pitch.x * na.arange(0, 3, axis="aperture_x"),
            y=pitch.y * na.arange(0, 2, axis="aperture_y"),
        )

        aperture = optika.apertures.RectangularAperture(
            half_width=half_width,
            transformation=na.transformations.Translation(displacement),
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")
    """

    half_width: u.Quantity | na.AbstractScalar | na.Cartesian2dVectorArray = 0 * u.mm
    """The distance from the origin to a perpendicular edge."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.half_width),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        half_width = na.asanyarray(
            self.half_width,
            like=na.Cartesian2dVectorArray(),
        )
        active = self.active
        inverted = self.inverted
        if self.transformation is not None:
            position = self.transformation.inverse(position)
        position = position.xy

        shape = na.shape_broadcasted(half_width, active, inverted, position)

        half_width = na.broadcast_to(half_width, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask = (-half_width <= position) & (position <= half_width)
        mask = mask.x & mask.y

        mask[inverted] = ~mask[inverted]
        mask[~active] = True

        return mask

    @property
    def vertices(self):
        half_width = na.asanyarray(self.half_width, like=na.Cartesian2dVectorArray())
        r = np.sqrt(2)
        az = na.linspace(0, 360, axis="vertex", num=4, endpoint=False) * u.deg
        az = az + 45 * u.deg
        result = na.Cartesian3dVectorArray(
            x=r * np.cos(az).value,
            y=r * np.sin(az).value,
            z=0,
        )
        result.x = result.x * half_width.x
        result.y = result.y * half_width.y
        unit = na.unit(half_width.x)
        if unit is not None:
            result.z = result.z * unit
        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRegularPolygonalAperture(
    AbstractPolygonalAperture,
):
    """An interface describing a regular polygonal aperture."""

    @property
    @abc.abstractmethod
    def radius(self) -> na.ScalarLike:
        """
        The radial distance from the origin to each vertex.
        """

    @property
    @abc.abstractmethod
    def num_vertices(self) -> int:
        """
        Number of vertices in this regular polygon.
        """

    @property
    def vertices(self) -> na.AbstractCartesian3dVectorArray:
        radius = self.radius
        unit = na.unit(radius)
        angle = na.linspace(
            start=0 * u.deg,
            stop=360 * u.deg,
            axis="vertex",
            num=self.num_vertices,
            endpoint=False,
        )
        result = na.Cartesian3dVectorArray(
            x=radius * np.cos(angle).value,
            y=radius * np.sin(angle).value,
            z=0,
        )
        if unit is not None:
            result.z = result.z * unit
        return result


@dataclasses.dataclass(eq=False, repr=False)
class RegularPolygonalAperture(
    AbstractRegularPolygonalAperture,
):
    """A regular polygonal aperture or obstruction."""

    radius: float | u.Quantity | na.AbstractScalar = 0 * u.mm
    """The radial distance from the origin to each vertex."""

    num_vertices: int = 0
    """The number of vertices in this polygon."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractOctagonalAperture(
    AbstractRegularPolygonalAperture,
):
    """An interface describing a octagon aperture."""

    @property
    def num_vertices(self) -> int:
        return 8


@dataclasses.dataclass(eq=False, repr=False)
class OctagonalAperture(
    AbstractOctagonalAperture,
):
    """A octagonal aperture or obstruction."""

    radius: float | u.Quantity | na.AbstractScalar = 0 * u.mm
    """The radial distance from the origin to each vertex."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractIsoscelesTrapezoidalAperture(
    AbstractPolygonalAperture,
):
    """A generalized isosceles-trapezoidal aperture."""

    @property
    @abc.abstractmethod
    def x_left(self) -> na.ScalarLike:
        """
        The :math:`x` coordinate of the left base of the trapezoid
        in local coordinates.
        """

    @property
    @abc.abstractmethod
    def x_right(self) -> na.ScalarLike:
        """
        The :math:`x` coordinate of the right base of the trapezoid
        in local coordinates.
        """

    @property
    @abc.abstractmethod
    def angle(self) -> na.ScalarLike:
        """The angle between the two legs of the trapezoid."""

    @property
    def vertices(self) -> na.Cartesian3dVectorArray:
        x_left = self.x_left
        x_right = self.x_right
        angle = self.angle

        m = np.tan(angle / 2)
        left = na.Cartesian3dVectorArray(
            x=x_left,
            y=m * x_left,
            z=0 * x_left,
        )
        right = na.Cartesian3dVectorArray(
            x=x_right,
            y=m * x_right,
            z=0 * x_right,
        )

        upper = na.stack([left, right], axis="vertex")

        lower = upper[dict(vertex=slice(None, None, -1))]
        lower = lower * na.Cartesian3dVectorArray(1, -1, 1)

        result = na.concatenate([upper, lower], axis="vertex")

        return result


@dataclasses.dataclass(eq=False, repr=False)
class IsoscelesTrapezoidalAperture(
    AbstractIsoscelesTrapezoidalAperture,
):
    """
    An isosceles-trapezoidal aperture or obstruction.

    This aperture is useful if you want to break a circular aperture up
    into different sectors.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        num_sectors = 8

        roll = na.linspace(0, 360, axis="roll", num=num_sectors, endpoint=False) * u.deg

        aperture = optika.apertures.IsoscelesTrapezoidalAperture(
            x_left=10 * u.mm,
            x_right=40 * u.mm,
            angle=(360 * u.deg) / num_sectors,
            transformation=na.transformations.TransformationList([
                na.transformations.Cartesian3dTranslation(x=5 * u.mm),
                na.transformations.Cartesian3dRotationZ(roll),
            ])
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(components=("x", "y"), color="black")
    """

    x_left: na.ScalarLike = 0 * u.mm
    """The :math:`x` coordinate of the left base of the trapezoid."""

    x_right: na.ScalarLike = 0 * u.mm
    """The :math:`x` coordinate of the right base of the trapezoid."""

    angle: na.ScalarLike = 0 * u.deg
    """The angle between the two legs of the trapezoid."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.x_left),
            optika.shape(self.x_right),
            optika.shape(self.angle),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )
