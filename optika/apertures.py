"""
A collection of apertures that can be applied to optical surfaces to block light.
"""

import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractAperture",
    "CircularAperture",
    "CircularSectorAperture",
    "EllipticalAperture",
    "AbstractPolygonalAperture",
    "PolygonalAperture",
    "AbstractRegularPolygonalAperture",
    "AbstractOctagonalAperture",
    "OctagonalAperture",
    "AbstractIsoscelesTrapezoidalAperture",
    "IsoscelesTrapezoidalAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractAperture(
    optika.mixins.Printable,
    optika.mixins.Plottable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    @property
    @abc.abstractmethod
    def samples_wire(self):
        """
        default number of samples used for :meth:`wire`
        """

    @property
    @abc.abstractmethod
    def active(self):
        """
        flag controlling whether the aperture can clip rays during a raytrace
        """

    @property
    @abc.abstractmethod
    def inverted(self):
        """
        flag controlling whether the interior or the exterior of the aperture
        allows light to pass through.
        If :obj:`True`, the interior of the aperture allows light to pass
        through.
        If :obj:`False`, the exterior of the aperture allows light to pass
        through.
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
            the points to check
        """

    def clip_rays(self, rays: optika.rays.RayVectorArray):
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
        """

    @property
    @abc.abstractmethod
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        """
        The upper-right corner of the aperture's rectangular footprint
        """

    @property
    @abc.abstractmethod
    def vertices(self) -> None | na.AbstractCartesian3dVectorArray:
        """
        The vertices of the polygon representing this aperture
        """

    @abc.abstractmethod
    def wire(self, num: None | int = None) -> na.AbstractCartesian3dVectorArray:
        """
        The sequence of points representing this aperture

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
    """the radius of the aperture"""

    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
        result.x = result.x - self.radius
        result.y = result.y - self.radius
        return result

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result.x = result.x + self.radius
        result.y = result.y + self.radius
        return result

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
    """
    The ending angle of the circular sector.
    Must be between :math:`-2 \pi` and :math:`+2 \pi` radians and 
    counterclockwise from `angle_start`.
    """

    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        return self.wire().min()

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        return self.wire().max()

    @property
    def vertices(self) -> None:
        return None

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire
        az = na.linspace(
            start=self.angle_start,
            stop=self.angle_stop,
            axis="wire",
            num=num - 2,
        )
        unit_radius = na.unit(self.radius)
        result = na.Cartesian3dVectorArray(
            x=self.radius * np.cos(az),
            y=self.radius * np.sin(az),
            z=0 * unit_radius if unit_radius is not None else 0,
        )

        vertex = na.Cartesian3dVectorArray().add_axes("wire")
        vertex = vertex * unit_radius if unit_radius is not None else vertex
        result = np.concatenate(
            [
                vertex,
                result,
                vertex,
            ],
            axis="wire",
        )
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
    """the semi major/minor axes of the elliptical aperture."""

    samples_wire: int = 101
    """default number of samples used for :meth:`wire`"""

    active: bool | na.AbstractScalar = True
    """flag controlling whether the aperture can clip rays during a raytrace"""

    inverted: bool | na.AbstractScalar = False
    """
    flag controlling whether the interior or the exterior of the aperture
    allows light to pass through.
    If :obj:`True`, the interior of the aperture allows light to pass
    through.
    If :obj:`False`, the exterior of the aperture allows light to pass
    through.
    """

    transformation: None | na.transformations.AbstractTransformation = None
    """
    the coordinate transformation between the global coordinate system
    and this object's local coordinate system
    """

    kwargs_plot: None | dict = None
    """
    Extra keyword arguments that will be used in the call to
    :func:`named_arrays.plt.plot` within the :meth:`plot` method.
    """

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

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result.x = result.x - self.radius.x
        result.y = result.y - self.radius.y
        return result

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result.x = result.x + self.radius.x
        result.y = result.y + self.radius.y
        return result

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
    Base class for any type of polygonal aperture
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

        shape = na.shape_broadcasted(
            vertices[dict(vertex=0)],
            active,
            inverted,
            position,
        )

        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        result = False
        for v in range(vertices.shape["vertex"]):
            vert_j = na.broadcast_to(vertices[dict(vertex=v - 1)], shape)
            vert_i = na.broadcast_to(vertices[dict(vertex=v)], shape)
            slope = (vert_j.y - vert_i.y) / (vert_j.x - vert_i.x)
            condition_1 = (vert_i.y > position.y) != (vert_j.y > position.y)
            condition_2 = position.x < ((position.y - vert_i.y) / slope + vert_i.x)
            result = result ^ (condition_1 & condition_2)

        result[inverted] = ~result[inverted]
        result[~active] = True

        return result

    @property
    def bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        return self.vertices.min(axis="vertex")

    @property
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        return self.vertices.max(axis="vertex")

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
    vertices: na.Cartesian3dVectorArray = 0 * u.mm
    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
    """distance from the origin to a perpendicular edge"""

    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
        bound_lower = self.bound_lower
        bound_upper = self.bound_upper
        active = self.active
        inverted = self.inverted
        if self.transformation is not None:
            position = self.transformation.inverse(position)

        shape = na.shape_broadcasted(
            bound_lower, bound_upper, active, inverted, position
        )

        bound_lower = na.broadcast_to(bound_lower, shape)
        bound_upper = na.broadcast_to(bound_upper, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask = (bound_lower <= position) & (position <= bound_upper)
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
    @property
    @abc.abstractmethod
    def radius(self) -> na.ScalarLike:
        """
        the radial distance from the origin to each vertex
        """

    @property
    @abc.abstractmethod
    def num_vertices(self) -> int:
        """
        Number of vertices in this polygon
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
    radius: float | u.Quantity | na.AbstractScalar = 0 * u.mm
    num_vertices: int = 0
    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
    @property
    def num_vertices(self) -> int:
        return 8


@dataclasses.dataclass(eq=False, repr=False)
class OctagonalAperture(
    AbstractOctagonalAperture,
):
    radius: float | u.Quantity | na.AbstractScalar = 0 * u.mm
    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
    @property
    @abc.abstractmethod
    def x_left(self) -> na.ScalarLike:
        """:math:`x` coordinate of the left base of the trapezoid"""

    @property
    @abc.abstractmethod
    def x_right(self) -> na.ScalarLike:
        """:math:`x` coordinate of the right base of the trapezoid"""

    @property
    @abc.abstractmethod
    def angle(self) -> na.ScalarLike:
        """angle between the two legs of the trapezoid"""

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
    x_right: na.ScalarLike = 0 * u.mm
    angle: na.ScalarLike = 0 * u.deg
    samples_wire: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

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
