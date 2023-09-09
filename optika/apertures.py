import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika.mixins
import optika.plotting

__all__ = [
    "AbstractAperture",
    "CircularAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractAperture(
    optika.mixins.Printable,
    optika.plotting.Plottable,
    optika.transforms.Transformable,
):
    @property
    @abc.abstractmethod
    def samples_per_side(self):
        """
        number of samples per side of the polygon representing the aperture
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
        mask = self(rays.position)
        rays = rays.copy_shallow()
        rays.intensity = rays.intensity * mask
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

    @property
    @abc.abstractmethod
    def wire(self) -> na.AbstractCartesian3dVectorArray:
        """
        The sequence of points representing this aperture
        """

    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transform: None | optika.transforms.AbstractTransform = None,
        component_map: dict[str, str] = None,
        sag: None | optika.sags.AbstractSag = None,
        **kwargs,
    ) -> None | na.ScalarArray[npt.NDArray[None | matplotlib.lines.Line2D]]:
        if ax is None:
            ax = plt.gca()
        ax = na.as_named_array(ax)

        if component_map is None:
            component_map = dict()
        component_map = dict(x="x", y="y", z="z") | component_map

        wire = self.wire.explicit

        if not wire.length.unit_normalized.is_equivalent(u.mm):
            return None

        if sag is not None:
            wire.z = sag(wire)

        if transform is not None:
            wire = transform(wire)

        kwargs_plot = self.kwargs_plot
        if kwargs_plot is None:
            kwargs_plot = dict()

        kwargs = kwargs_plot | kwargs

        wire = wire.components
        wire = na.Cartesian3dVectorArray(
            x=wire[component_map["x"]],
            y=wire[component_map["y"]],
            z=wire[component_map["z"]],
        )

        return na.plt.plot(
            wire,
            ax=ax,
            axis="wire",
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
            aperture.plot(color="black")

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
            na.Cartesian3dVectorArray() * u.mm,
            displacement
        ], axis="segment")

        aperture = optika.apertures.CircularAperture(
            radius=radius,
            transform=optika.transforms.Translation(displacement),
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(color="black")
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """the radius of the aperture"""

    samples_per_side: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transform: None | optika.transforms.AbstractTransform = None
    kwargs_plot: None | dict = None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        active = self.active
        inverted = self.inverted
        if self.transform is not None:
            position = self.transform.inverse(position)

        shape = na.shape_broadcasted(radius, active, inverted, position)

        radius = na.broadcast_to(radius, shape)
        active = na.broadcast_to(active, shape)
        inverted = na.broadcast_to(inverted, shape)
        position = na.broadcast_to(position, shape)

        mask = position.length <= radius

        mask[inverted] = ~mask[inverted]
        mask[~active] = True

        return mask

    @property
    def bound_lower(self) -> na.Cartesian3dVectorArray:
        result = na.Cartesian3dVectorArray(
            x=-self.radius,
            y=-self.radius,
            z=0 * self.radius.unit,
        )
        if self.transform is not None:
            result = self.transform(result, use_matrix=False)
        return result

    @property
    def bound_upper(self) -> na.Cartesian3dVectorArray:
        result = na.Cartesian3dVectorArray(
            x=self.radius,
            y=self.radius,
            z=0 * self.radius.unit,
        )
        if self.transform is not None:
            result = self.transform(result, use_matrix=False)
        return result

    @property
    def vertices(self) -> None:
        return None

    @property
    def wire(self) -> na.Cartesian3dVectorArray:
        az = na.linspace(
            start=0 * u.deg,
            stop=360 * u.deg,
            axis="wire",
            num=self.samples_per_side,
        )
        result = na.Cartesian3dVectorArray(
            x=self.radius * np.cos(az),
            y=self.radius * np.sin(az),
            z=0 * self.radius.unit,
        )
        if self.transform is not None:
            result = self.transform(result)
        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolygonalAperture(
    AbstractAperture,
):
    """
    Base class for any type of polygonal aperture
    """

    @property
    def bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        return self.vertices.min(axis="vertex")

    @property
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        return self.vertices.max(axis="vertex")

    @property
    def wire(self) -> na.Cartesian3dVectorArray:
        vertices = self.vertices.broadcasted
        num_vertices = vertices.shape["vertex"]
        ind_rolled = dict(vertex=na.arange(0, num_vertices, axis="vertex") - 1)
        vertices_left = vertices[ind_rolled]
        vertices_right = vertices
        diff = vertices_right - vertices_left
        t = na.linspace(
            start=0,
            stop=1,
            axis="wire",
            num=self.samples_per_side,
            endpoint=False,
        )
        wire = vertices_left + diff * t
        wire = wire.combine_axes(axes=("vertex", "wire"), axis_new="wire")
        return wire


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
            aperture.plot(color="black")

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

    Create a grid of rectangular apertures using the :attr:`transform`
    parameter and the :class:`optika.transforms.Translation` transformation.

    .. jupyter-execute::

        pitch = 2 * half_width + 10 * u.mm

        displacement = na.Cartesian3dVectorArray(
            x=pitch.x * na.arange(0, 3, axis="aperture_x"),
            y=pitch.y * na.arange(0, 2, axis="aperture_y"),
        )

        aperture = optika.apertures.RectangularAperture(
            half_width=half_width,
            transform=optika.transforms.Translation(displacement),
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            aperture.plot(color="black")
    """

    half_width: u.Quantity | na.AbstractScalar | na.Cartesian2dVectorArray = 0 * u.mm
    """distance from the origin to a perpendicular edge"""

    samples_per_side: int = 101
    active: bool | na.AbstractScalar = True
    inverted: bool | na.AbstractScalar = False
    transform: None | optika.transforms.AbstractTransform = None
    kwargs_plot: None | dict = None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        bound_lower = self.bound_lower
        bound_upper = self.bound_upper
        active = self.active
        inverted = self.inverted
        if self.transform is not None:
            position = self.transform.inverse(position)

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
            x=r * np.cos(az),
            y=r * np.sin(az),
            z=0,
        )
        result.x = result.x * half_width.x
        result.y = result.y * half_width.y
        result.z = 0 * half_width.length.unit
        if self.transform is not None:
            result = self.transform(result)
        return result
