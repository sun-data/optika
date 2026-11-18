import abc
import dataclasses
import functools
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import mpl_toolkits.mplot3d.art3d
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


def _wire_parameters(num: int, num_sides: int) -> list[na.AbstractScalar]:
    """
    Share the samples of a closed outline out among its sides.

    The samples are shared out in integers, so that they always add up to
    `num` and the outline always ends where it began.  With more samples
    than sides, every side starts with its first corner and the last side
    ends at the first corner again, so every corner is sampled.  With as
    many samples as sides, the last side has room only for the sample which
    closes the outline, so its own corner is left out.  With fewer, some
    sides get none, so all but the closing sample are shared out with the
    first side served first, and the closing sample is added to the last.

    Parameters
    ----------
    num
        The total number of samples.
    num_sides
        The number of sides of the outline.

    Returns
    -------
        For each side, its samples along the ``wire`` axis, as the fraction
        of the way from the start of the side to its end.
    """
    few = num < num_sides
    if few:
        bounds = [-(-s * (num - 1) // num_sides) for s in range(num_sides + 1)]
    else:
        bounds = [s * num // num_sides for s in range(num_sides + 1)]
    result = []
    for s in range(num_sides):
        num_s = bounds[s + 1] - bounds[s]
        last = s == num_sides - 1
        if last and few:
            num_s += 1
        t = na.linspace(
            # a single sample left for the last side is the point which
            # closes the outline, not the corner that side starts from
            start=1 if last and num_s == 1 else 0,
            stop=1,
            axis="wire",
            num=num_s,
            endpoint=last,
        )
        result.append(t)
    return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractAperture(
    optika.mixins.DxfWritable,
    optika.mixins.Printable,
    optika.mixins.Replaceable,
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

    If :obj:`False`, the interior of the aperture allows light to pass through.
    If :obj:`True`, the exterior of the aperture allows light to pass through.
    """

    transformation: None | na.transformations.AbstractTransformation = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The transformation between the local surface coordinates and the aperture."""

    axis_compound: None | str = dataclasses.field(default=None, kw_only=True)
    """
    The logical axis along which this aperture is made of several pieces
    acting together as one.

    An aperture whose parameters vary along an axis is an array of apertures,
    each of which clips only the rays it is broadcast against, as each slit
    of a slit array does.  Naming that axis here makes the pieces along it
    act together instead: a ray passes if it is inside at least one active,
    non-inverted piece (or there are none) and outside every active, inverted
    one.  So an array of openings is their union, as the segments of a
    segmented mirror are, an array of obscurations has to be cleared all at
    once, and the two can be mixed through an :attr:`inverted` which varies
    along this axis, for an opening with an obscuration in it.

    This axis is left out of :attr:`shape` and reduced over by
    :attr:`bound_lower` and :attr:`bound_upper`, so it never reaches the rays
    of a surface which uses this aperture, but :meth:`wire` keeps it, so that
    every piece is drawn.  It has to be an axis the parameters of this
    aperture vary along, and not one the points it is called on vary along,
    since each point would then be tested against only the piece with the
    same index.  An aperture with a compound axis cannot yet be the field
    stop or pupil stop of a system, since a stop is solved from a single
    outline.

    If :obj:`None` (the default), every axis of this aperture is an array of
    separate apertures.
    """

    kwargs_plot: None | dict = dataclasses.field(default=None, kw_only=True)
    """
    Extra keyword arguments that will be used in the call to
    :func:`named_arrays.plt.plot` within the :meth:`plot` method.
    """

    @property
    def shape(self) -> dict[str, int]:
        shape = self._shape_parameters
        axis = self.axis_compound
        if axis is not None:
            self._check_axis_compound(shape)
            del shape[axis]
        return shape

    @property
    @abc.abstractmethod
    def _shape_parameters(self) -> dict[str, int]:
        """
        The broadcasted shape of the parameters of this aperture, which
        includes :attr:`axis_compound`, unlike :attr:`shape`.
        """

    def _check_axis_compound(self, shape: dict[str, int]) -> None:
        """
        Raise if :attr:`axis_compound` is not one of the axes in `shape`.

        Parameters
        ----------
        shape
            The shape of the parameters of this aperture, as
            :attr:`_shape_parameters` gives it.
        """
        axis = self.axis_compound
        if axis not in shape:
            raise ValueError(
                f"the compound axis {axis!r} is not one of the axes the "
                f"parameters of this aperture vary along, {tuple(shape)}"
            )

    def _reduce_compound(
        self,
        a: na.AbstractArray,
        reduction: str,
    ) -> na.AbstractArray:
        """
        Reduce `a` over :attr:`axis_compound` with the method named
        `reduction`, if `a` varies along it.

        Parameters
        ----------
        a
            The array to reduce.
        reduction
            The name of the reducing method of `a`, such as ``"any"`` or
            ``"min"``.
        """
        axis = self.axis_compound
        if axis in na.shape(a):
            a = getattr(a, reduction)(axis=axis)
        return a

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Check if a given point is inside the aperture.

        A point passes if it is inside the aperture, or outside it if the
        aperture is :attr:`inverted`, or anywhere if the aperture is not
        :attr:`active`.  The pieces along :attr:`axis_compound`, if there is
        one, act together as described there.

        Parameters
        ----------
        position
            Points in surface coordinates.
            They may not vary along :attr:`axis_compound`.
        """
        axis = self.axis_compound
        if axis is not None:
            self._check_axis_compound(self._shape_parameters)
            if axis in na.shape(position):
                raise ValueError(
                    f"the points vary along the compound axis {axis!r}, so "
                    f"each would be tested against only the piece with the "
                    f"same index"
                )

        active = na.as_named_array(self.active)
        inverted = na.as_named_array(self.inverted)

        # an inactive aperture passes every point, so they are not tested
        if np.any(active):
            if self.transformation is not None:
                position = self.transformation.inverse(position)
            inside = self._inside(position)
        else:
            inside = na.ScalarArray(True)

        # an opening passes the points inside it, an obscuration those outside
        # it, and an inactive piece every point, and the plain defaults of the
        # flags leave the points as they are, so they skip a pass over them
        if axis is None:
            result = inside
            if self.inverted is not False:
                result = result ^ inverted
            if self.active is not True:
                result = result | ~active
            return result

        opening = active & ~inverted
        obscuration = active & inverted

        through = self._reduce_compound(inside & opening, "any")
        through = through | ~self._reduce_compound(opening, "any")
        clear = self._reduce_compound(~inside | ~obscuration, "all")

        return through & clear

    @abc.abstractmethod
    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Check if a given point is inside the outline of the aperture,
        regardless of :attr:`active` and :attr:`inverted`.

        Parameters
        ----------
        position
            Points in the local coordinates of the aperture, before
            :attr:`transformation`.
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
        else:  # pragma: nocover
            raise ValueError(f"aperture with unit {unit} is not supported")
        rays = rays.copy_shallow()
        rays.unvignetted = rays.unvignetted & mask
        return rays

    @property
    def bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        """
        The lower-left corner of the aperture's rectangular footprint
        in surface coordinates.
        """
        return self._reduce_compound(self._bound_lower, "min")

    @property
    @abc.abstractmethod
    def _bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        """
        :attr:`bound_lower` of each piece along :attr:`axis_compound`.
        """

    @property
    def bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        """
        The upper-right corner of the aperture's rectangular footprint
        in surface coordinates.
        """
        return self._reduce_compound(self._bound_upper, "max")

    @property
    @abc.abstractmethod
    def _bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        """
        :attr:`bound_upper` of each piece along :attr:`axis_compound`.
        """

    @abc.abstractmethod
    def wire(self, num: None | int = None) -> na.AbstractCartesian3dVectorArray:
        """
        A sequence of points representing this aperture in surface coordinates.

        Each piece along :attr:`axis_compound` has a wire of its own.

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
        unit: None | u.UnitBase = None,
        **kwargs,
    ) -> (
        None
        | na.ScalarArray[
            npt.NDArray[
                None
                | matplotlib.lines.Line2D
                | mpl_toolkits.mplot3d.art3d.Poly3DCollection
            ]
        ]
    ):
        """
        Draw this aperture onto the given axes.

        On a 2D axes the aperture is drawn as a closed line. On a 3D axes it is
        drawn as a filled polygon instead, so that it hides whatever lies behind
        it, and the colour asked for becomes the edge of that polygon while the
        face is left blank.

        Parameters
        ----------
        ax
            The matplotlib axes to draw onto.
            If :obj:`None`, the current axes is used.
        transformation
            Any extra transformation to apply before drawing.
        components
            Which components of the aperture to draw, in the order of the axes
            of the plot.
        sag
            An optional sag profile to lift the aperture onto.
        unit
            The unit to express every drawn length in.
        kwargs
            Additional keyword arguments passed along to
            :func:`named_arrays.plt.plot`, or to :func:`named_arrays.plt.fill`
            on a 3D axes, where ``color`` becomes the edge colour and
            ``facecolor`` the face.
        """
        if ax is None:
            ax = plt.gca()
        ax = na.as_named_array(ax)

        wire = self.wire().explicit

        if not wire.length.unit_normalized.is_equivalent(u.mm):
            return None

        if sag is not None:
            wire.z = sag(wire)

        if unit is not None:
            wire = wire.to(unit)

        kwargs_plot = self.kwargs_plot
        if kwargs_plot is None:
            kwargs_plot = dict()

        kwargs = kwargs_plot | kwargs

        # On a 3D axes the aperture is drawn as a filled polygon rather than a
        # closed line. The fill hides whatever is behind the surface, which an
        # outline cannot, and carrying the edge on the same artist means the
        # two can never be sorted apart: matplotlib orders a 3D axes by one
        # depth per artist, so a separate outline lying on its own face is
        # placed in front of or behind it at random.
        if optika.plot.is_3d(ax):
            return na.plt.fill(
                wire,
                ax=ax,
                axis="wire",
                transformation=transformation,
                components=components,
                **optika.plot.kwargs_filled(kwargs),
            )

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

        wire = wire.broadcasted

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
    The segments act together as one mirror, so ``"segment"`` is a compound
    axis, and a point passes if it is inside any of them.

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
            axis_compound="segment",
        )

        position = na.Cartesian3dVectorLinearSpace(
            start=aperture.bound_lower,
            stop=aperture.bound_upper,
            axis=na.Cartesian3dVectorArray("x", "y", "z"),
            num=na.Cartesian3dVectorArray(201, 201, 1),
        )

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            na.plt.pcolormesh(
                position.x,
                position.y,
                C=aperture(position).astype(float),
                cmap="gray",
            )
            aperture.plot(components=("x", "y"), color="tab:red")
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The radius of the aperture."""

    @property
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        return position.xy.length <= self.radius

    @property
    def _bound_lower(self) -> na.Cartesian3dVectorArray:
        unit = na.unit(self.radius)
        result = na.Cartesian3dVectorArray()
        if unit is not None:
            result = result * unit
        if self.transformation is not None:
            result = self.transformation(result)
        result = result - self.radius
        return result

    @property
    def _bound_upper(self) -> na.Cartesian3dVectorArray:
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
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.angle_start),
            optika.shape(self.angle_stop),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        angle_start = self.angle_start
        angle_stop = self.angle_stop

        mask_radius = position.xy.length <= self.radius

        angle = np.arctan2(position.y, position.x)
        angle_positive = angle % (+2 * np.pi * u.rad)
        angle_negative = angle % (-2 * np.pi * u.rad)
        mask_positive = (angle_start < angle_positive) & (angle_positive < angle_stop)
        mask_negative = (angle_start < angle_negative) & (angle_negative < angle_stop)
        mask_angle = mask_positive | mask_negative

        return mask_radius & mask_angle

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
    def _bound_lower(self) -> na.Cartesian3dVectorArray:
        lower, upper = self._bound_extrema()
        return lower

    @property
    def _bound_upper(self) -> na.Cartesian3dVectorArray:
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
        segments = []
        for s, t in enumerate(_wire_parameters(num, num_sides=3)):
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
class AnnularAperture(
    AbstractAperture,
):
    r"""
    An annular (ring-shaped) aperture, optionally restricted to a sector.

    Defined by an inner and outer radius.  By default it spans the full ring,
    but the optional ``angle_start`` / ``angle_stop`` arguments restrict it to an
    annular sector, analogous to :class:`CircularSectorAperture`.
    Setting ``radius_inner=0`` recovers a circular sector.

    Examples
    --------

    Plot an annular aperture and an annular sector

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define a full annulus and a 90-degree annular sector
        aperture = optika.apertures.AnnularAperture(
            radius_inner=30 * u.mm,
            radius_outer=50 * u.mm,
            angle_start=na.ScalarArray([0, 0] * u.deg, axes="aperture"),
            angle_stop=na.ScalarArray([360, 90] * u.deg, axes="aperture"),
        )

        # Define points to sample the aperture with
        points = na.Cartesian3dVectorLinearSpace(
            start=aperture.bound_lower,
            stop=aperture.bound_upper,
            axis=na.Cartesian3dVectorArray("x", "y", "z"),
            num=na.Cartesian3dVectorArray(21, 21, 1),
        )

        # Compute which points are inside the aperture
        where = aperture(points)

        # Plot the apertures
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(ncols=2, figsize=(8, 4), constrained_layout=True)
            ax = na.ScalarArray(ax, axes="aperture")
            for a in [ax[dict(aperture=0)], ax[dict(aperture=1)]]:
                a.ndarray.set_aspect("equal")
            aperture.plot(ax=ax, components=("x", "y"), color="black")
            na.plt.scatter(points.x, points.y, ax=ax, c=where.astype(float))
    """

    radius_inner: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The inner radius of the annulus."""

    radius_outer: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The outer radius of the annulus."""

    angle_start: u.Quantity | na.AbstractScalar = 0 * u.deg
    r"""
    The starting angle of the annular sector.
    Must be between :math:`-2 \pi` and :math:`+2 \pi` radians.
    """

    angle_stop: u.Quantity | na.AbstractScalar = 360 * u.deg
    r"""
    The ending angle of the annular sector,
    counterclockwise from :attr:`angle_start`.
    The default spans the full ring.
    """

    @property
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius_inner),
            optika.shape(self.radius_outer),
            optika.shape(self.angle_start),
            optika.shape(self.angle_stop),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        angle_start = self.angle_start
        angle_stop = self.angle_stop

        radius = position.xy.length
        mask_radius = (self.radius_inner <= radius) & (radius <= self.radius_outer)

        angle = np.arctan2(position.y, position.x)
        angle_relative = (angle - angle_start) % (360 * u.deg)
        mask_angle = angle_relative <= (angle_stop - angle_start)

        return mask_radius & mask_angle

    def _bound_extrema(
        self,
    ) -> tuple[na.Cartesian3dVectorArray, na.Cartesian3dVectorArray]:
        """
        Compute the axis-aligned bounding box of this aperture analytically.

        The extremum of the annulus along a given world axis is attained
        either at one of the four corners, where an arc meets a radial arm,
        or at an interior point of one of the two arcs where the boundary is
        tangent to the world axis, if that point lies within the angular
        range of the annulus.  Taking the extrema of a sampled wire instead
        would fall short of the true extent of the arcs by an amount which
        depends on how densely they happen to be sampled.
        """
        angle_start = self.angle_start
        angle_stop = self.angle_stop

        zero = 0 * self.radius_outer
        origin = na.Cartesian3dVectorArray(x=zero, y=zero, z=zero)

        # A complete annulus has `angle_stop - angle_start` equal to a full
        # turn, which is zero modulo a turn, so it has to be recognized
        # separately: otherwise every interior extremum is rejected as lying
        # outside a zero-width angular range and the bounding box collapses
        # onto the two coincident corners.
        span = angle_stop - angle_start
        span_is_full = np.abs(span) >= 360 * u.deg
        span = span % (360 * u.deg)

        candidates = {c: [] for c in ("x", "y", "z")}

        for radius in (self.radius_inner, self.radius_outer):

            axis_a = na.Cartesian3dVectorArray(x=radius, y=zero, z=zero)
            axis_b = na.Cartesian3dVectorArray(x=zero, y=radius, z=zero)
            center = origin
            if self.transformation is not None:
                center = self.transformation(origin)
                axis_a = self.transformation(axis_a) - center
                axis_b = self.transformation(axis_b) - center

            for c in candidates:
                center_c = getattr(center, c)
                coeff_a = getattr(axis_a, c)
                coeff_b = getattr(axis_b, c)

                def point(angle, _a=coeff_a, _b=coeff_b, _c=center_c):
                    return _c + _a * np.cos(angle) + _b * np.sin(angle)

                # the corners, where this arc meets the two radial arms
                point_start = point(angle_start)
                candidates[c] += [point_start, point(angle_stop)]

                # interior extrema of this arc along this axis, kept only if
                # they lie within the angular range of the annulus
                # `arctan2` of two plain numbers is a plain number, so give the
                # result an angular unit either way
                angle_critical = np.arctan2(coeff_b, coeff_a) << u.rad
                for angle in (angle_critical, angle_critical + 180 * u.deg):
                    where = ((angle - angle_start) % (360 * u.deg)) <= span
                    where = span_is_full | where
                    candidates[c].append(np.where(where, point(angle), point_start))

        result_lower = na.Cartesian3dVectorArray()
        result_upper = na.Cartesian3dVectorArray()
        for c in candidates:
            setattr(result_lower, c, functools.reduce(np.minimum, candidates[c]))
            setattr(result_upper, c, functools.reduce(np.maximum, candidates[c]))

        return result_lower, result_upper

    @property
    def _bound_lower(self) -> na.Cartesian3dVectorArray:
        lower, upper = self._bound_extrema()
        return lower

    @property
    def _bound_upper(self) -> na.Cartesian3dVectorArray:
        lower, upper = self._bound_extrema()
        return upper

    @property
    def vertices(self) -> None:
        return None

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire

        # split the samples between the outer and inner arcs, reserving one
        # point to close the loop
        num_arc = num - 1
        num_outer = num_arc // 2
        num_inner = num_arc - num_outer

        unit_radius = na.unit(self.radius_outer)
        z = 0 * unit_radius if unit_radius is not None else 0

        azimuth_outer = na.linspace(
            self.angle_start, self.angle_stop, axis="wire", num=num_outer
        )
        azimuth_inner = na.linspace(
            self.angle_stop, self.angle_start, axis="wire", num=num_inner
        )

        outer = na.Cartesian3dVectorArray(
            x=self.radius_outer * np.cos(azimuth_outer),
            y=self.radius_outer * np.sin(azimuth_outer),
            z=z,
        )
        inner = na.Cartesian3dVectorArray(
            x=self.radius_inner * np.cos(azimuth_inner),
            y=self.radius_inner * np.sin(azimuth_inner),
            z=z,
        )

        result = np.concatenate([outer, inner], axis="wire")
        # close the loop back to the first point
        result = np.concatenate(
            [result, result[dict(wire=slice(0, 1))]],
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
    """The semi major/minor axes of the elliptical aperture."""

    @property
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        return np.square(position.x / radius.x) + np.square(position.y / radius.y) <= 1

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
    def _bound_lower(self) -> na.Cartesian3dVectorArray:
        center, half = self._bound_center_half()
        return center - half

    @property
    def _bound_upper(self) -> na.Cartesian3dVectorArray:
        center, half = self._bound_center_half()
        return center + half

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

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        vertices = self.vertices

        return na.geometry.point_in_polygon(
            x=position.x,
            y=position.y,
            vertices_x=vertices.x,
            vertices_y=vertices.y,
            axis="vertex",
        )

    @property
    def _bound_lower(self) -> na.AbstractCartesian3dVectorArray:
        vertices = self.vertices
        if self.transformation is not None:
            vertices = self.transformation(vertices)
        return vertices.min(axis="vertex")

    @property
    def _bound_upper(self) -> na.AbstractCartesian3dVectorArray:
        vertices = self.vertices
        if self.transformation is not None:
            vertices = self.transformation(vertices)
        return vertices.max(axis="vertex")

    def wire(self, num: None | int = None) -> na.Cartesian3dVectorArray:
        if num is None:
            num = self.samples_wire
        vertices = self.vertices.broadcasted
        num_vertices = vertices.shape["vertex"]
        index_right = na.arange(0, num_vertices, axis="vertex") + 1
        index_right = index_right % num_vertices
        index_right = dict(vertex=index_right)
        vertices_left = vertices
        vertices_right = vertices[index_right]
        wire = []
        for v, t in enumerate(_wire_parameters(num, num_sides=num_vertices)):
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
    def _shape_parameters(self) -> dict[str, int]:
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
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.half_width),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )

    def _inside(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        half_width = na.asanyarray(
            self.half_width,
            like=na.Cartesian2dVectorArray(),
        )
        position = position.xy

        mask = (-half_width <= position) & (position <= half_width)

        return mask.x & mask.y

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
    def _shape_parameters(self) -> dict[str, int]:
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
    def _shape_parameters(self) -> dict[str, int]:
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
    def _shape_parameters(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.x_left),
            optika.shape(self.x_right),
            optika.shape(self.angle),
            optika.shape(self.active),
            optika.shape(self.inverted),
            optika.shape(self.transformation),
        )
