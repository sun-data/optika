"""The shape of an optical surface used to direct and focus light."""

from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika

__all__ = [
    "AbstractSag",
    "NoSag",
    "AbstractSphericalSag",
    "SphericalSag",
    "CylindricalSag",
    "AbstractConicSag",
    "ConicSag",
    "ParabolicSag",
    "ToroidalSag",
]

RadiusT = TypeVar(
    "RadiusT",
    bound=float | u.Quantity | na.AbstractScalar,
)
ConicT = TypeVar(
    "ConicT",
    bound=float | u.Quantity | na.AbstractScalar,
)
RadiusOfRotationT = TypeVar(
    "RadiusOfRotationT",
    bound=float | u.Quantity | na.AbstractScalar,
)
FocalLengthT = TypeVar(
    "FocalLengthT",
    bound=float | u.Quantity | na.AbstractScalar,
)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSag(
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
    optika.propagators.AbstractRayPropagator,
):
    """
    Base class for all types of sag profiles.
    """

    transformation: None | na.transformations.AbstractTransformation = (
        dataclasses.field(default=None, kw_only=True)
    )
    """
    The transformation between the surface coordinate system and the sag
    coordinate system.
    """

    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The slope error parameters for this sag profile."""

    parameters_roughness: None | optika.metrology.RoughnessParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The roughness parameters for this sag profile."""

    parameters_microroughness: None | optika.metrology.RoughnessParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The microroughness parameters for this sag profile."""

    @abc.abstractmethod
    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        pass

    @abc.abstractmethod
    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The vector perpendicular to the surface at the given position.

        Parameters
        ----------
        position
            The location on the surface to evaluate the normal vector
        """

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.RayVectorArray:
        """
        A set of new rays with the same direction as the input rays,
        but with the :attr:`optika.rays.RayVectorArray.position` updated to
        their interception point with this sag function.

        Parameters
        ----------
        rays
            input rays that will intercept this sag function
        """

        def line(t: na.AbstractScalar) -> na.Cartesian3dVectorArray:
            return rays.position + rays.direction * t

        def func(t: na.AbstractScalar) -> na.AbstractScalar:
            a = line(t)
            z = self(a)
            return a.z - z

        t_intercept = na.optimize.root_secant(
            function=func,
            guess=0 * u.mm,
            min_step_size=1e-6 * u.mm,
        )

        result = rays.copy_shallow()
        result.position = line(t_intercept)
        return result

    def propagate_rays(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        result = self.intercept(rays)

        displacement = result.position - rays.position

        f = np.exp(-result.attenuation * displacement.length)

        result.intensity = f * result.intensity

        return result


@dataclasses.dataclass(eq=False, repr=False)
class NoSag(
    AbstractSag,
):
    """A flat sag profile."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        result = position.replace(z=0 * u.mm)

        if self.transformation is not None:
            result = self.transformation(result)

        return result.z

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(0, 0, -1)

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        d = -rays.position.z / rays.direction.z

        position = rays.position + rays.direction * d

        result = rays.replace(position=position)

        if self.transformation is not None:
            result = self.transformation(result)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSphericalSag(
    AbstractSag,
    Generic[RadiusT],
):
    """
    Base class for all sag functions that can be approximated as a sphere.
    """

    @property
    @abc.abstractmethod
    def radius(self) -> RadiusT:
        """
        The radius of curvature of this sphere.
        """

    @property
    def curvature(self) -> float | RadiusT:
        """
        The curvature of the spherical surface.
        Equal to the reciprocal of :attr:`radius`.
        """
        return 1 / self.radius


@dataclasses.dataclass(eq=False, repr=False)
class SphericalSag(
    AbstractSphericalSag,
    Generic[RadiusT],
):
    r"""
    A spherical sag function.

    The sag (:math:`z` coordinate) of a spherical surface is calculated using
    the expression

    .. math::

        z(x, y) = \frac{c (x^2 + y^2)}{1 + \sqrt{1 - c^2 (x^2 + y^2)}}

    where :math:`c` is the :attr:`curvature`,
    and :math:`x,y`, are the 2D components of the evaluation point.

    Examples
    --------
    Plot a slice through the sag surface

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        sag = optika.sags.SphericalSag(
            radius=na.linspace(100, 300, axis="radius", num=3) * u.mm,
        )

        position = na.Cartesian3dVectorArray(
            x=na.linspace(-90, 90, axis="x", num=101) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm
        )

        z = sag(position)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            na.plt.plot(position.x, z, axis="x", label=sag.radius)
            plt.legend(title="radius")
    """

    radius: RadiusT = np.inf * u.mm
    """The radius of curvature of this sphere."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Evaluate the sag function at the given position

        Parameters
        ----------
        position
            point where the sag function will be calculated
        """
        c = self.curvature
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, position)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        r2 = np.square(position.x) + np.square(position.y)
        sz = c * r2 / (1 + np.sqrt(1 - np.square(c) * r2))
        return sz

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        c = self.curvature

        unit = 1 / c.unit

        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        c = c.value
        x = position.x.to(unit).value
        y = position.y.to(unit).value

        nx = c * x
        ny = c * y
        nz = na.numexpr.evaluate("-sqrt(1 - nx**2 - ny**2)")

        return na.Cartesian3dVectorArray(nx, ny, nz)

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        r = self.radius

        unit = r.unit

        r = r.value

        o = rays.position.to(unit).value

        u = rays.direction

        p = o.replace(z=o.z - r)

        px = p.x  # noqa: F841
        py = p.y  # noqa: F841
        pz = p.z  # noqa: F841

        ux = u.x  # noqa: F841
        uy = u.y  # noqa: F841
        uz = u.z  # noqa: F841

        position = na.numexpr.evaluate(
            "o + u * ("
            "   -(ux * px + uy * py + uz * pz)"
            "   - sign(r * uz) * sqrt("
            "       (ux * px + uy * py + uz * pz)**2"
            "       - (px**2 + py**2 + pz**2 - r**2)"
            "   )"
            ")"
        )

        result = rays.replace(position=position << unit)

        if self.transformation is not None:
            result = self.transformation(result)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class CylindricalSag(
    AbstractSag,
    Generic[RadiusT],
):
    r"""
    A cylindrical sag function, where the local :math:`y` axis is the axis of
    symmetry for the cylinder.

    The sag (:math:`z` coordinate) of a spherical surface is calculated using
    the expression

    .. math::

        z(x, y) = \frac{c x^2}{1 + \sqrt{1 - c^2 x^2}}

    where :math:`c` is the :attr:`curvature`,
    and :math:`x` is the horizontal component of the evaluation point.

    Examples
    --------
    Plot a slice through the sag surface

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        sag = optika.sags.SphericalSag(
            radius=na.linspace(100, 300, axis="radius", num=3) * u.mm,
        )

        position = na.Cartesian3dVectorArray(
            x=na.linspace(-90, 90, axis="x", num=101) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm
        )

        z = sag(position)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            na.plt.plot(position.x, z, axis="x", label=sag.radius)
            plt.legend(title="radius")
    """

    radius: RadiusT = np.inf * u.mm
    """The radius of this cylinder."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        c = 1 / self.radius
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, position)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        r2 = np.square(position.x)
        sz = c * r2 / (1 + np.sqrt(1 - np.square(c) * r2))
        return sz

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:

        r = self.radius

        unit = r.unit

        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        r = r.value
        x = position.x.to(unit).value

        nx = x / r
        ny = 0
        nz = na.numexpr.evaluate("-sqrt(1 - nx**2)")

        return na.Cartesian3dVectorArray(nx, ny, nz)

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        r = self.radius

        o = rays.position

        n = rays.direction

        b = na.Cartesian3dVectorArray(z=r)

        b = b - o

        a = na.Cartesian3dVectorArray(y=1)

        n_cross_a = n.cross(a)
        n_cross_a_squared = n_cross_a @ n_cross_a

        negative_b = n_cross_a @ b.cross(a)
        b_squared = n_cross_a_squared * np.square(r)
        four_ac = np.square(b @ n_cross_a)
        two_a = n_cross_a_squared

        discriminant = b_squared - four_ac

        sgn = np.sign(r * n.z)

        d = np.where(
            discriminant > 0,
            (negative_b - sgn * np.sqrt(discriminant)) / two_a,
            -o.z / n.z,
        )

        position = o + d * n

        result = rays.replace(position=position)

        if self.transformation is not None:
            result = self.transformation(result)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConicSag(
    AbstractSag,
    Generic[RadiusT, ConicT],
):
    """An interface describing a general conic surface of revolution."""

    @property
    @abc.abstractmethod
    def radius(self) -> RadiusT:
        """The effective radius of this conic section."""

    @property
    @abc.abstractmethod
    def conic(self) -> ConicT:
        """The conic constant of this conic section."""

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        c = 1 / self.radius
        conic = self.conic
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, conic, position)
        c = na.broadcast_to(c, shape)
        conic = na.broadcast_to(conic, shape)
        position = na.broadcast_to(position, shape)

        r2 = np.square(position.x) + np.square(position.y)
        sz = c * r2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * r2))
        return sz

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        c = 1 / self.radius
        conic = self.conic
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, conic, position)
        c = na.broadcast_to(c, shape)
        conic = na.broadcast_to(conic, shape)
        position = na.broadcast_to(position, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * position.x / g, c * position.y / g
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length


@dataclasses.dataclass(eq=False, repr=False)
class ConicSag(
    AbstractConicSag[RadiusT, ConicT],
):
    r"""
    Surface of revolution of a conic section

    The sag (:math:`z` coordinate) of a conic sag function is calculated using
    the expression

    .. math::

        z(x, y) = \frac{c (x^2 + y^2)}{1 + \sqrt{1 - c^2 (1 + k) (x^2 + y^2)}}

    where :math:`c` is the :attr:`curvature`,
    :math:`x,y`, are the 2D components of the evaluation point.
    and :math:`k` is the :attr:`conic` constant. See the table below for the
    meaning of the conic constant.

    ================== ==================
    conic constant     conic section type
    ================== ==================
    :math:`k < -1`     hyperbola
    :math:`k = -1`     parabola
    :math:`-1 < k < 0` ellipse
    :math:`k = 0`      sphere
    :math:`k > 0`      oblate ellipsoid
    ================== ==================

    Examples
    --------
    Plot a slice through the sag surface

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        sag = optika.sags.ConicSag(
            radius=100 * u.mm,
            conic=na.ScalarArray(
                ndarray=[-1.5, -1, -0.5, 0, 0.5] * u.dimensionless_unscaled,
                axes="conic",
            )
        )

        position = na.Cartesian3dVectorArray(
            x=na.linspace(-90, 90, axis="x", num=101) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm
        )

        z = sag(position)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.gca().set_aspect("equal")
            na.plt.plot(position.x, z, axis="x", label=sag.conic)
            plt.legend(title="conic constant")
    """

    radius: RadiusT = np.inf * u.mm
    """The effective radius of this conic section."""

    conic: ConicT = 0 * u.dimensionless_unscaled
    """The conic constant of this conic section."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.conic),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )


@dataclasses.dataclass(eq=False, repr=False)
class ParabolicSag(
    AbstractConicSag[RadiusT, int],
):
    """A parabolic sag profile."""

    focal_length: FocalLengthT = np.inf * u.mm
    """The focal length of this parabola."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.focal_length),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    @property
    def radius(self) -> RadiusT:
        return 2 * self.focal_length

    @property
    def conic(self) -> int:
        return -1

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:

        r = self.radius

        unit = r.unit

        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        r = r.value
        position = position.to(unit).value

        position.z = -r

        _x = position.x
        _y = position.y

        return na.numexpr.evaluate(
            "position / sqrt((_x / r)**2 + (_y / r)**2 + 1) / r",
        )

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.RayVectorArray:
        r"""
        Compute the intercept of a ray with this parabolic surface.

        Parameters
        ----------
        rays
            The rays to find the intercept for.

        Notes
        -----

        The equation for a paraboloid is

        .. math::

            z(x, y) = \frac{x^2 + y^2}{4 f},

        where :math:`x`, :math:`y`, and :math:`z` are points on the paraboloid
        and :math:`f` is the focal length of the paraboloid.
        The equation for a line is

        .. math::

            \mathbf{x} = \mathbf{o} + d \mathbf{u},

        where :math:`\mathbf{o}` is the starting point of the line,
        :math:`\mathbf{u}` is a unit vector pointing in the direction of the line,
        and :math:`d` is the distance from the origin to the intercept with
        the paraboloid.
        Combining these equations gives

        .. math::

            o_z + d u_z = \frac{(o_x + d u_x)^2 + (o_y + d u_y)^2}{4 f},

        which can then be solved for :math:`d` using the quadratic equation,

        .. math::

            d = \frac{-o_x u_x - o_y u_y + 2 f u_z
                      + \sqrt{-o_y^2 u_x^2 - o_x^2 u_y^2 + 2 o_y u_y (o_x u_x - 2 f u_z)
                              + 4 f (o_z (u_x^2 + u_y^2) - o_x u_x u_z + f u_z^2}
                      }{u_x^2 + u_y^2}.

        If the line is parallel to the :math:`z` axis, then the above equation
        is singular and we need to solve the corresponding linear equation to find

        .. math::

            d = \frac{o_x^2 + o_y^2 - 4 f o_z}{4 f u_z}.
        """

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        f = self.focal_length

        unit = f.unit

        f = f.value  # noqa: F841

        o = rays.position.to(unit).value

        u = rays.direction.value

        ox = o.x  # noqa: F841
        oy = o.y  # noqa: F841
        oz = o.z  # noqa: F841

        ux = u.x  # noqa: F841
        uy = u.y  # noqa: F841
        uz = u.z  # noqa: F841

        position = na.numexpr.evaluate(
            "o + u * where("
            "   (ux**2 + uy**2) > 1e-10,"
            "   (-ox * ux - oy * uy + 2 * f * uz - sign(f * uz) * sqrt("
            "       -(oy * ux)**2 - (ox * uy)**2 + 2 * oy * uy * (ox * ux - 2 * f * uz)"
            "       + 4 * f * (oz * (ux**2 + uy**2) - ox * ux * uz + f * uz**2)"
            "   )) / (ux**2 + uy**2),"
            "   (ox**2 + oy**2 - 4 * f * oz) / (4 * f * uz),"
            ")"
        )

        result = rays.replace(position=position << unit)

        if self.transformation is not None:
            result = self.transformation(result)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class ToroidalSag(
    AbstractSphericalSag[RadiusT],
    Generic[RadiusT, RadiusOfRotationT],
):
    """
    A toroidal sag profile.
    """

    radius: RadiusT = np.inf * u.mm
    """The minor radius of this toroidal surface."""

    radius_of_rotation: RadiusOfRotationT = 0 * u.mm
    """The major radius of this toroidal surface."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.radius_of_rotation),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        c = self.curvature
        r = self.radius_of_rotation
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(position, c, r)
        position = na.broadcast_to(position, shape)
        c = na.broadcast_to(c, shape)
        r = na.broadcast_to(r, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        zy = c * y2 / (1 + np.sqrt(1 - np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        return z

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        c = self.curvature
        r = self.radius_of_rotation
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(position, c, r)
        position = na.broadcast_to(position, shape)
        c = na.broadcast_to(c, shape)
        r = na.broadcast_to(r, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = position.x / f
        dzydy = c * position.y / g
        dzdy = (r - zy) * dzydy / f
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length
