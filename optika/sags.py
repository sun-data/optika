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
):
    """
    Base class for all types of sag surfaces.
    """

    @property
    @abc.abstractmethod
    def parameters_slope_error(self) -> None | optika.metrology.SlopeErrorParameters:
        """collection of parameters to use when computing the slope error"""

    @property
    @abc.abstractmethod
    def parameters_roughness(self) -> None | optika.metrology.RoughnessParameters:
        """collection of parameters to use when computing the roughness"""

    @property
    @abc.abstractmethod
    def parameters_microroughness(self) -> None | optika.metrology.RoughnessParameters:
        """collection of parameters to use when computing the microroughness"""

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
    ) -> optika.rays.AbstractRayVectorArray:
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
        )

        result = rays.copy_shallow()
        result.position = line(t_intercept)
        return result


@dataclasses.dataclass(eq=False, repr=False)
class NoSag(
    AbstractSag,
):
    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    parameters_microroughness: None | optika.metrology.RoughnessParameters = None

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    @property
    def transformation(self) -> None:
        return None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        return 0 * u.mm

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(0, 0, -1)


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
        radius of curvature of the sag surface
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
    transformation: None | na.transformations.AbstractTransformation = None
    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    parameters_microroughness: None | optika.metrology.RoughnessParameters = None

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
    ) -> na.AbstractCartesian3dVectorArray:
        c = self.curvature
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, position)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        x2, y2 = np.square(position.x), np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - c2 * (x2 + y2))
        dzdx, dzdy = c * position.x / g, c * position.y / g
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length


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
    """The radius of the cylinder."""

    transformation: None | na.transformations.AbstractTransformation = None
    """
    The transformation between the surface coordinate system and the sag
    coordinate system.
    """

    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    """A set of parameters describing the slope error of the sag function."""

    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    """A set of parameters describing the roughness of the sag function."""

    parameters_microroughness: None | optika.metrology.RoughnessParameters = None
    """A set of parameters describing the microroughness of the sag function."""

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

        c = 1 / self.radius
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(c, position)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        x2 = np.square(position.x)
        c2 = np.square(c)
        g = np.sqrt(1 - c2 * x2)
        dzdx = c * position.x / g
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=0,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConicSag(
    AbstractSag,
    Generic[RadiusT, ConicT],
):
    @property
    @abc.abstractmethod
    def radius(self) -> RadiusT:
        """the effective radius of the conic section"""

    @property
    @abc.abstractmethod
    def conic(self) -> ConicT:
        """conic constant of the conic section"""

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
    conic: ConicT = 0 * u.dimensionless_unscaled
    """the conic constant of the conic section"""
    transformation: None | na.transformations.AbstractTransformation = None
    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    parameters_microroughness: None | optika.metrology.RoughnessParameters = None

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
    focal_length: FocalLengthT = np.inf * u.mm
    transformation: None | na.transformations.AbstractTransformation = None
    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    parameters_microroughness: None | optika.metrology.RoughnessParameters = None

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


@dataclasses.dataclass(eq=False, repr=False)
class ToroidalSag(
    AbstractSphericalSag[RadiusT],
    Generic[RadiusT, RadiusOfRotationT],
):
    """
    A toroidal sag profile.
    """

    radius: RadiusT = np.inf * u.mm
    radius_of_rotation: RadiusOfRotationT = 0 * u.mm
    transformation: None | na.transformations.AbstractTransformation = None
    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = None
    parameters_roughness: None | optika.metrology.RoughnessParameters = None
    parameters_microroughness: None | optika.metrology.RoughnessParameters = None

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
