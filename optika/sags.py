from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika

__all__ = [
    "AbstractSag",
    "SphericalSag",
    "ConicSag",
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


@dataclasses.dataclass
class AbstractSag(
    optika.transforms.Transformable,
):
    """
    Base class for all types of sag surfaces.
    """

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


@dataclasses.dataclass
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


@dataclasses.dataclass
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
    transform: None | optika.transforms.AbstractTransform = None

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
        radius = self.radius
        c = self.curvature
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

        shape = na.shape_broadcasted(radius, c, position)
        radius = na.broadcast_to(radius, shape)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        r2 = np.square(position.x) + np.square(position.y)
        sz = c * r2 / (1 + np.sqrt(1 - np.square(c) * r2))
        mask = r2 >= np.square(radius)
        sz[mask] = 0
        return sz

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        radius = self.radius
        c = self.curvature
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

        shape = na.shape_broadcasted(radius, c, position)
        radius = na.broadcast_to(radius, shape)
        c = na.broadcast_to(c, shape)
        position = na.broadcast_to(position, shape)

        x2, y2 = np.square(position.x), np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - c2 * (x2 + y2))
        dzdx, dzdy = c * position.x / g, c * position.y / g
        mask = (x2 + y2) >= np.square(radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length


@dataclasses.dataclass
class ConicSag(
    AbstractSphericalSag[RadiusT],
    Generic[RadiusT, ConicT],
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
            plt.legend(title="conic")
    """

    radius: RadiusT = np.inf * u.mm
    conic: ConicT = 0 * u.dimensionless_unscaled
    """the conic constant of the conic section"""
    transform: None | optika.transforms.AbstractTransform = None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        c = self.curvature
        conic = self.conic
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

        shape = na.shape_broadcasted(radius, c, conic, position)
        radius = na.broadcast_to(radius, shape)
        c = na.broadcast_to(c, shape)
        conic = na.broadcast_to(conic, shape)
        position = na.broadcast_to(position, shape)

        r2 = np.square(position.x) + np.square(position.y)
        sz = c * r2 / (1 + np.sqrt(1 - (1 + conic) * np.square(c) * r2))
        mask = r2 >= np.square(radius)
        sz[mask] = 0
        return sz

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        radius = self.radius
        c = self.curvature
        conic = self.conic
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

        shape = na.shape_broadcasted(radius, c, conic, position)
        radius = na.broadcast_to(radius, shape)
        c = na.broadcast_to(c, shape)
        conic = na.broadcast_to(conic, shape)
        position = na.broadcast_to(position, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - (1 + conic) * c2 * (x2 + y2))
        dzdx, dzdy = c * position.x / g, c * position.y / g
        mask = (x2 + y2) >= np.square(radius)
        dzdx[mask] = 0
        dzdy[mask] = 0
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length


@dataclasses.dataclass
class ToroidalSag(
    AbstractSphericalSag[RadiusT],
    Generic[RadiusT, RadiusOfRotationT],
):
    """
    A toroidal sag profile.
    """

    radius: RadiusT = np.inf * u.mm
    radius_of_rotation: RadiusOfRotationT = 0 * u.mm
    transform: None | optika.transforms.AbstractTransform = None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        c = self.curvature
        r = self.radius_of_rotation
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

        shape = na.shape_broadcasted(position, c, r)
        position = na.broadcast_to(position, shape)
        c = na.broadcast_to(c, shape)
        r = na.broadcast_to(r, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        mask = np.abs(position.x) > r
        zy = c * y2 / (1 + np.sqrt(1 - np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        z[mask] = (r - np.sqrt(np.square(r - zy) - np.square(r)))[mask]
        return z

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        c = self.curvature
        r = self.radius_of_rotation
        transform = self.transform
        if transform is not None:
            position = transform.inverse(position)

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
        mask = np.abs(position.x) > r
        dzdx[mask] = 0
        dzdy[mask] = 0
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length
