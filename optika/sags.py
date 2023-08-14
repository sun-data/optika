from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractSag",
    "SphericalSag",
    "ConicSag",
    "ToroidalSag",
]

RadiusT = TypeVar("RadiusT", bound=na.ScalarLike)
ConicT = TypeVar("ConicT", bound=na.ScalarLike)
RadiusOfRotationT = TypeVar("RadiusOfRotationT", bound=na.ScalarLike)


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
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        pass

    @abc.abstractmethod
    def normal(
        self,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The vector perpendicular to the surface at the given position.

        Parameters
        ----------
        position
            The location on the surface to evaluate the normal vector
        """


@dataclasses.dataclass
class SphericalSag(
    AbstractSag,
    Generic[RadiusT],
):
    """
    A spherical sag profile
    """

    radius: RadiusT = np.inf * u.mm
    """the radius of the spherical surface"""

    @property
    def curvature(self) -> RadiusT:
        return 1 / self.radius

    def __call__(
        self,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        c = self.curvature

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
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        radius = self.radius
        c = self.curvature

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
    SphericalSag[RadiusT],
    Generic[RadiusT, ConicT],
):
    """
    A conic section sag profile
    """

    conic: ConicT = 0 * u.dimensionless_unscaled
    """the conic constant of the conic section"""

    def __call__(
        self,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        radius = self.radius
        c = self.curvature
        conic = self.conic

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
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        radius = self.radius
        c = self.curvature
        conic = self.conic

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
    SphericalSag[RadiusT],
    Generic[RadiusT, RadiusOfRotationT],
):
    """
    A toroidal sag profile.
    """

    radius_of_rotation: RadiusOfRotationT = 0 * u.mm

    def __call__(
        self,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        c = self.curvature
        r = self.radius_of_rotation

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
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        c = self.curvature
        r = self.radius_of_rotation

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
