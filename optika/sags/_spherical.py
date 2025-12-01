import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag

__all__ = [
    "AbstractSphericalSag",
    "SphericalSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSphericalSag(
    AbstractSag,
):
    """
    Base class for all sag functions that can be approximated as a sphere.
    """

    @property
    @abc.abstractmethod
    def radius(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of curvature of this sphere.
        """

    @property
    def curvature(self) -> u.Quantity | na.AbstractScalar:
        """
        The curvature of the spherical surface.
        Equal to the reciprocal of :attr:`radius`.
        """
        return 1 / self.radius


@dataclasses.dataclass(eq=False, repr=False)
class SphericalSag(
    AbstractSphericalSag,
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

    radius: u.Quantity | na.AbstractScalar = np.inf * u.mm
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
