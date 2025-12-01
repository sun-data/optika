import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag

__all__ = [
    "AbstractConicSag",
    "ConicSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConicSag(
    AbstractSag,
):
    """An interface describing a general conic surface of revolution."""

    @property
    @abc.abstractmethod
    def radius(self) -> u.Quantity | na.AbstractScalar:
        """The effective radius of this conic section."""

    @property
    @abc.abstractmethod
    def conic(self) -> float | na.AbstractScalar:
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
    AbstractConicSag,
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

    radius: u.Quantity | na.AbstractScalar = np.inf * u.mm
    """The effective radius of this conic section."""

    conic: float | na.AbstractScalar = 0 * u.dimensionless_unscaled
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
