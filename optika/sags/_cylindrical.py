import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag

__all__ = [
    "CylindricalSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class CylindricalSag(
    AbstractSag,
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

    radius: u.Quantity | na.AbstractScalar = np.inf * u.mm
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
