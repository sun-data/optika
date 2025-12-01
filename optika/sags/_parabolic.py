import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._conic import AbstractConicSag

__all__ = [
    "ParabolicSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class ParabolicSag(
    AbstractConicSag,
):
    """A parabolic sag profile."""

    focal_length: u.Quantity | na.AbstractScalar = np.inf * u.mm
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
    def radius(self) -> u.Quantity | na.AbstractScalar:
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
                      - \text{sgn}(f u_z) \sqrt{-o_y^2 u_x^2 - o_x^2 u_y^2 + 2 o_y u_y (o_x u_x - 2 f u_z)
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
