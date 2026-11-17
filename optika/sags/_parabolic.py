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

        which is the quadratic equation :math:`A d^2 + B d + C = 0`, with

        .. math::

            A &= u_x^2 + u_y^2 \\
            B &= 2 (o_x u_x + o_y u_y - 2 f u_z) \\
            C &= o_x^2 + o_y^2 - 4 f o_z.

        Its roots are computed with the numerically stable form of the
        quadratic formula,

        .. math::

            q &= -\frac{B + \operatorname{sgn}(B) \sqrt{B^2 - 4 A C}}{2} \\
            d &= \frac{q}{A}, \quad \frac{C}{q},

        with :math:`\operatorname{sgn}(0) = 1`, so that :math:`q` vanishes only
        where both :math:`B` and the discriminant do.
        The intercept is the root with the smaller :math:`|d|`, the crossing
        nearest the ray's current position.
        The textbook form, :math:`(-B \pm \sqrt{B^2 - 4 A C}) / 2 A`,
        subtracts two nearly equal numbers to find the nearer root of a ray
        travelling nearly parallel to the axis, where :math:`A` is close to
        zero, and the stable form also needs no special case for a ray
        parallel to the axis, where :math:`A = 0` and :math:`C / q` reduces to
        the root of the linear equation.
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

        # The coefficients of the quadratic equation and the stable form of
        # the quadratic formula, see the notes above.
        a = "(ux**2 + uy**2)"
        b = "(2 * (ox * ux + oy * uy - 2 * f * uz))"
        c = "(ox**2 + oy**2 - 4 * f * oz)"
        q = f"(-({b} + where({b} >= 0, 1.0, -1.0) * sqrt({b}**2 - 4 * {a} * {c})) / 2)"
        root_q = f"({q} / {a})"
        root_c = f"({c} / {q})"

        # Take the root nearest the ray's current position, as
        # `AbstractConicSag.intercept` does for every other conic.  Selecting
        # the root by `sign(f * uz)` instead returns the far crossing for a ray
        # that starts on the surface, which moves it to the opposite side of
        # the paraboloid; `optika.systems.SequentialSystem` does exactly that
        # when it propagates the solved stop rays back through the surface they
        # were launched from.
        #
        # `root_q` is infinite or not-a-number for a ray parallel to the axis
        # (`a == 0`), and `root_c` is not-a-number for a ray which starts on
        # the surface tangent to it (`q == c == 0`), so each is taken only
        # where the other is not-a-number or farther away.  The comparison is
        # the same as in `AbstractConicSag.intercept`.
        #
        # Both roots are built into one `numexpr` expression on purpose:
        # evaluating them separately and combining them afterwards measures
        # about 50% slower.
        position = na.numexpr.evaluate(
            "o + u * where("
            f"   (abs({root_q}) <= abs({root_c})) | ({root_c} != {root_c}),"
            f"   {root_q},"
            f"   {root_c},"
            ")"
        )

        result = rays.replace(position=position << unit)

        if self.transformation is not None:
            result = self.transformation(result)

        return result
