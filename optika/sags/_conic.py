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

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.RayVectorArray:
        r"""
        Compute the intercept of the given rays with this conic surface of
        revolution.

        The intersection is found in closed form by solving the ray-quadric
        intersection, which avoids the spurious root that an iterative solver
        can converge to on the steep flank of a grazing-incidence conic.

        Parameters
        ----------
        rays
            The rays to intercept with this surface.

        Notes
        -----
        A conic of revolution about the :math:`z` axis, with its vertex at the
        origin, satisfies the implicit equation

        .. math::

            c (x^2 + y^2) + (1 + k) c z^2 - 2 z = 0,

        where :math:`c = 1 / R` is the vertex curvature and :math:`k` is the
        conic constant.  Substituting the parametric ray
        :math:`\mathbf{x} = \mathbf{o} + t \mathbf{u}` gives a quadratic
        :math:`A t^2 + B t + C = 0` in the path length :math:`t`, with

        .. math::

            A &= c \left[ u_x^2 + u_y^2 + (1 + k) u_z^2 \right] \\
            B &= 2 \left[ c (o_x u_x + o_y u_y + (1 + k) o_z u_z) - u_z \right] \\
            C &= c \left[ o_x^2 + o_y^2 + (1 + k) o_z^2 \right] - 2 o_z.

        Of the (up to two) real roots, the intercept is the one on the same
        sheet of the conic as the vertex (identified by
        :math:`z \, (c (x^2 + y^2) - z) \geq 0`) that is nearest the ray's
        starting point.  An iterative solver, by contrast, can converge to the
        far or wrong-sheet root on the steep flank of a grazing conic.
        """
        transformation = self.transformation
        if transformation is not None:
            rays = transformation.inverse(rays)

        c = 1 / self.radius
        kp1 = 1 + self.conic

        o = rays.position
        u = rays.direction

        a = c * (np.square(u.x) + np.square(u.y) + kp1 * np.square(u.z))
        b = 2 * (c * (o.x * u.x + o.y * u.y + kp1 * o.z * u.z) - u.z)
        cc = c * (np.square(o.x) + np.square(o.y) + kp1 * np.square(o.z)) - 2 * o.z

        discriminant = np.square(b) - 4 * a * cc
        real = discriminant >= 0
        sqrt_discriminant = np.sqrt(np.where(real, discriminant, 0))

        # guard against the degenerate (nearly linear, A -> 0) case
        unit_a = na.unit_normalized(a)
        degenerate = np.abs(a) < (1e-12 * unit_a)
        denominator = np.where(degenerate, 1 * unit_a, 2 * a)
        t_linear = -cc / b

        def root(sign: int) -> na.AbstractScalar:
            t = np.where(
                degenerate,
                t_linear,
                (-b + sign * sqrt_discriminant) / denominator,
            )
            position = o + u * t
            r2 = np.square(position.x) + np.square(position.y)
            on_vertex_sheet = (position.z * (c * r2 - position.z)) >= 0
            valid = real & on_vertex_sheet
            return np.where(valid, t, np.inf * na.unit_normalized(t))

        # of the (up to two) roots, take the one on the vertex sheet nearest the
        # ray's current position.  This selects the physical intercept and avoids
        # the far / wrong-sheet root that an iterative solver can land on along
        # the steep flank of a grazing conic.
        t_a = root(-1)
        t_b = root(+1)
        t = np.where(np.abs(t_a) <= np.abs(t_b), t_a, t_b)

        result = rays.copy_shallow()
        result.position = o + u * t

        if transformation is not None:
            result = transformation(result)

        return result


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
