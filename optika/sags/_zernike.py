import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag
from ._flat import NoSag

__all__ = [
    "ZernikeSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class ZernikeSag(
    AbstractSag,
):
    r"""
    A sag profile consisting of a base profile plus a sum of Zernike
    polynomials.

    This is useful for representing measured or modeled figure errors of an
    optical surface.
    Since the perturbation modifies the actual shape of the surface, it is
    seen consistently by both geometric raytraces and physical-optics
    calculations.

    Examples
    --------
    Plot a slice through a parabolic sag profile with a large coma
    perturbation.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        sag = optika.sags.ZernikeSag(
            base=optika.sags.ParabolicSag(focal_length=500 * u.mm),
            coefficients=[0, 0, 0, 0, 0, 0, 0, 1] * u.mm,
            radius=50 * u.mm,
        )

        # the coma term (Noll index 8) varies along x, so slice along x
        position = na.Cartesian3dVectorArray(
            x=na.linspace(-50, 50, axis="x", num=101) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm,
        )

        z = sag(position)
        z_base = sag.base(position)

        with astropy.visualization.quantity_support():
            plt.figure()
            na.plt.plot(position.x, z, axis="x", label="perturbed")
            na.plt.plot(position.x, z_base, axis="x", label="base")
            plt.legend()
    """

    base: None | AbstractSag = None
    """
    The base sag profile to be perturbed.
    If :obj:`None` (the default), a flat profile, :class:`optika.sags.NoSag`,
    is used.
    """

    coefficients: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The magnitudes of the Zernike polynomial terms along the logical axis
    `axis`, where element :math:`i` corresponds to Noll index :math:`j = i + 1`.
    If given as a bare array or scalar, it is interpreted as being along
    `axis`.
    """

    radius: u.Quantity | na.AbstractScalar = 1 * u.mm
    """
    The radius of the unit disk on which the Zernike polynomials are defined,
    in the same coordinate system as the evaluation points.
    """

    axis: str = "zernike"
    """The logical axis of `coefficients` indexing the Noll terms."""

    @property
    def base_(self) -> AbstractSag:
        """The base sag profile, with :obj:`None` resolved to a flat profile."""
        if self.base is None:
            return NoSag()
        return self.base

    @property
    def coefficients_(self) -> na.AbstractScalar:
        """
        The coefficients as a named array guaranteed to vary along `axis`.

        A bare array or scalar is interpreted as being along `axis`.
        A named array which does not vary along `axis` is rejected here, so
        that :attr:`shape` fails fast instead of reporting the misnamed axis
        as if it were a batch axis.
        """
        result = self.coefficients
        if not isinstance(result, na.AbstractArray):
            result = np.atleast_1d(u.Quantity(result))
            result = na.ScalarArray(result, axes=(self.axis,))
        if self.axis not in result.shape:
            raise ValueError(
                f"`coefficients` must vary along `axis`, {self.axis!r}, "
                f"got an array with shape {result.shape}."
            )
        return result

    @property
    def shape(self) -> dict[str, int]:
        shape_coefficients = dict(self.coefficients_.shape)
        shape_coefficients.pop(self.axis, None)
        return na.broadcast_shapes(
            optika.shape(self.base_),
            shape_coefficients,
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

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        result = self.base_(position)

        return result + optika.zernikes.zernike_sum(
            position=position.xy / self.radius,
            coefficients=self.coefficients_,
            axis=self.axis,
        )

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

        Notes
        -----
        There is no closed-form intercept with an arbitrary Zernike sum, so
        this is found iteratively, as in
        :meth:`optika.sags.AbstractSag.intercept`.
        Since the perturbation is usually small compared to the base profile,
        the search is seeded from the intercept with the base profile alone,
        and the Zernike sum is collected into its harmonics once rather than
        on every evaluation.
        """
        base = self.base_
        radius = self.radius
        harmonics = optika.zernikes._harmonics(
            coefficients=self.coefficients_,
            axis=self.axis,
        )

        transformation = self.transformation
        if transformation is not None:
            rays = transformation.inverse(rays)

        # Start the search from the base profile, which the perturbation
        # usually moves by nanometres.
        rays = base.intercept(rays)

        def line(t: na.AbstractScalar) -> na.Cartesian3dVectorArray:
            return rays.position + rays.direction * t

        def func(t: na.AbstractScalar) -> na.AbstractScalar:
            a = line(t)
            z = base(a) + optika.zernikes._sum(a.xy / radius, harmonics)
            return a.z - z

        t_intercept = na.optimize.root_secant(
            function=func,
            guess=0 * u.mm,
            min_step_size=1e-6 * u.mm,
        )

        result = rays.copy_shallow()
        result.position = line(t_intercept)

        if transformation is not None:
            result = transformation(result)

        return result

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        normal_base = self.base_.normal(position)

        radius = self.radius

        gradient = optika.zernikes.zernike_sum_gradient(
            position=position.xy / radius,
            coefficients=self.coefficients_,
            axis=self.axis,
        )
        gradient = gradient / radius

        # The perturbed normal is parallel to (gradient_base + gradient, -1),
        # where gradient_base is the slope of the base profile.
        # Multiplying through by -normal_base.z turns the first term into
        # normal_base.xy without ever dividing by normal_base.z, so the result
        # stays finite where the base profile is vertical.
        result = na.Cartesian3dVectorArray(
            x=normal_base.x - normal_base.z * gradient.x,
            y=normal_base.y - normal_base.z * gradient.y,
            z=normal_base.z,
        )

        return result / result.length
