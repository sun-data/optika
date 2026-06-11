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
    optical surface, since the perturbation modifies the actual shape of the
    surface, it is seen consistently by both geometric raytraces and
    physical-optics calculations.

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
            sag=optika.sags.ParabolicSag(focal_length=500 * u.mm),
            coefficients=[0, 0, 0, 0, 0, 0, 0, 1] * u.mm,
            radius=50 * u.mm,
        )

        position = na.Cartesian3dVectorArray(
            x=0 * u.mm,
            y=na.linspace(-50, 50, axis="y", num=101) * u.mm,
            z=0 * u.mm,
        )

        z = sag(position)
        z_base = sag.sag(position)

        with astropy.visualization.quantity_support():
            plt.figure()
            na.plt.plot(position.y, z, axis="y", label="perturbed")
            na.plt.plot(position.y, z_base, axis="y", label="base")
            plt.legend()
    """

    sag: None | AbstractSag = None
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

    def __post_init__(self):
        if self.sag is None:
            self.sag = NoSag()

    @property
    def _coefficients_normalized(self) -> na.AbstractScalar:
        """The coefficients as a named array guaranteed to contain `axis`."""
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
        shape_coefficients = dict(self._coefficients_normalized.shape)
        shape_coefficients.pop(self.axis, None)
        return na.broadcast_shapes(
            optika.shape(self.sag),
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

        result = self.sag(position)

        coefficients = self._coefficients_normalized
        position_normalized = position.xy / self.radius

        for i in range(coefficients.shape[self.axis]):
            c = coefficients[{self.axis: i}]
            result = result + c * optika.zernikes.zernike(
                j=i + 1,
                position=position_normalized,
            )

        return result

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        normal_base = self.sag.normal(position)

        gradient_x = normal_base.x / -normal_base.z
        gradient_y = normal_base.y / -normal_base.z

        coefficients = self._coefficients_normalized
        radius = self.radius
        position_normalized = position.xy / radius

        for i in range(coefficients.shape[self.axis]):
            c = coefficients[{self.axis: i}]
            gradient = optika.zernikes.zernike_gradient(
                j=i + 1,
                position=position_normalized,
            )
            gradient_x = gradient_x + c * gradient.x / radius
            gradient_y = gradient_y + c * gradient.y / radius

        norm = np.sqrt(np.square(gradient_x) + np.square(gradient_y) + 1)

        return na.Cartesian3dVectorArray(
            x=gradient_x / norm,
            y=gradient_y / norm,
            z=-1 / norm,
        )
