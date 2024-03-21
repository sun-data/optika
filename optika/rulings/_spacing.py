import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractRulingSpacing",
    "ConstantRulingSpacing",
    "Polynomial1dRulingSpacing",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulingSpacing(
    optika.mixins.Printable,
    optika.mixins.Transformable,
):
    """
    An interface describing the instantaneous ruling spacing on the surface
    of a diffraction grating.

    This is useful if you want to define a grating with variable line spacing.
    """

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The local ruling vector at the given position

        Parameters
        ----------
        position
            The location on the grating at which to evaluate the ruling spacing.
        """


@dataclasses.dataclass(eq=False, repr=False)
class ConstantRulingSpacing(
    AbstractRulingSpacing,
):
    """
    The simplest type of ruling spacing, a constant distance between each ruling.
    """

    constant: u.Quantity | na.AbstractScalar
    """The constant describing the ruling spacing."""

    normal: na.AbstractCartesian3dVectorArray
    """The unit vector normal to the planes of the rulings."""

    @property
    def transformation(self) -> None:
        return None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return self.constant * self.normal


@dataclasses.dataclass(eq=False, repr=False)
class Polynomial1dRulingSpacing(
    AbstractRulingSpacing,
):
    """
    Ruling spacing specified by a 1-dimensional polynomial.
    """

    coefficients: dict[int, u.Quantity | na.AbstractScalar]
    """
    The coefficients of the polynomial represented as a dictionary where
    the values are the coefficients and the keys are the power associated
    with each coefficient.
    """

    normal: na.AbstractCartesian3dVectorArray
    """The unit vector normal to the planes of the rulings."""

    transformation: None | na.transformations.AbstractTransformation = None
    """
    An arbitrary coordinate system transformation applied to the argument
    of the polynomial.
    """

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        coefficients = self.coefficients
        normal = self.normal
        transformation = self.transformation

        if transformation is not None:
            position = transformation(position)

        x = position @ normal

        result = 0 * u.mm
        for power, coefficient in coefficients.items():
            result = result + coefficient * (x**power)

        return result * normal
