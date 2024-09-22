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
    optika.mixins.Shaped,
):
    """
    An interface describing the instantaneous ruling spacing on the surface
    of a diffraction grating.

    This is useful if you want to define a grating with variable line spacing.
    """

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The local ruling vector at the given position

        Parameters
        ----------
        position
            The location on the grating at which to evaluate the ruling spacing.
        normal
            The unit vector perpendicular to the surface at the given position.
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
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.constant),
            optika.shape(self.normal),
        )

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

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.coefficients),
            optika.shape(self.normal),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        coefficients = self.coefficients
        normal_rulings = self.normal
        transformation = self.transformation

        if transformation is not None:
            position = transformation(position)

        x = position @ normal_rulings

        result = 0 * u.mm
        for power, coefficient in coefficients.items():
            result = result + coefficient * (x**power)

        return result * normal_rulings


@dataclasses.dataclass(eq=False, repr=False)
class HolographicRulingSpacing(
    AbstractRulingSpacing,
):
    """
    Rulings created by interfering two beams.
    """

    x1: na.AbstractCartesian3dVectorArray
    """
    The origin of the first recording beam in local coordinates.
    """

    x2: na.AbstractCartesian3dVectorArray
    """
    The origin of the second recording beam in local coordinates.
    """

    wavelength: u.Quantity | na.AbstractScalar
    """
    The wavelength of the recording beams.
    """

    is_diverging_1: bool | na.AbstractScalar = True
    """
    A boolean flag indicating if rays are diverging from the origin of the
    first beam.
    """

    is_diverging_2: bool | na.AbstractScalar = True
    """
    A boolean flag indicating if rays are diverging from the origin of the
    second beam.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.x1),
            optika.shape(self.x2),
            optika.shape(self.is_diverging_1),
            optika.shape(self.is_diverging_2),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        x1 = self.x1.normalized
        x2 = self.x2.normalized
        wavelength = self.wavelength
        is_diverging_1 = self.is_diverging_1
        is_diverging_2 = self.is_diverging_2

        x = (x2 - x1) / wavelength

        result = normal.cross(x)

        return result
