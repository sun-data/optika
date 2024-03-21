import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractRulings",
    "AbstractPolynomialDensityRulings",
    "PolynomialDensityRulings",
    "AbstractConstantDensityRulings",
    "ConstantDensityRulings",
    "AbstractPolynomialSpacingRulings",
    "PolynomialSpacingRulings",
    "AbstractConstantSpacingRulings",
    "ConstantSpacingRulings",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
    optika.mixins.Printable,
    optika.mixins.Transformable,
):
    """
    Interface for the interaction of a ruled surface with incident light
    """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> na.ScalarLike:
        """
        the diffraction order to simulate
        """

    @abc.abstractmethod
    def spacing(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        spacing between adjacent rulings at the given position

        Parameters
        ----------
        position
            location to evaluate the ruling spacing
        """

    @abc.abstractmethod
    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        vector normal to the planes of the rulings

        Parameters
        ----------
        position
            location to evaluate the normal vector
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolynomialDensityRulings(
    AbstractRulings,
):
    @property
    @abc.abstractmethod
    def coefficients(self) -> None | dict[int, na.ScalarLike]:
        """The coefficients of the polynomial describing the ruling density"""

    def frequency(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        Density of rulings at the given position.
        Equivalent to the multiplicative inverse of the ruling spacing.

        Parameters
        ----------
        position
            location to evaluate the ruling density
        """
        transformation = self.transformation
        if transformation is not None:
            position = transformation(position)

        x = position @ self.normal(position)

        coefficients = self.coefficients
        if coefficients is None:
            coefficients = dict()

        result = 0 / u.mm
        for power, coefficient in coefficients.items():
            result = result + coefficient * (x**power)

        return result

    def spacing(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1 / self.frequency(position)

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(1, 0, 0)


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialDensityRulings(
    AbstractPolynomialDensityRulings,
):
    coefficients: None | dict[int, na.ScalarLike] = None
    diffraction_order: na.ScalarLike = 1
    transformation: None | na.transformations.AbstractTransformation = None


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConstantDensityRulings(
    AbstractPolynomialDensityRulings,
):
    @property
    @abc.abstractmethod
    def density(self) -> na.ScalarLike:
        """
        the frequency of the ruling pattern
        """

    @property
    def coefficients(self) -> None | dict[int, na.ScalarLike]:
        return {0: self.density}


@dataclasses.dataclass(eq=False, repr=False)
class ConstantDensityRulings(
    AbstractConstantDensityRulings,
):
    density: na.ScalarLike = 0 / u.mm
    diffraction_order: na.ScalarLike = 1
    transformation: None | na.transformations.AbstractTransformation = None


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolynomialSpacingRulings(
    AbstractRulings,
):
    @property
    @abc.abstractmethod
    def coefficients(self) -> None | dict[int, na.ScalarLike]:
        """The coefficients of the polynomial describing the ruling density"""

    def spacing(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        transformation = self.transformation
        if transformation is not None:
            position = transformation(position)

        x = position @ self.normal(position)

        coefficients = self.coefficients
        if coefficients is None:
            coefficients = dict()

        result = 0 * u.mm
        for power, coefficient in coefficients.items():
            result = result + coefficient * (x**power)

        return result

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(1, 0, 0)


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialSpacingRulings(
    AbstractPolynomialSpacingRulings,
):
    coefficients: None | dict[int, na.ScalarLike] = None
    diffraction_order: na.ScalarLike = 1
    transformation: None | na.transformations.AbstractTransformation = None


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConstantSpacingRulings(
    AbstractPolynomialSpacingRulings,
):
    @property
    @abc.abstractmethod
    def period(self) -> na.ScalarLike:
        """
        the spacing of the ruling pattern
        """

    @property
    def coefficients(self) -> None | dict[int, na.ScalarLike]:
        return {0: self.period}


@dataclasses.dataclass(eq=False, repr=False)
class ConstantSpacingRulings(
    AbstractConstantSpacingRulings,
):
    period: na.ScalarLike = 0 * u.mm
    diffraction_order: na.ScalarLike = 1
    transformation: None | na.transformations.AbstractTransformation = None
