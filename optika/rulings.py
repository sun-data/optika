import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import optika.mixins

__all__ = [
    "AbstractRulings",
    "AbstractConstantDensityRulings",
    "ConstantDensityRulings",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
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
    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def ruling_density(self) -> na.ScalarLike:
        """
        the frequency of the ruling pattern
        """

    def spacing(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1 / self.ruling_density

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(1, 0, 0)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConstantDensityRulings(
    AbstractPolynomialDensityRulings,
):
    def frequency(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return self.ruling_density


@dataclasses.dataclass(eq=False, repr=False)
class ConstantDensityRulings(
    AbstractConstantDensityRulings,
):
    ruling_density: na.ScalarLike = 0 / u.mm
    diffraction_order: na.ScalarLike = 1
    transformation: None | na.transformations.AbstractTransformation = None
