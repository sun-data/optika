import abc
import dataclasses
from dataclasses import MISSING
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractRulingSpacing

__all__ = [
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
    optika.mixins.Printable,
):
    """
    Interface for the interaction of a ruled surface with incident light
    """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> int | na.AbstractScalar:
        """
        the diffraction order to simulate
        """

    @property
    @abc.abstractmethod
    def spacing(
        self,
    ) -> u.Quantity | na.AbstractScalar | AbstractRulingSpacing:
        """
        Spacing between adjacent rulings at the given position.
        """

    @property
    def spacing_(self) -> AbstractRulingSpacing:
        """
        A normalized version of :attr:`spacing` that is guaranteed to be
        an instance of :class:`optika.rulings.AbstractRulingSpacing`.
        """
        spacing = self.spacing
        if not isinstance(spacing, optika.rulings.AbstractRulingSpacing):
            spacing = optika.rulings.ConstantRulingSpacing(
                constant=spacing,
                normal=na.Cartesian3dVectorArray(1, 0, 0),
            )
        return spacing

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        """
        The fraction of light that is diffracted into a given order.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        """


@dataclasses.dataclass(eq=False, repr=False)
class Rulings(
    AbstractRulings
):
    """
    An idealized set of rulings which have perfect efficiency in all diffraction
    orders.
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    def efficiency(
            self,
            rays: optika.rays.RayVectorArray,
            normal: na.AbstractCartesian3dVectorArray,
    ) -> float:
        return 1


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredRulings(
    AbstractRulings,
):
    """
    A set of rulings where the efficiency has been measured or calculated
    by an independent source.
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    efficiency_measured: na.FunctionArray = MISSING
    """The discrete measurements of the efficiency."""

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        measurement = self.efficiency_measured

        wavelength = measurement.inputs.wavelength
        direction = measurement.inputs.direction
        efficiency = measurement.outputs

        if direction.size != 1:  # pragma: nocover
            raise ValueError(
                f"Interpolating over different incidence angles is not supported."
            )

        if wavelength.ndim != 1:  # pragma: nocover
            raise ValueError(
                f"wavelength must be one dimensional, got shape {wavelength.shape}"
            )

        return na.interp(
            x=rays.wavelength,
            xp=wavelength,
            fp=efficiency,
        )
