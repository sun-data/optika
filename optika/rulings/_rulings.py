import abc
import dataclasses
from dataclasses import MISSING
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractRulingSpacing

__all__ = [
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
    "SquareRulings",
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
    AbstractRulings,
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


@dataclasses.dataclass(eq=False, repr=False)
class SquareRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a square wave with a 50% duty cycle.

    Examples
    --------

    Compute the 1st-order groove efficiency of square rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SquareRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), square rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                \cos^2(\pi \gamma / 2) & i = 0 \\
                0 & i = \text{even} \\
                (2 / i \pi)^2 \sin^2 (\pi \gamma / 2) & i = \text{odd}, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """

        normal_rulings = self.spacing_(rays.position).normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        wavelength = rays.wavelength
        cos_theta = -direction @ normal
        d = self.depth
        i = self.diffraction_order

        gamma = np.pi * d / (wavelength * cos_theta)

        result = np.where(
            i % 2 == 0,
            x=0,
            y=np.square(2 * np.sin(np.pi * gamma / 2 * u.rad) / (i * np.pi)),
        )
        result = np.where(
            i == 0,
            x=np.square(np.cos(np.pi * gamma / 2 * u.rad)),
            y=result,
        )

        return result
