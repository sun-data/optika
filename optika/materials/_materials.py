from __future__ import annotations
import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from ._layers import Layer

__all__ = [
    "AbstractMaterial",
    "Vacuum",
    "AbstractMirror",
    "Mirror",
    "MeasuredMirror",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMaterial(
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    @abc.abstractmethod
    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the index of refraction of this material for the given input rays

        Parameters
        ----------
        rays
            input rays used to evaluate the index of refraction
        """

    @abc.abstractmethod
    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the attenuation coefficient of the given rays

        Parameters
        ----------
        rays
            input rays to calculate the attenuation coefficient for
        """

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        The fraction of light that passes through the interface.

        Parameters
        ----------
        rays
            the input rays to calculate the efficiency for
        normal
            the vector perpendicular to the optical surface
        """

    @property
    @abc.abstractmethod
    def is_mirror(self) -> bool:
        """
        flag controlling whether this material reflects or transmits light
        """


@dataclasses.dataclass(eq=False, repr=False)
class Vacuum(
    AbstractMaterial,
):
    @property
    def shape(self) -> dict[str, int]:
        return dict()

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 0 / u.mm

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMirror(
    AbstractMaterial,
):

    @property
    @abc.abstractmethod
    def substrate(self) -> None | Layer:
        """
        A layer representing the substrate supporting the reflective surface.
        """

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.index_refraction

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.attenuation

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return True


@dataclasses.dataclass(eq=False, repr=False)
class Mirror(
    AbstractMirror,
):
    substrate: None | Layer = None
    """A layer representing the substrate supporting the reflective surface."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.substrate),
        )


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredMirror(
    AbstractMirror,
):
    """
    A mirror where the reflectivity has been measured by an external source as
    a function of wavelength.

    Examples
    --------

    Create a mirror where the reflectivity is a Gaussian centered
    at 304 Angstroms.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the mean and standard deviation of the reflectivity peak.
        center = 304 * u.AA
        width = 10 * u.AA

        # Define a grid of wavelengths
        wavelength_min = center - 3 * width
        wavelength_max = center + 3 * width
        wavelength = na.linspace(
            start=wavelength_min,
            stop=wavelength_max,
            axis="wavelength",
            num=11,
        )

        # Define an array of simulated reflectivity measurements
        efficiency = na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            outputs=np.exp(-np.square((wavelength - center) / width) / 2),
        )

        # Create an instance of a MeasuredMirror object
        mirror = optika.materials.MeasuredMirror(efficiency)

        # Define a new grid of wavelengths at which to evaluate the interpolated
        # reflectivity
        wavelength_interp = na.linspace(
            start=wavelength_min,
            stop=wavelength_max,
            axis="wavelength",
            num=1001,
        )

        # Evaluate the interpolated reflectivity
        efficiency_interp = mirror.efficiency(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength_interp,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Plot the interpolated reflectivity vs the measured reflectivity
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(wavelength_interp, efficiency_interp, label="interpolated");
            na.plt.scatter(efficiency.inputs.wavelength, efficiency.outputs, label="measured");
            ax.set_xlabel(f"wavelength ({ax.xaxis.get_label().get_text()})");
            ax.set_ylabel(f"reflectivity");
            ax.legend();
    """

    efficiency_measured: na.FunctionArray[
        na.SpectralDirectionalVectorArray, na.AbstractScalar
    ] = dataclasses.MISSING
    """
    A function array that maps wavelengths and incidence angles to the
    measured reflectivity.
    """

    substrate: None | Layer = None
    """A layer representing the substrate supporting the reflective surface."""

    serial_number: None | str | na.AbstractArray = None
    """A unique number associated with this material"""

    @property
    def shape(self) -> dict[str, int]:
        axis_wavelength = self.efficiency_measured.inputs.wavelength.axes
        shape = optika.shape(self.efficiency_measured.outputs)
        for ax in axis_wavelength:
            shape.pop(ax, None)
        return na.broadcast_shapes(
            shape,
            optika.shape(self.substrate),
            optika.shape(self.serial_number),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:

        measurement = self.efficiency_measured

        wavelength = measurement.inputs.wavelength
        direction = measurement.inputs.direction
        efficiency = measurement.outputs

        if direction.size != 1:  # pragma: nocover
            raise ValueError(
                "Interpolating over different incidence angles is not supported."
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
