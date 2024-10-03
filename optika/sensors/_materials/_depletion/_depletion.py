import abc
import functools
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractDepletionModel",
    "AbstractJanesickDepletionModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDepletionModel(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """
    An arbitrary model of the depletion region of a semiconducting imaging
    sensor.
    """

    @property
    @abc.abstractmethod
    def thickness(self) -> u.Quantity:
        """The thickness of the depletion region."""


@dataclasses.dataclass(eq=False, repr=False)
class AbstractJanesickDepletionModel(
    AbstractDepletionModel,
):
    """
    A depletion model that relies on :cite:t:`Janesick2001`'s model of
    charge diffusion to measure the thickness of the field-free region
    of an imaging sensor and infer the thickness of the depletion region.
    """

    @property
    @abc.abstractmethod
    def chemical_substrate(self) -> optika.chemicals.AbstractChemical:
        """
        The optical properties of the substrate material.
        """

    @property
    @abc.abstractmethod
    def thickness_substrate(self) -> u.Quantity:
        """
        The thickness of the light-sensitive layer of the imaging sensor.
        """

    @property
    @abc.abstractmethod
    def width_pixel(self) -> u.Quantity:
        """The size of a pixel."""

    def mean_charge_capture(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The mean charge capture of this sensor for the given wavelength
        computed using :func:`optika.sensors.mean_charge_capture`

        Parameters
        ----------
        wavelength
            The wavelengths at which to evaluate the mean charge capture.
        """
        return optika.sensors.mean_charge_capture(
            width_diffusion=optika.sensors.charge_diffusion(
                absorption=self.chemical_substrate.absorption(wavelength),
                thickness_substrate=self.thickness_substrate,
                thickness_depletion=self.thickness,
            ),
            width_pixel=self.width_pixel,
        )

    @property
    @abc.abstractmethod
    def mean_charge_capture_measured(self) -> na.FunctionArray:
        """
        The measured mean charge capture that will be fit by the function
        :func:`optika.sensors.mean_charge_capture`.
        """

    @functools.cached_property
    def thickness(self) -> u.Quantity:

        thickness_substrate = self.thickness_substrate
        width_pixel = self.width_pixel

        mcc_measured = self.mean_charge_capture_measured

        absorption = self.chemical_substrate.absorption(mcc_measured.inputs)

        unit = u.um

        def objective(thickness_depletion: float) -> float:

            width_diffusion = optika.sensors.charge_diffusion(
                absorption=absorption,
                thickness_substrate=thickness_substrate,
                thickness_depletion=thickness_depletion * unit,
            )

            mcc = optika.sensors.mean_charge_capture(
                width_diffusion=width_diffusion,
                width_pixel=width_pixel,
            )

            diff = mcc_measured.outputs - mcc

            result = np.sqrt(np.mean(np.square(diff))).ndarray

            return result

        fit = scipy.optimize.minimize_scalar(
            fun=objective,
            bounds=(
                0,
                thickness_substrate.to_value(unit),
            ),
        )

        return fit.x * unit
