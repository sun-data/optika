import abc
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractDepletionModel",
    "JanesickDepletionModel",
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
class JanesickDepletionModel(
    AbstractDepletionModel,
):
    """
    A depletion model that relies on :cite:t:`Janesick2001`'s model of
    charge diffusion to measure the thickness of the field-free region
    of an imaging sensor and infer the thickness of the depletion region.
    """

    thickness: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The thickness of the depletion region of this sensor."""

    thickness_substrate: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The thickness of the light-sensitive region of this sensor."""

    chemical_substrate: optika.chemicals.AbstractChemical = dataclasses.MISSING
    """A model of the optical properties of the substrate material."""

    width_pixel: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The physical size of a pixel on this sensor."""

    mcc_measured: None | na.FunctionArray = None
    """
    An optional measurement of the mean charge capture that can be used to
    compute residuals.
    """

    @classmethod
    def fit_mcc(
        cls,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        chemical_substrate: optika.chemicals.AbstractChemical,
        width_pixel: u.Quantity | na.AbstractScalar,
        mcc_measured: None | na.FunctionArray = None,
    ):
        """
        Given a measured mean charge capture,
        find the thickness of the depletion region which best matches the
        measurements.

        Parameters
        ----------
        thickness_substrate
            The thickness of the light-sensitive region of this sensor.
        chemical_substrate
            A model of the optical properties of the substrate material.
        width_pixel
            The physical size of a pixel on this sensor.
        mcc_measured
            The measured mean charge capture that will be fit by the function
            :func:`optika.sensors.mean_charge_capture`.
        """
        absorption = chemical_substrate.absorption(mcc_measured.inputs)

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

        thickness = fit.x * unit

        return cls(
            thickness=thickness,
            thickness_substrate=thickness_substrate,
            chemical_substrate=chemical_substrate,
            width_pixel=width_pixel,
            mcc_measured=mcc_measured,
        )

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
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            na.shape(self.thickness),
            na.shape(self.thickness_substrate),
            na.shape(self.width_pixel),
            na.shape(self.mcc_measured),
        )
