import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "MeanChargeCaptureFunctionArray",
]


@dataclasses.dataclass(eq=False, repr=False, kw_only=True)
class MeanChargeCaptureFunctionArray(
    na.FunctionArray[na.AbstractScalar, na.AbstractScalar],
):
    """
    A measured mean charge capture as a function of the vacuum wavelength of
    the incident photons, together with the sensor it was measured on.

    The mean charge capture depends on the thickness of the light-sensitive
    region and on the width of a pixel as well as on how far the charge
    spreads, so a measurement carries the sensor it was made on.
    :meth:`~optika.sensors.diffusion.AbstractDiffusionModel.fit_mean_charge_capture`
    reads them from it, and they are needed to compare any model against it.
    """

    thickness_substrate: u.Quantity | na.AbstractScalar
    """The thickness of the light-sensitive region of the sensor that was measured."""

    width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
    """
    The width of a pixel of the sensor that was measured.
    A scalar gives square pixels; a
    :class:`named_arrays.AbstractCartesian2dVectorArray` gives rectangular
    pixels.
    """

    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si"
    """
    The material of the light-sensitive region of the sensor that was
    measured, which gives its absorption coefficient.
    """
