import pathlib
from typing import Literal
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._measurements import MeanChargeCapture
from .._models import SlabDiffusionModel

__all__ = [
    "mcc_stern2004",
    "e2v_ccd64_thick",
    "e2v_ccd64_thin",
]

_thickness_substrate = dict(
    thick=15 * u.um,
    thin=8 * u.um,
)
"""
The thickness of the light-sensitive region of each e2v CCD64 measured by
:cite:t:`Stern2004`.
"""

_width_pixel = 16 * u.um
"""The width of a pixel of the e2v CCD64."""


def mcc_stern2004(
    kind: Literal["thick", "thin"],
) -> MeanChargeCapture:
    r"""
    The mean charge capture of an e2v CCD64 measured by :cite:t:`Stern2004`,
    as a function of the vacuum wavelength of the incident photons,
    together with the CCD it was measured on.

    The CCD64 has 16 micron pixels, and was made in two versions:
    a "thick" one of 100 :math:`\Omega`-cm silicon with a 15 micron
    light-sensitive region,
    and a "thin" one of 20 :math:`\Omega`-cm silicon with an 8 micron
    light-sensitive region.
    The result carries both as
    :attr:`~optika.sensors.diffusion.MeanChargeCapture.thickness_substrate`
    and
    :attr:`~optika.sensors.diffusion.MeanChargeCapture.width_pixel`.

    Parameters
    ----------
    kind
        Which version of the CCD64, ``"thick"`` or ``"thin"``.
    """
    path = pathlib.Path(__file__).parent / f"_e2v_ccd64_{kind}.csv"

    energy, mcc = np.genfromtxt(
        fname=path,
        delimiter=", ",
        unpack=True,
    )
    energy = energy << u.keV
    wavelength = energy.to(u.AA, equivalencies=u.spectral())

    return MeanChargeCapture(
        inputs=na.ScalarArray(wavelength, axes="wavelength"),
        outputs=na.ScalarArray(mcc, axes="wavelength"),
        thickness_substrate=_thickness_substrate[kind],
        width_pixel=_width_pixel,
    )


def _e2v_ccd64(
    kind: Literal["thick", "thin"],
) -> SlabDiffusionModel:
    """The model of the field-free region fitted to the measurement of one CCD64."""
    return SlabDiffusionModel(
        thickness_depletion=0 * u.um,
    ).fit_mean_charge_capture(
        mcc_measured=mcc_stern2004(kind),
    )


@optika.memory.cache
def e2v_ccd64_thick() -> SlabDiffusionModel:
    r"""
    The model of charge diffusing across the field-free region,
    :class:`~optika.sensors.diffusion.SlabDiffusionModel`,
    for a "thick" (100 :math:`\Omega`-cm) e2v CCD64,
    with the thickness of its depletion region fitted to the mean charge
    capture measured by :cite:t:`Stern2004`, :func:`mcc_stern2004`.

    Examples
    --------

    Plot the measured mean charge capture against the fitted one.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Load the model and the measurement it was fitted to
        model = optika.sensors.diffusion.e2v_ccd64_thick()
        mcc_measured = optika.sensors.diffusion.mcc_stern2004("thick")

        # Evaluate the fitted mean charge capture over a grid of wavelengths
        wavelength = na.geomspace(1, 10000, axis="wavelength", num=1001) * u.AA
        # on the CCD the measurement was made on
        mcc_fit = model.mean_charge_capture(
            absorption=optika.chemicals.Chemical("Si").absorption(wavelength),
            thickness_substrate=mcc_measured.thickness_substrate,
            width_pixel=mcc_measured.width_pixel,
        )

        # Plot the measurement against the fit
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                mcc_measured.inputs,
                mcc_measured.outputs,
                ax=ax,
                label="measured",
            )
            na.plt.plot(wavelength, mcc_fit, ax=ax, label="fit")
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel("mean charge capture")
            ax.legend()

    The thickness of the depletion region found by the fit is

    .. jupyter-execute::

        model.thickness_depletion

    The model of :cite:t:`Janesick2001` fitted to the same measurement,
    which fits it a little worse, is

    .. jupyter-execute::

        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=0 * u.um,
        ).fit_mean_charge_capture(mcc_measured)
    """
    return _e2v_ccd64("thick")


@optika.memory.cache
def e2v_ccd64_thin() -> SlabDiffusionModel:
    r"""
    The model of charge diffusing across the field-free region,
    :class:`~optika.sensors.diffusion.SlabDiffusionModel`,
    for a "thin" (20 :math:`\Omega`-cm) e2v CCD64,
    with the thickness of its depletion region fitted to the mean charge
    capture measured by :cite:t:`Stern2004`, :func:`mcc_stern2004`.

    Examples
    --------

    The thickness of the depletion region found by the fit is

    .. jupyter-execute::

        import astropy.units as u
        import optika

        optika.sensors.diffusion.e2v_ccd64_thin().thickness_depletion

    The model of :cite:t:`Janesick2001` fitted to the same measurement,
    which fits it a little worse, is

    .. jupyter-execute::

        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=0 * u.um,
        ).fit_mean_charge_capture(
            mcc_measured=optika.sensors.diffusion.mcc_stern2004("thin"),
        )
    """
    return _e2v_ccd64("thin")
