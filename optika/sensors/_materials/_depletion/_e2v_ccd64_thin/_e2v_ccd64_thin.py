import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._depletion import AbstractJanesickDepletionModel

__all__ = [
    "E2VCCD64ThinDepletionModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class E2VCCD64ThinDepletionModel(
    AbstractJanesickDepletionModel,
):
    """
    A model of the depletion region of a "thin" (20 :math:`\Omega`-cm)
    e2v CCD64 imaging sensor, which uses charge diffusion measurements from
    :cite:t:`Stern2004` to estimate the thickness of the depletion region.

    Examples
    --------

    Plot the measured e2v CCD64 measured mean charge capture vs. the fitted
    mean charge capture.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Create a new instance of the e2v CCD64 depletion region model
        depletion = optika.sensors.E2VCCD64ThinDepletionModel()

        # Store the wavelengths at which the MCC was measured
        wavelength_measured = depletion.mean_charge_capture_measured.inputs

        # Store the MCC measurements
        mcc_measured = depletion.mean_charge_capture_measured.outputs

        # Define a grid of wavelengths with which to evaluate the fitted MCC
        wavelength_fit = na.geomspace(1, 10000, axis="wavelength", num=1001) * u.AA
        energy_fit = wavelength_fit.to(u.eV, equivalencies=u.spectral())

        # Evaluate the fitted MCC using the given wavelengths
        mcc_fit = depletion.mean_charge_capture(wavelength_fit)

        # Plot the measured QE vs the fitted QE
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            ax2 = ax.twiny()
            ax2.invert_xaxis()
            na.plt.scatter(
                wavelength_measured,
                mcc_measured,
                ax=ax,
                label="measured",
            )
            na.plt.plot(
                wavelength_fit,
                mcc_fit,
                ax=ax,
                label="fit",
            )
            na.plt.plot(
                energy_fit,
                mcc_fit,
                ax=ax2,
                linestyle="None",
            )
            ax.set_xscale("log")
            ax2.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax2.set_xlabel(f"energy ({ax2.get_xlabel()})")
            ax.set_ylabel("mean charge capture")
            ax.legend()

    The thickness of the depletion region found by the fit is

    .. jupyter-execute::

        depletion.thickness
    """

    @property
    def shape(self) -> dict[str, int]:
        return dict()

    @property
    def chemical_substrate(self) -> optika.chemicals.Chemical:
        return optika.chemicals.Chemical("Si")

    @property
    def thickness_substrate(self) -> u.Quantity:
        return 8 * u.um

    @property
    def width_pixel(self) -> u.Quantity:
        return 16 * u.um

    @property
    def mean_charge_capture_measured(self) -> na.FunctionArray:
        directory = pathlib.Path(__file__).parent

        energy, mcc = np.genfromtxt(
            fname=directory / "_e2v_ccd64_thin_stern2004.csv",
            delimiter=", ",
            unpack=True,
        )
        energy = energy << u.keV
        wavelength = energy.to(u.AA, equivalencies=u.spectral())

        return na.FunctionArray(
            inputs=na.ScalarArray(wavelength, axes="wavelength"),
            outputs=na.ScalarArray(mcc, axes="wavelength"),
        )
