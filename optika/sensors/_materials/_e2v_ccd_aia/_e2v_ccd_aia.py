import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
from .._materials import AbstractStern1994BackilluminatedCCDMaterial

__all__ = [
    "E2VCCDAIAMaterial",
]


class E2VCCDAIAMaterial(
    AbstractStern1994BackilluminatedCCDMaterial,
):
    """
    A model of the light-sensitive material of the custom e2v CCD sensors
    on board the Atmospheric Imaging Assembly :cite:p:`Lemen2012` from
    :cite:t:`Boerner2012`

    Examples
    --------

    Plot the measured AIA CCD quantum efficiency vs the fitted
    quantum efficiency calculated using the method of :cite:t:`Stern1994`.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Create a new instance of the e2v CCD97 light-sensitive material
        material_ccd_aia = optika.sensors.E2VCCDAIAMaterial()

        # Store the wavelengths at which the QE was measured
        wavelength_measured = material_ccd_aia.quantum_efficiency_measured.inputs

        # Store the QE measurements
        qe_measured = material_ccd_aia.quantum_efficiency_measured.outputs

        # Define a grid of wavelengths with which to evaluate the fitted QE
        wavelength_fit = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Evaluate the fitted QE using the given wavelengths
        qe_fit = material_ccd_aia.quantum_efficiency_effective(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength_fit,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the measured QE vs the fitted QE
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                wavelength_measured,
                qe_measured,
                label="measured",
            )
            na.plt.plot(
                wavelength_fit,
                qe_fit,
                label="fit",
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({wavelength_fit.unit:latex_inline})")
            ax.set_ylabel("quantum efficiency")
            ax.legend()

    The thickness of the oxide layer found by the fit is

    .. jupyter-execute::

        material_ccd_aia.thickness_oxide

    The thickness of the implant layer found by the fit is

    .. jupyter-execute::

        material_ccd_aia.thickness_implant

    The thickness of the substrate found by the fit is

    .. jupyter-execute::

        material_ccd_aia.thickness_substrate

    And the differential charge collection efficiency at the backsurface
    found by the fit is

    .. jupyter-execute::

        material_ccd_aia.cce_backsurface
    """

    @property
    def quantum_efficiency_measured(self) -> na.FunctionArray:
        directory = pathlib.Path(__file__).parent
        (
            wavelength_1,
            qe_1,
            wavelength_2,
            qe_2,
            wavelength_3,
            qe_3,
            wavelength_4,
            qe_4,
        ) = np.genfromtxt(
            fname=directory / "e2v_ccd_aia_qe_boerner2012.csv",
            skip_header=2,
            delimiter=",",
            unpack=True,
        )
        wavelength = (wavelength_1 + wavelength_2 + wavelength_3 + wavelength_4) / 4
        wavelength = wavelength << u.AA
        qe = (qe_1 + qe_2 + qe_3 + qe_4) / 4
        return na.FunctionArray(
            inputs=na.ScalarArray(wavelength, axes="wavelength"),
            outputs=na.ScalarArray(qe, axes="wavelength"),
        )
