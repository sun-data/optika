import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
from .._materials import AbstractStern1994BackilluminatedCCDMaterial

__all__ = [
    "E2VCCD97Material",
]


class E2VCCD97Material(
    AbstractStern1994BackilluminatedCCDMaterial,
):
    """
    A model of the light-sensitive material of an e2v CCD97 sensor based on
    measurements by :cite:t:`Moody2017`.

    Examples
    --------

    Plot the measured E2VCCD97 quantum efficiency vs the fitted
    quantum efficiency calculated using the method of :cite:t:`Stern1994`.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Create a new instance of the e2v CCD97 light-sensitive material
        material_ccd97 = optika.sensors.E2VCCD97Material()

        # Store the wavelengths at which the QE was measured
        wavelength_measured = material_ccd97.quantum_efficiency_measured.inputs

        # Store the QE measurements
        qe_measured = material_ccd97.quantum_efficiency_measured.outputs

        # Define a grid of wavelengths with which to evaluate the fitted QE
        wavelength_fit = na.geomspace(5, 10000, axis="wavelength", num=1001) * u.AA

        # Evaluate the fitted QE using the given wavelengths
        qe_fit = material_ccd97.quantum_efficiency_effective(
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

        material_ccd97.thickness_oxide

    The thickness of the implant layer found by the fit is

    .. jupyter-execute::

        material_ccd97.thickness_implant

    The thickness of the substrate found by the fit is

    .. jupyter-execute::

        material_ccd97.thickness_substrate

    And the differential charge collection efficiency at the backsurface
    found by the fit is

    .. jupyter-execute::

        material_ccd97.cce_backsurface

    Plot the effective quantum efficiency of the fit to this data vs. the fit
    to the data in :class:`optika.sensors.E2VCCDAIAMaterial`

    .. jupyter-execute::

        material_ccd_aia = optika.sensors.E2VCCDAIAMaterial()

        qe_fit_aia = material_ccd_aia.quantum_efficiency_effective(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength_fit,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                wavelength_measured,
                qe_measured,
                label="Moody 2017 measured",
            )
            na.plt.plot(
                wavelength_fit,
                qe_fit,
                label="Moody 2017 fit",
            )
            na.plt.scatter(
                material_ccd_aia.quantum_efficiency_measured.inputs,
                material_ccd_aia.quantum_efficiency_measured.outputs,
                label="Boerner 2012 measured",
            )
            na.plt.plot(
                wavelength_fit,
                qe_fit_aia,
                label="Boerner 2012 fit",
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({wavelength_fit.unit:latex_inline})")
            ax.set_ylabel("quantum efficiency")
            ax.legend()
    """

    @property
    def quantum_efficiency_measured(self) -> na.FunctionArray:
        directory = pathlib.Path(__file__).parent
        energy, qe = np.genfromtxt(
            fname=directory / "e2v_ccd97_qe_moody2017.csv",
            delimiter=", ",
            unpack=True,
        )
        energy = energy << u.eV
        wavelength = energy.to(u.AA, equivalencies=u.spectral())
        return na.FunctionArray(
            inputs=na.ScalarArray(wavelength, axes="wavelength"),
            outputs=na.ScalarArray(qe, axes="wavelength"),
        )
