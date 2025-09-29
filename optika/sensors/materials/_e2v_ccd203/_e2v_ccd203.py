import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..depletion import e2v_ccd64_thick
from .._materials import BackIlluminatedSiliconSensorMaterial

__all__ = [
    "e2v_ccd203",
]


@optika.memory.cache
def _e2v_ccd203() -> BackIlluminatedSiliconSensorMaterial:
    """Cached version of :func:`e2v_ccd203` which does not depend on temperature."""

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
        fname=directory / "e2v_ccd203_qe_boerner2012.csv",
        skip_header=2,
        delimiter=",",
        unpack=True,
    )

    wavelength = (wavelength_1 + wavelength_2 + wavelength_3 + wavelength_4) / 4
    wavelength = wavelength << u.AA

    qe = (qe_1 + qe_2 + qe_3 + qe_4) / 4

    qe = na.FunctionArray(
        inputs=na.ScalarArray(wavelength, axes="wavelength"),
        outputs=na.ScalarArray(qe, axes="wavelength"),
    )

    return BackIlluminatedSiliconSensorMaterial.fit_eqe(
        thickness_substrate=16 * u.um,
        depletion=e2v_ccd64_thick(),
        eqe_measured=qe,
    )


def e2v_ccd203(
    temperature: u.Quantity | na.AbstractScalar = 300 * u.K,
) -> BackIlluminatedSiliconSensorMaterial:
    """
    A model of the light-sensitive material of the custom e2v CCD sensors
    on board the Atmospheric Imaging Assembly :cite:p:`Lemen2012` from
    :cite:t:`Boerner2012`

    This model uses :func:`~optika.sensors.materials.depletion.e2v_ccd64_thick`
    to represent the depletion region.

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
        material = optika.sensors.materials.e2v_ccd203()

        # Store the wavelengths at which the QE was measured
        wavelength_measured = material.eqe_measured.inputs

        # Store the QE measurements
        eqe_measured = material.eqe_measured.outputs

        # Define a grid of wavelengths with which to evaluate the fitted QE
        wavelength_fit = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Evaluate the fitted QE using the given wavelengths
        eqe_fit = material.quantum_efficiency_effective(
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
                eqe_measured,
                label="measured",
            )
            na.plt.plot(
                wavelength_fit,
                eqe_fit,
                label="fit",
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({wavelength_fit.unit:latex_inline})")
            ax.set_ylabel("quantum efficiency")
            ax.legend()

    The thickness of the oxide layer found by the fit is

    .. jupyter-execute::

        material.thickness_oxide

    The thickness of the implant layer found by the fit is

    .. jupyter-execute::

        material.thickness_implant

    The thickness of the substrate is

    .. jupyter-execute::

        material.thickness_substrate

    The differential charge collection efficiency at the backsurface
    found by the fit is

    .. jupyter-execute::

        material.cce_backsurface

    And the roughness of the substrate found by the fit is

    .. jupyter-execute::

        material.roughness_substrate

    |

    Plot the width of the charge diffusion kernel for this sensor as a function
    of wavelength.

    .. jupyter-execute::

        # Compute the width of the charge diffusion kernel
        # for each wavelength.
        width = material.width_charge_diffusion(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength_fit,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the results
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                wavelength_fit,
                width,
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"width ({ax.get_ylabel()})")
    """
    result = _e2v_ccd203()
    result.temperature = temperature
    return result
