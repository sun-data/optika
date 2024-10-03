import astropy.units as u
import named_arrays as na
from .._depletion import E2VCCD64ThinDepletionModel
from .._materials import AbstractStern1994BackilluminatedCCDMaterial

__all__ = [
    "TektronixTK512CBMaterial",
]


class TektronixTK512CBMaterial(
    AbstractStern1994BackilluminatedCCDMaterial,
):
    """
    A model of the light-sensitive material of a Tektronix TK512CB sensor based on
    measurements by :cite:t:`Stern1994`.

    This model uses the :class:`optika.sensors.E2VCCD64ThinDepletionModel`
    to represent the depletion region.

    Examples
    --------

    Plot the measured TK512CB quantum efficiency vs the fitted
    quantum efficiency calculated using the method of :cite:t:`Stern1994`.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Create a new instance of the e2v CCD97 light-sensitive material
        material = optika.sensors.TektronixTK512CBMaterial()

        # Store the wavelengths at which the QE was measured
        wavelength_measured = material.quantum_efficiency_measured.inputs

        # Store the QE measurements
        qe_measured = material.quantum_efficiency_measured.outputs

        # Define a grid of wavelengths with which to evaluate the fitted QE
        wavelength_fit = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Evaluate the fitted QE using the given wavelengths
        qe_fit = material.quantum_efficiency_effective(
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

    @property
    def quantum_efficiency_measured(self) -> na.FunctionArray:
        wavelength = [
            13.3,
            23.6,
            44.7,
            67.6,
            114.0,
            135.5,
            171.4,
            256.0,
            303.8,
            461.0,
            584.0,
            736.0,
            1215.5,
            2537.0,
            3650.0,
            4050.0,
        ] * u.AA
        qe = [
            0.91,
            0.80,
            0.48,
            0.32,
            0.42,
            0.86,
            0.82,
            0.60,
            0.58,
            0.53,
            0.30,
            0.085,
            0.055,
            0.06,
            0.09,
            0.29,
        ] * u.dimensionless_unscaled
        return na.FunctionArray(
            inputs=na.ScalarArray(wavelength, axes="wavelength"),
            outputs=na.ScalarArray(qe, axes="wavelength"),
        )

    @property
    def thickness_substrate(self) -> u.Quantity:
        return 7 * u.um

    @property
    def depletion(self) -> E2VCCD64ThinDepletionModel:
        return E2VCCD64ThinDepletionModel()

    @property
    def shape(self) -> dict[str, int]:
        return dict()
