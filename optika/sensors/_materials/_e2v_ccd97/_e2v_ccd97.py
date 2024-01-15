import functools
import pathlib
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
from .._materials import quantum_efficiency_effective
from .._materials import AbstractBackilluminatedCCDMaterial

__all__ = [
    "E2VCCD97Material",
]


class E2VCCD97Material(
    AbstractBackilluminatedCCDMaterial,
):
    """
    A model of the light-sensitive material of an e2v CCD90 sensor based on
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

    @functools.cached_property
    def _quantum_efficiency_fit(self) -> dict[str, float | u.Quantity]:
        qe_measured = self.quantum_efficiency_measured

        unit_thickness_oxide = u.AA
        unit_thickness_implant = u.AA
        unit_thickness_substrate = u.um

        def eqe_rms_difference(x: tuple[float, float, float, float]):
            (
                thickness_oxide,
                thickness_implant,
                thickness_substrate,
                cce_backsurface,
            ) = x
            qe_fit = quantum_efficiency_effective(
                wavelength=qe_measured.inputs,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
                thickness_oxide=thickness_oxide << unit_thickness_oxide,
                thickness_implant=thickness_implant << unit_thickness_implant,
                thickness_substrate=thickness_substrate << unit_thickness_substrate,
                cce_backsurface=cce_backsurface,
            )

            return np.sqrt(np.mean(np.square(qe_measured.outputs - qe_fit))).ndarray

        thickness_oxide_guess = 50 * u.AA
        thickness_implant_guess = 2317 * u.AA
        thickness_substrate_guess = 7 * u.um
        cce_backsurface_guess = 0.21

        fit = scipy.optimize.minimize(
            fun=eqe_rms_difference,
            x0=[
                thickness_oxide_guess.to_value(unit_thickness_oxide),
                thickness_implant_guess.to_value(unit_thickness_implant),
                thickness_substrate_guess.to_value(unit_thickness_substrate),
                cce_backsurface_guess,
            ],
            method="nelder-mead",
        )

        thickness_oxide, thickness_implant, thickness_substrate, cce_backsurface = fit.x
        thickness_oxide = thickness_oxide << unit_thickness_oxide
        thickness_implant = thickness_implant << unit_thickness_implant
        thickness_substrate = thickness_substrate << unit_thickness_substrate

        return dict(
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            cce_backsurface=cce_backsurface,
        )

    @property
    def thickness_oxide(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_oxide"]

    @property
    def thickness_implant(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_implant"]

    @property
    def thickness_substrate(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_substrate"]

    @property
    def cce_backsurface(self) -> float:
        return self._quantum_efficiency_fit["cce_backsurface"]
