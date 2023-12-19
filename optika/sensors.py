from typing import TypeVar
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "quantum_yield_ideal",
    "AbstractImagingSensor",
    "AbstractCCD",
]


MaterialT = TypeVar("MaterialT", bound=optika.materials.AbstractMaterial)


energy_bandgap = 1.12 * u.eV
"""the bandgap energy of silicon"""

energy_electron_hole = 3.65 * u.eV
"""
the high-energy limit of the energy required to create an electron-hole pair
in silicon at room temperature
"""


def quantum_yield_ideal(
    wavelength: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    Calculate the ideal quantum yield of a silicon detector for a given
    wavelength.

    Parameters
    ----------
    wavelength
        the wavelength of the incident photons

    Notes
    -----
    The quantum yield is the number of electron-hole pairs produced per photon.

    The ideal quantum yield is given in :cite:t:`Janesick2001` as:

    .. math::

        \text{QY}(\epsilon) = \begin{cases}
            0, & \epsilon < E_\text{g}\\
            1, &  E_\text{g} \leq \epsilon < E_\text{e-h} \\
            E_\text{e-h} / \epsilon, & E_\text{e-h} \leq \epsilon,
        \end{cases},

    where :math:`\epsilon` is the energy of the incident photon,
    :math:`E_\text{g} = 1.12\;\text{eV}` is the bandgap energy of silicon,
    and :math:`E_\text{e-h} = 3.65\;\text{eV}` is the energy required to
    generate 1 electron-hole pair in silicon at room temperature.

    Examples
    --------

    Plot the quantum yield vs wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths
        wavelength = na.geomspace(100, 100000, axis="wavelength", num=101) << u.AA

        # Compute the quantum yield
        qy = optika.sensors.quantum_yield_ideal(wavelength)

        # Plot the quantum yield vs wavelength
        fig, ax = plt.subplots()
        na.plt.plot(wavelength, qy, ax=ax);
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("quantum yield");
    """
    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    result = energy / energy_electron_hole
    result = np.where(energy > energy_electron_hole, result, 1)
    result = np.where(energy > energy_bandgap, result, 0)

    return result


class AbstractImagingSensor(
    optika.surfaces.AbstractSurface[
        None,
        MaterialT,
        optika.apertures.RectangularAperture,
        optika.apertures.RectangularAperture,
        None,
    ],
):
    @property
    def sag(self) -> None:
        return None

    @property
    def rulings(self) -> None:
        return None


class AbstractCCD(
    AbstractImagingSensor[MaterialT],
):
    @property
    @abc.abstractmethod
    def quantum_efficiency_effective(self, rays: optika.rays.RayVectorArray):
        r"""
        Compute the effective quantum efficiency (EQE) introduced by :cite:t:`Stern1994`,
        which includes both the recombination rate and the actual quantum efficiency.

        Parameters
        ----------
        rays
            the incident light rays

        Notes
        -----
        Our goal is to recover Equation 11 in :cite:t:`Stern1994`.
        From inspecting Equations 6 and 9 in :cite:t:`Stern1994`,
        we can see that the effective quantum efficiency is:

        .. math::
            :label: eqe

            \text{EQE} = T_\lambda \int_0^\infty \alpha \eta(x) e^{-\alpha x} \; dx

        where :math:`T_\lambda` is the net transmission of photons through the backsurface
        oxide layer (accounting for both absorption and reflections),
        :math:`\alpha` is the absorption coefficient of silicon,
        :math:`x` is the distance from the backsurface,
        and :math:`\eta(x)` is the differential charge collection efficiency (CCE).

        :cite:t:`Stern1994` assumes that the differential CCE takes the following
        linear form,

        .. math::
            :label: differential-cce

            \eta(x) = \begin{cases}
                \eta_0 + (1 - \eta_0) x / W, & x < W \\
                1, & x > W,
            \end{cases}

        where :math:`\eta_0` is the differential CCE at the backsurface,
        and :math:`W` is the width of the implant region.

        Plugging Equation :eq:`differential-cce` into Equation :eq:`eqe` yields

        .. math::

            \text{EQE} &= \alpha T_\lambda \left\{
                            \int_0^W \left[ \eta_0 + \left( \frac{1 - \eta_0}{W} \right) x \right] e^{-\alpha x} \; dx
                            + \int_W^\infty e^{-\alpha x} \; dx
            \right\} \\
            &= \alpha T_\lambda \left\{
                \eta_0 \int_0^W e^{-\alpha x} \; dx
                + \left( \frac{1 - \eta_0}{W} \right) \int_0^W x e^{-\alpha x} \; dx
                + \int_W^\infty e^{-\alpha x} \; dx
            \right\} \\
            &= \alpha T_\lambda \left\{
                -\left[ \frac{\eta_0}{\alpha} e^{-\alpha x} \right|_0^W
                - \left( \frac{1 - \eta_0}{W} \right) \left[ \left( \frac{\alpha x + 1}{\alpha^2} \right) e^{-\alpha x} \right|_0^W
                - \left[ \frac{1}{\alpha} e^{-\alpha x} \right|_W^\infty
            \right\} \\
            &= T_\lambda \left\{
                - \left[ \eta_0 (e^{-\alpha W} - 1) \right]
                - \left( \frac{1 - \eta_0}{\alpha W} \right) \left[ (\alpha W + 1) e^{-\alpha W} - 1 \right]
                - \left[ 0 - e^{-\alpha W} \right]
            \right\} \\
            &= T_\lambda \left\{
                - \eta_0 e^{-\alpha W}
                + \eta_0
                - e^{-\alpha W}
                + \eta_0 e^{-\alpha W}
                + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
                + e^{-\alpha W}
            \right\} \\
            &= T_\lambda \left\{
                \eta_0
                + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
            \right\} \\

        Compute the limit of :math:`W` approaching infinity

        .. math::

            \lim_{W \to \infty} \text{EQE} = T_\lambda \eta_0

        Compute the limit of :math:`W` approaching zero

        .. math::

            \lim_{W \to 0} \text{EQE} &= T_\lambda \left\{ \eta_0 + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - (1 - \alpha W + \alpha^2 W^2 + ...))  \right\} \\
                                        &= T_\lambda

        Compute the limit of :math:`\eta_0` approaching 1

        .. math::

            \lim_{\eta_0 \to 1} = T_\lambda

        """
