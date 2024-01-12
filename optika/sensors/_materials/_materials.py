import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "energy_bandgap",
    "energy_electron_hole",
    "quantum_yield_ideal",
]

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
