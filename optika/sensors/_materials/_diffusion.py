import astropy.units as u
import numpy as np

import named_arrays as na

__all__ = [
    "charge_diffusion",
]


def charge_diffusion(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The standard deviation of the charge diffusion in a backilluminated CCD
    given by :cite:t:`Janesick2001`.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive layer for the
        incident photon.
    thickness_implant
        The thickness of the partial-charge collection region of the imaging
        sensor.
    thickness_substrate
        The thickness of the light-sensitive region of the imaging sensor.
    thickness_depletion
        The thickness of the depletion region of the imaging sensor.

    Examples
    --------

    Plot the width of the charge diffusion kernel as a function of wavelength
    and energy for the sensor parameters in :cite:t:`Heymes2020`.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(1, 10000, axis="wavelength", num=1001) * u.AA

        # Convert the grid to energies as well
        energy = wavelength.to(u.eV, equivalencies=u.spectral())

        # Load the optical properties of silicon
        si = optika.chemicals.Chemical("Si")

        # Retrieve the absorption coefficient of silicon
        # for the given wavelenghts.
        absorption = si.absorption(wavelength)

        # Compute the charge diffusion
        width_diffusion = optika.sensors.charge_diffusion(
            absorption=absorption,
            thickness_implant=40 * u.nm,
            thickness_substrate=14 * u.um,
            thickness_depletion=2.4 * u.um,
        )

        # Plot the charge diffusion as a function
        # of wavelength and energy
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            ax2 = ax.twiny()
            ax2.invert_xaxis()
            na.plt.plot(
                wavelength,
                width_diffusion,
                ax=ax,
            )
            na.plt.plot(
                energy,
                width_diffusion,
                ax=ax2,
                linestyle="None",
            )
            ax.set_xscale("log")
            ax2.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"charge diffusion ({ax.get_ylabel()})")

    Notes
    -----

    The standard deviation of the charge diffusion is given by
    :cite:t:`Janesick2001` as

    .. math::

        \sigma_d = x_{ff} \left( 1 - \frac{L_A}{x_{ff}} \right)^{1/2}

    where

    .. math::

        L_A &= \frac{\int_0^{x_s} x e^{-\alpha x} dx}{\int_0^\infty dx}
            &= \frac{1 - (\alpha x_s + 1) e^{-\alpha x_s}}{\alpha^2 x_s}

    is the average distance from the back surface at which to photon is absorbed,
    :math:`\alpha` is the absorption coefficient of the light-sensitive layer,
    :math:`x_s` is the total thickness of the light-sensitive layer,

    .. math::

        x_{ff} = x_s - x_p - x_d,

    is the thickness of the field-free region of the sensor,
    :math:`x_p` is the thickness of the partial-charge collection region,
    and :math:`x_d` is the thickness of the depletion region.
    """
    d = thickness_substrate

    x_ff = d - thickness_implant - thickness_depletion

    a = absorption

    depth_avg = (1 - (a * d + 1) * np.exp(-a * d)) / (np.square(a) * d)

    result = x_ff * np.sqrt(1 - depth_avg / x_ff)

    return result
