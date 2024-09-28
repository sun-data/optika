import astropy.units as u
import numpy as np
import scipy.special
import named_arrays as na

__all__ = [
    "charge_diffusion",
    "mean_charge_capture",
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

        L_A &= \frac{\int_0^{x_s} x e^{-\alpha x} dx}{\int_0^{x_s} dx} \\
            &= \frac{1 - (\alpha x_s + 1) e^{-\alpha x_s}}{\alpha^2 x_s}

    is the average distance from the back surface at which to photon is absorbed,
    :math:`\alpha` is the absorption coefficient of the light-sensitive layer,
    :math:`x_s` is the total thickness of the light-sensitive layer,

    .. math::

        x_{ff} = x_s - x_p - x_d

    is the thickness of the field-free region of the sensor,
    :math:`x_p` is the thickness of the partial-charge collection region,
    and :math:`x_d` is the thickness of the depletion region.
    """
    d = thickness_substrate

    x_ff = d - thickness_implant - thickness_depletion

    a = absorption

    depth_avg = (1 - (a * d + 1) * np.exp(-a * d)) / (np.square(a) * d)

    result = x_ff * np.sqrt(1 - depth_avg / x_ff)

    return np.nan_to_num(result)


def mean_charge_capture(
    width_diffusion: u.Quantity | na.AbstractScalar,
    width_pixel: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    A function to compute the mean charge capture :cite:p:`Stern2004`,
    the fraction of charge from each photon event retained in the central pixel.

    Parameters
    ----------
    width_diffusion
        The standard deviation of the charge diffusion kernel.
    width_pixel
        The width of a pixel on the sensor.

    Examples
    --------

    Plot the mean charge capture as a function of wavelength
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

        # Compute the mean charge capture
        mcc = optika.sensors.mean_charge_capture(
            width_diffusion=width_diffusion,
            width_pixel=16 * u.um,
        )

        # Plot the mean charge capture as a function
        # of wavelength and energy
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            ax2 = ax.twiny()
            ax2.invert_xaxis()
            na.plt.plot(
                wavelength,
                mcc,
                ax=ax,
            )
            na.plt.plot(
                energy,
                mcc,
                ax=ax2,
                linestyle="None",
            )
            ax.set_xscale("log")
            ax2.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax2.set_xlabel(f"energy ({ax2.get_xlabel()})")
            ax.set_ylabel(f"mean charge capture")

    Notes
    -----
    Naively, the mean charge capture (MCC) is the integral of the charge
    diffusion kernel over the extent of a pixel.
    However, since a photon can strike anywhere within the central pixel,
    the charge diffusion kernel should be convolved with a rectangle function
    the width of a pixel before integrating.
    So, our definition for the MCC is

    .. math::

        P_\text{MCC} = \left\{ \frac{1}{d} \int_{-d/2}^{d/2} \left[ K(x') * \Pi \left( \frac{x'}{d} \right) \right](x) \, dx \right\}^2,

    where :math:`K(x)` is the charge diffusion kernel,
    :math:`\Pi(x)` is the `rectangle function <https://en.wikipedia.org/wiki/Rectangular_function>`_,
    and :math:`d` is the width of a pixel.
    If we assume that the charge diffusion kernel is a Gaussian with standard
    deviation :math:`\sigma`,

    .. math::

        K(x) = \frac{1}{\sqrt{2\pi} \sigma} \exp \left( -\frac{x^2}{2 \sigma^2} \right),

    then we can analytically solve for the MCC,

    .. math::

        P_\text{MCC} &= \left\{ \frac{1}{2d} \int_{-d/2}^{d/2} \left[ \text{erf} \left( \frac{d - 2x}{2 \sqrt{2} \sigma} \right) + \text{erf} \left( \frac{d + 2x}{2 \sqrt{2} \sigma} \right) \right] dx \right\}^2 \\
                     &= \left\{ \sqrt{\frac{2}{\pi}} \frac{\sigma}{d} \left[ \exp \left( -\frac{d^2}{2 \sigma^2} \right) - 1 \right] + \text{erf} \left( \frac{d}{\sqrt{2} \sigma} \right) \right\}^2,

    where :math:`\text{erf}(x)` is the `error function <https://en.wikipedia.org/wiki/Error_function>`_.
    """
    a = width_pixel / width_diffusion

    t1 = np.sqrt(2 / np.pi) * (np.exp(-np.square(a) / 2) - 1) / a
    t2 = scipy.special.erf(a / np.sqrt(2))

    result = np.square(t1 + t2)

    return result.to(u.dimensionless_unscaled)
