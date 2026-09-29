import astropy.units as u
import numpy as np
import scipy.special
import named_arrays as na

__all__ = [
    "charge_diffusion",
    "mean_charge_capture",
    "kernel_diffusion",
]


def charge_diffusion(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
    width_max: None | u.Quantity | na.AbstractScalar = None,
    width_depleted: None | u.Quantity | na.AbstractScalar = None,
) -> na.AbstractScalar:
    r"""
    The standard deviation of the charge diffusion in a backilluminated CCD
    given by :cite:t:`Janesick2001`, averaged over the depth at which the
    photons are absorbed.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive layer for the
        incident photon.
    thickness_substrate
        The thickness of the light-sensitive region of the imaging sensor.
    thickness_depletion
        The thickness of the depletion region of the imaging sensor.
    width_max
        The standard deviation of the charge cloud of a photon absorbed at the
        back surface, before it crosses the depletion region.
        If :obj:`None` (the default), the thickness of the field-free region,
        as in the model of :cite:t:`Janesick2001`.
    width_depleted
        The standard deviation acquired by charge drifting across the full
        thickness of the depletion region.
        If :obj:`None` (the default), charge does not spread in the depletion
        region, as in the model of :cite:t:`Janesick2001`.

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
        # for the given wavelengths.
        absorption = si.absorption(wavelength)

        # Compute the charge diffusion
        width_diffusion = optika.sensors.charge_diffusion(
            absorption=absorption,
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

    The standard deviation :math:`\sigma_\text{cd}(x)` of the charge
    diffusion kernel of a photon absorbed at a distance :math:`x` from the
    back surface is given by
    :meth:`optika.sensors.diffusion.JanesickDiffusionModel.width`.
    The `average` variance of the charge diffusion kernel is then
    the weighted average,

    .. math::

        \overline{\sigma}_\text{cd}^2 &= \dfrac{\displaystyle \int_0^{x_s} \left( \sigma_\text{cd}(x) \right)^2 e^{-\alpha x} dx}
                                               {\displaystyle \int_0^{x_s} e^{-\alpha x} dx} \\[1mm]
                                      &= \dfrac{\displaystyle \int_0^{x_{ff}} \sigma_\text{max}^2 \left( 1 - \frac{x}{x_{ff}} \right) e^{-\alpha x} dx
                                                + \int_0^{x_s} \sigma_d^2 \, g(x) \, e^{-\alpha x} dx}
                                               {\displaystyle \int_0^{x_s} e^{-\alpha x} dx} \\[1mm]
                                      &= \dfrac{\dfrac{\sigma_\text{max}^2}{x_{ff}} \left( \alpha x_{ff} + e^{-\alpha x_{ff}} - 1 \right)
                                                + \dfrac{\sigma_d^2}{x_d} \left( \alpha x_d - e^{-\alpha x_{ff}} + e^{-\alpha x_s} \right)}
                                               {\alpha \left( 1 - e^{-\alpha x_s} \right)}

    where :math:`\alpha` is the absorption coefficient of the light-sensitive
    layer, :math:`x_s` is the thickness of the light-sensitive region,
    :math:`x_d` is the thickness of the depletion region,
    :math:`x_{ff} = x_s - x_d` is the thickness of the field-free region,
    :math:`\sigma_\text{max}` is `width_max`,
    :math:`\sigma_d` is `width_depleted`,
    and :math:`g(x)` is the fraction of the depletion region crossed by
    charge created at :math:`x`.
    With :math:`\sigma_\text{max} = x_{ff}` and :math:`\sigma_d = 0`,
    the default, this is the model of :cite:t:`Janesick2001`,

    .. math::

        \overline{\sigma}_\text{cd}^2 = \dfrac{x_{ff} \left( \alpha x_{ff} + e^{-\alpha x_{ff}} - 1 \right)}
                                              {\alpha \left( 1 - e^{-\alpha x_s} \right)}.
    """
    s = thickness_substrate
    d = thickness_depletion
    f = s - d

    if width_max is None:
        width_max = f

    az_s = (absorption * s).to(u.dimensionless_unscaled).value
    az_d = (absorption * d).to(u.dimensionless_unscaled).value
    az_f = az_s - az_d

    # The fraction of the photons entering the sensor which are absorbed
    # in the light-sensitive region.
    absorbed = -np.expm1(-az_s)

    variance = np.square(width_max) * _absorbed_ramp(az_f)

    if width_depleted is not None:
        crossed = -np.expm1(-az_f) + np.exp(-az_f) * _absorbed_ramp(az_d)
        variance = variance + np.square(width_depleted) * crossed

    result = np.sqrt(variance / absorbed).to(u.um)

    return result


def _absorbed_ramp(
    optical_depth: float | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The integral of a weight falling linearly from one to zero across a layer,
    weighted by the probability of a photon being absorbed at each depth.

    For a layer of thickness :math:`L` and an absorption coefficient
    :math:`\alpha`, this is

    .. math::

        \int_0^L \left( 1 - \frac{x}{L} \right) \alpha e^{-\alpha x} dx
            = \frac{\alpha L + e^{-\alpha L} - 1}{\alpha L},

    which goes to zero with the optical depth :math:`\alpha L`,
    so that a layer of zero thickness contributes nothing.

    Parameters
    ----------
    optical_depth
        The optical depth :math:`\alpha L` of the layer.
    """
    where = optical_depth > 0
    optical_depth = np.where(where, optical_depth, 1)
    result = (optical_depth + np.expm1(-optical_depth)) / optical_depth
    return np.where(where, result, 0)


def _probability_same_pixel(
    sigma: float | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The probability that two electrons from the same photon land in the same
    column (or row) of pixels, averaged over the uniformly-distributed
    sub-pixel position of the photon.

    Both electrons start from the same sub-pixel position
    :math:`u \sim \mathcal{U}(-1/2, 1/2)` and are displaced independently by
    :math:`\mathcal{N}(0, \sigma^2)`, so their separation is
    :math:`\delta \sim \mathcal{N}(0, 2 \sigma^2)`, and the probability that
    they are binned into the same pixel is
    :math:`\left\langle (1 - |\delta|)_+ \right\rangle`,
    which evaluates to

    .. math::

        d(\sigma) = \text{erf} \left( \frac{1}{2 \sigma} \right)
            - \frac{2 \sigma}{\sqrt{\pi}} \left( 1 - e^{-1 / 4 \sigma^2} \right).

    Parameters
    ----------
    sigma
        The standard deviation of the charge diffusion kernel in units of the
        pixel width.
    """
    where = sigma > 0
    sigma = np.where(where, sigma, 1)
    result = scipy.special.erf(1 / (2 * sigma)) + (
        2 * sigma / np.sqrt(np.pi) * np.expm1(-1 / (4 * np.square(sigma)))
    )
    return np.where(where, result, 1)


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


def _kernel_1d(
    width_diffusion: u.Quantity,
    width_pixel: u.Quantity,
    index_pixel: int | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The charge diffusion kernel in 1 dimension.
    Designed to be used in an outer product to make a 2D version.

    Parameters
    ----------
    width_diffusion
        The standard deviation of the charge diffusion kernel.
    width_pixel
        The physical size of the pixel.
    index_pixel
        The indices of the pixels to compute,
        relative to the center of the kernel.
    """

    w = width_diffusion
    d = width_pixel
    n = index_pixel

    x = d / w
    x2 = np.square(x)

    c = 1 / (x * np.sqrt(2 * np.pi))

    def g(m: int | na.AbstractScalar) -> na.AbstractScalar:
        return np.exp(-x2 * m / 2)

    def e(m: int | na.AbstractScalar) -> na.AbstractScalar:
        return m * scipy.special.erf(x * m / np.sqrt(2))

    g1 = g(np.square(n - 1))
    g2 = -2 * g(np.square(n))
    g3 = g(np.square(n + 1))

    e1 = e(n - 1) / 2
    e2 = -e(n)
    e3 = e(n + 1) / 2

    result = c * (g1 + g2 + g3) + e1 + e2 + e3

    return result


def kernel_diffusion(
    width_diffusion: u.Quantity,
    width_pixel: u.Quantity,
    axis_x: str,
    axis_y: str,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    The charge diffusion kernel convolved with a pixel and then integrated
    over the extent of each pixel.

    Parameters
    ----------
    width_diffusion
        The standard deviation of the charge diffusion kernel.
        Often computed using :func:`~optika.sensors.charge_diffusion`.
    width_pixel
        The width of a pixel.
    axis_x
        The name of the horizontal axis.
    axis_y
        The name of the vertical axis.

    Examples
    --------

    Plot this diffusion kernel

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the wavelength to compute the charge diffusion kernel for.
        wavelength = 1403 * u.AA

        # Define the width of pixel
        width_pixel = 13 * u.um

        # Load the optical properties of silicon
        si = optika.chemicals.Chemical("Si")

        # Retrieve the absorption coefficient of silicon
        # for the given wavelengths.
        absorption = si.absorption(wavelength)

        # Compute the standard deviation of the charge diffusion kernel
        width_diffusion = optika.sensors.charge_diffusion(
            absorption=absorption,
            thickness_substrate=14 * u.um,
            thickness_depletion=8.7 * u.um,
        )

        # Compute the charge diffusion kernel.
        kernel = optika.sensors.kernel_diffusion(
            width_diffusion=width_diffusion,
            width_pixel=width_pixel,
            axis_x="x",
            axis_y="y",
        )

        # Plot the charge diffusion kernel.
        fig, ax = plt.subplots(
            figsize=(3, 3),
            constrained_layout=True,
        )
        na.plt.pcolormesh(
            kernel.inputs.x,
            kernel.inputs.y,
            C=kernel.outputs,
            facecolors="None",
            edgecolors="black",
        )
        na.plt.text(
            x=kernel.inputs.x,
            y=kernel.inputs.y,
            s=kernel.outputs.to_string_array(format_value="%.3f"),
            color="black",
            ha="center",
            va="center",
        )
        ax.set_xlabel("detector $x$ (pix)")
        ax.set_ylabel("detector $y$ (pix)")
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]);
        ax.set_yticks([-1, 0, 1]);
    """

    index_x = na.linspace(-1, 1, axis=axis_x, num=3)
    index_y = na.linspace(-1, 1, axis=axis_y, num=3)

    kx = _kernel_1d(
        width_diffusion=width_diffusion,
        width_pixel=width_pixel,
        index_pixel=index_x,
    )
    ky = _kernel_1d(
        width_diffusion=width_diffusion,
        width_pixel=width_pixel,
        index_pixel=index_y,
    )

    result = kx * ky

    result = result / result.sum(axis=(axis_x, axis_y))

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(index_x, index_y),
        outputs=result,
    )
