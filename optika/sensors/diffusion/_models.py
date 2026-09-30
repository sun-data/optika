import abc
import dataclasses
from typing_extensions import Self
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
import named_arrays as na
import optika
from ._gaussian import (
    _width_average,
    _ratio,
    _probability_same_pixel,
    _capture,
    _kernel_1d,
)

__all__ = [
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
]


def _pixel_vector(
    width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
) -> na.AbstractCartesian2dVectorArray:
    """A pixel width as a vector, square if a scalar is given."""
    if not isinstance(width_pixel, na.AbstractCartesian2dVectorArray):
        width_pixel = na.Cartesian2dVectorArray(width_pixel, width_pixel)
    return width_pixel


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDiffusionModel(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    """
    An arbitrary model of the lateral diffusion of charge in a
    back-illuminated sensor.

    Charge created in the field-free region between the back surface and
    the depletion region diffuses until it reaches the depletion region,
    and then drifts to the gates.
    Each model gives the charge cloud that arrives at the gates as a function
    of the depth at which the charge was created,
    and the quantities that follow from it, either at a single depth or
    averaged over the depths at which photons with a given absorption
    coefficient are absorbed.

    The thickness of the light-sensitive substrate and the width of a pixel
    are properties of the sensor rather than of the model,
    so the methods take them as arguments.

    Every concrete model has a field named ``thickness_depletion``,
    which :meth:`fit_mean_charge_capture` fits.
    """

    @property
    @abc.abstractmethod
    def thickness_depletion(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of the depletion region of the sensor."""

    @abc.abstractmethod
    def width(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The standard deviation of the charge cloud along one axis for charge
        created at a given depth.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            charge was created, between zero and `thickness_substrate`.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """

    @abc.abstractmethod
    def cdf(
        self,
        position: u.Quantity | na.AbstractScalar,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The fraction of the charge created at a given depth which arrives at
        the gates at less than a given position along one axis,
        relative to where the charge was created.

        This is the profile of the charge cloud along one axis,
        integrated over the other axis, so the fraction collected in a column
        of pixels is a difference of this function at the edges of the column.

        Parameters
        ----------
        position
            The position along one axis relative to where the charge was
            created.
        depth
            The distance from the back surface of the sensor at which the
            charge was created, between zero and `thickness_substrate`.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """

    @abc.abstractmethod
    def probability_same_pixel(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        """
        The probability that two electrons created at a given depth by the
        same photon are collected in the same pixel, averaged over the
        position of the photon within its pixel.

        This is the factor by which charge diffusion reduces the
        photon-correlated part of the noise of an image;
        see :func:`optika.sensors.vmr_signal`.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            electrons were created.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
            A scalar gives square pixels; a
            :class:`named_arrays.AbstractCartesian2dVectorArray` gives
            rectangular pixels.
            A pixel of zero width turns off the effect of diffusion,
            so that the result is one.
        """

    @abc.abstractmethod
    def width_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The square root of the variance of the charge cloud along one axis
        averaged over the depth at which photons are absorbed.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons, per unit of perpendicular depth.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """

    @abc.abstractmethod
    def kernel(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: int = 3,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        """
        The fraction of the charge from each photon collected in the pixel it
        was absorbed in and in each of the pixels around it, averaged over
        the position of the photon within its pixel and over the depth at
        which it was absorbed.

        The fractions are normalized to sum to one over the kernel,
        so that convolving an image with it conserves charge.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons, per unit of perpendicular depth.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
            A scalar gives square pixels; a
            :class:`named_arrays.AbstractCartesian2dVectorArray` gives
            rectangular pixels.
        axis_x
            The name of the horizontal axis of the kernel.
        axis_y
            The name of the vertical axis of the kernel.
        num
            The number of pixels along each axis of the kernel,
            which must be odd so that the kernel is centered on the pixel the
            photon was absorbed in.
        """

    @abc.abstractmethod
    def mean_charge_capture(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        """
        The fraction of the charge from each photon collected in the pixel it
        was absorbed in, averaged over the position of the photon within its
        pixel and over the depth at which it was absorbed :cite:p:`Stern2004`.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons, per unit of perpendicular depth.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
            A scalar gives square pixels; a
            :class:`named_arrays.AbstractCartesian2dVectorArray` gives
            rectangular pixels.
        """

    @abc.abstractmethod
    def _parameters_monte_carlo(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> dict[str, u.Quantity | na.AbstractScalar]:
        """
        The parameters of this model in the form the Monte Carlo simulation of
        :func:`optika.sensors.electrons_measured` takes them.

        Parameters
        ----------
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """

    def fit_mean_charge_capture(
        self,
        mcc_measured: na.AbstractFunctionArray,
        thickness_substrate: u.Quantity,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
        chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    ) -> Self:
        """
        A copy of this model with the thickness of its depletion region fitted
        to a measured mean charge capture, holding its other parameters fixed.

        The fit minimizes the root-mean-square difference between the
        measured and the modeled mean charge capture,
        with the thickness of the depletion region between zero and
        `thickness_substrate`.

        Parameters
        ----------
        mcc_measured
            The measured mean charge capture as a function of the vacuum
            wavelength of the incident photons.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor that was
            measured.
        width_pixel
            The width of a pixel of the sensor that was measured.
        chemical_substrate
            The material of the light-sensitive region, which gives its
            absorption coefficient.
        """
        if isinstance(chemical_substrate, str):
            chemical_substrate = optika.chemicals.Chemical(chemical_substrate)

        absorption = chemical_substrate.absorption(mcc_measured.inputs)

        unit = u.um

        def objective(thickness_depletion: float) -> float:
            model = self.replace(thickness_depletion=thickness_depletion * unit)
            mcc = model.mean_charge_capture(
                absorption=absorption,
                thickness_substrate=thickness_substrate,
                width_pixel=width_pixel,
            )
            diff = mcc_measured.outputs - mcc
            return np.sqrt(np.mean(np.square(diff))).ndarray

        fit = scipy.optimize.minimize_scalar(
            fun=objective,
            bounds=(0, thickness_substrate.to_value(unit)),
        )

        return self.replace(thickness_depletion=fit.x * unit)


@dataclasses.dataclass(eq=False, repr=False)
class JanesickDiffusionModel(
    AbstractDiffusionModel,
):
    r"""
    A Gaussian charge cloud whose width depends on the depth at which the
    charge was created, as given by :cite:t:`Janesick2001`, with an optional
    spread acquired in the depletion region.

    Examples
    --------

    Plot the width of the charge cloud against the depth at which the charge
    was created, with and without a spread in the depletion region.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the thickness of the light-sensitive region of the sensor
        thickness_substrate = 14 * u.um

        # Define the model of Janesick (2001),
        # and the same model with a spread in the depletion region
        janesick = optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
        )
        depleted = janesick.replace(width_depletion=0.8 * u.um)

        # Define a grid of depths through the light-sensitive region
        depth = na.linspace(0, thickness_substrate, axis="depth", num=1001)

        # Plot the width of the charge cloud against depth
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                depth,
                janesick.width(depth, thickness_substrate),
                ax=ax,
                label="Janesick (2001)",
            )
            na.plt.plot(
                depth,
                depleted.width(depth, thickness_substrate),
                ax=ax,
                label="with a spread in the depletion region",
            )
            ax.axvline(
                thickness_substrate - janesick.thickness_depletion,
                color="gray",
                linestyle="--",
            )
            ax.set_xlabel(f"depth ({ax.get_xlabel()})")
            ax.set_ylabel(f"charge diffusion ({ax.get_ylabel()})")
            ax.legend()

    Plot the average width of the charge cloud, and the mean charge capture
    of a 16 micron pixel, as a function of wavelength.

    .. jupyter-execute::

        # Define a grid of wavelengths
        wavelength = na.geomspace(1, 10000, axis="wavelength", num=1001) * u.AA

        # Retrieve the absorption coefficient of silicon
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

        # Compute the average width and the mean charge capture
        width = janesick.width_average(absorption, thickness_substrate)
        mcc = janesick.mean_charge_capture(
            absorption=absorption,
            thickness_substrate=thickness_substrate,
            width_pixel=16 * u.um,
        )

        # Plot both against wavelength
        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True)
            na.plt.plot(wavelength, width, ax=axs[0])
            na.plt.plot(wavelength, mcc, ax=axs[1])
            axs[1].set_xscale("log")
            axs[1].set_xlabel(f"wavelength ({axs[1].get_xlabel()})")
            axs[0].set_ylabel(f"average width ({axs[0].get_ylabel()})")
            axs[1].set_ylabel("mean charge capture")

    Plot the kernel of a 13 micron pixel for photons of 1403 angstroms.

    .. jupyter-execute::

        # Compute the kernel
        kernel = janesick.kernel(
            absorption=optika.chemicals.Chemical("Si").absorption(1403 * u.AA),
            thickness_substrate=thickness_substrate,
            width_pixel=13 * u.um,
            axis_x="x",
            axis_y="y",
        )

        # Plot the kernel
        fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
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

    Notes
    -----

    :cite:t:`Janesick2001` gives the standard deviation of the charge cloud
    as

    .. math::

        \sigma_\text{ff}(x) = \begin{cases}
            x_{ff} \sqrt{1 - \frac{x}{x_{ff}}}, & 0 < x < x_{ff} \\
            0, & x_{ff} < x < x_s
        \end{cases}

    where :math:`x` is the distance from the back surface at which the charge
    was created,
    :math:`x_{ff} = x_s - x_d` is the thickness of the field-free region,
    :math:`x_s` is the thickness of the light-sensitive region,
    and :math:`x_d` is the thickness of the depletion region.
    This is the spread of charge diffusing through the field-free region
    until it reaches the edge of the depletion region.

    This model generalizes it in two ways.
    First, the width at the back surface, :math:`\sigma_\text{bs}`,
    may differ from the thickness of the field-free region,
    so that a measured width and a measured thickness can both be used.
    Second, charge that reaches the depletion region still diffuses as it
    drifts to the gates.
    In a uniform drift field the variance acquired grows linearly with the
    distance drifted, so the charge acquires a variance :math:`\sigma_d^2`
    crossing the full depletion region, and a proportional part of it
    if it was created inside that region.
    The total variance is

    .. math::

        \sigma^2(x) = \sigma_\text{bs}^2 \left( 1 - \frac{x}{x_{ff}} \right)_+
                      + \sigma_d^2 \, g(x),

    where :math:`(\cdot)_+` is zero for negative arguments and

    .. math::

        g(x) = \min \left( \frac{x_s - x}{x_d}, 1 \right)

    is the fraction of the depletion region crossed by charge created at
    :math:`x`.

    The variance averaged over the depth at which photons are absorbed is

    .. math::

        \overline{\sigma}^2 &= \dfrac{\displaystyle \int_0^{x_s} \sigma^2(x) \, e^{-\alpha x} dx}
                                     {\displaystyle \int_0^{x_s} e^{-\alpha x} dx} \\[1mm]
                            &= \dfrac{\dfrac{\sigma_\text{bs}^2}{x_{ff}} \left( \alpha x_{ff} + e^{-\alpha x_{ff}} - 1 \right)
                                      + \dfrac{\sigma_d^2}{x_d} \left( \alpha x_d - e^{-\alpha x_{ff}} + e^{-\alpha x_s} \right)}
                                     {\alpha \left( 1 - e^{-\alpha x_s} \right)},

    where :math:`\alpha` is the absorption coefficient,
    which reduces to the result of :cite:t:`Janesick2001` for
    :math:`\sigma_\text{bs} = x_{ff}` and :math:`\sigma_d = 0`.

    The kernel and the mean charge capture are those of a Gaussian whose
    variance is :math:`\overline{\sigma}^2`.
    Since a photon can strike anywhere within its pixel, the Gaussian is
    convolved with a rectangle function the width of a pixel before it is
    integrated over each pixel, so that the mean charge capture is

    .. math::

        P_\text{MCC} = \prod_{i \in \{x, y\}} \left\{
            \sqrt{\frac{2}{\pi}} \frac{\overline{\sigma}}{d_i}
            \left[ \exp \left( -\frac{d_i^2}{2 \overline{\sigma}^2} \right) - 1 \right]
            + \text{erf} \left( \frac{d_i}{\sqrt{2} \overline{\sigma}} \right)
        \right\},

    where :math:`d_i` is the width of a pixel along each axis.
    """

    thickness_depletion: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The thickness of the depletion region of the sensor."""

    width_backsurface: None | u.Quantity | na.AbstractScalar = None
    """
    The standard deviation of the charge cloud of a photon absorbed at the
    back surface, before it crosses the depletion region.

    If :obj:`None` (the default), the thickness of the field-free region,
    as in the model of :cite:t:`Janesick2001`.
    """

    width_depletion: None | u.Quantity | na.AbstractScalar = None
    """
    The standard deviation acquired by charge drifting across the full
    thickness of the depletion region.

    If :obj:`None` (the default), charge does not spread in the depletion
    region, as in the model of :cite:t:`Janesick2001`.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            na.shape(self.thickness_depletion),
            na.shape(self.width_backsurface),
            na.shape(self.width_depletion),
        )

    def width(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        s = thickness_substrate
        d = self.thickness_depletion
        f = s - d
        x = depth

        width_backsurface = self.width_backsurface
        if width_backsurface is None:
            width_backsurface = f

        # The fraction of the field-free region between the charge and the edge
        # of the depletion region, zero if there is no field-free region.
        remaining = np.maximum(f - x, 0 * f) / np.where(f > 0 * f, f, 1 * u.um)

        variance = np.square(width_backsurface) * remaining

        if self.width_depletion is not None:
            # The fraction of the depletion region the charge drifts across,
            # all of it if the depletion region has no thickness.
            crossed = np.minimum(np.maximum(s - x, 0 * d), d)
            crossed = np.where(d > 0 * d, crossed / np.where(d > 0 * d, d, 1 * u.um), 1)
            variance = variance + np.square(self.width_depletion) * crossed

        return np.sqrt(variance).to(u.um)

    def cdf(
        self,
        position: u.Quantity | na.AbstractScalar,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        width = self.width(depth, thickness_substrate)
        where = width > 0 * u.um
        width = np.where(where, width, 1 * u.um)
        t = (position / (np.sqrt(2) * width)).to(u.dimensionless_unscaled).value
        result = (1 + scipy.special.erf(t)) / 2
        return np.where(where, result, position >= 0 * u.um)

    def probability_same_pixel(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        width_pixel = _pixel_vector(width_pixel)
        width = self.width(depth, thickness_substrate)
        x = _probability_same_pixel(_ratio(width, width_pixel.x))
        y = _probability_same_pixel(_ratio(width, width_pixel.y))
        return x * y

    def width_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return _width_average(
            absorption=absorption,
            thickness_substrate=thickness_substrate,
            thickness_depletion=self.thickness_depletion,
            width_backsurface=self.width_backsurface,
            width_depletion=self.width_depletion,
        )

    def kernel(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: int = 3,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        if num % 2 != 1:
            raise ValueError(f"`num` must be odd, got {num}.")

        width_pixel = _pixel_vector(width_pixel)
        width = self.width_average(absorption, thickness_substrate)

        half = num // 2
        index_x = na.linspace(-half, half, axis=axis_x, num=num)
        index_y = na.linspace(-half, half, axis=axis_y, num=num)

        kx = _kernel_1d(_ratio(width, width_pixel.x), index_x)
        ky = _kernel_1d(_ratio(width, width_pixel.y), index_y)

        result = kx * ky
        result = result / result.sum(axis=(axis_x, axis_y))

        return na.FunctionArray(
            inputs=na.Cartesian2dVectorArray(index_x, index_y),
            outputs=result,
        )

    def mean_charge_capture(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        width_pixel = _pixel_vector(width_pixel)
        width = self.width_average(absorption, thickness_substrate)
        x = _capture(_ratio(width, width_pixel.x))
        y = _capture(_ratio(width, width_pixel.y))
        return x * y

    def _parameters_monte_carlo(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> dict[str, u.Quantity | na.AbstractScalar]:
        width_backsurface = self.width_backsurface
        if width_backsurface is None:
            width_backsurface = thickness_substrate - self.thickness_depletion
        width_depletion = self.width_depletion
        if width_depletion is None:
            width_depletion = 0 * u.um
        return dict(
            thickness_depletion=self.thickness_depletion,
            width_backsurface=width_backsurface,
            width_depletion=width_depletion,
        )
