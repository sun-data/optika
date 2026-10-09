import abc
from typing import Callable
import numbers
import dataclasses
from typing_extensions import Self
import numpy as np
import scipy.optimize
import scipy.special
import astropy.units as u
import named_arrays as na
import optika
from ._quadrature import _absorption_positive, _integrate_gauss_legendre
from ._measurements import MeanChargeCapture
from ._gaussian import (
    _width_average,
    _ratio,
    _probability_same_pixel,
    _kernel_1d,
)

__all__ = [
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
]


_tolerance_kernel = 1e-6
"""
The largest fraction of the charge created at any depth which a kernel of the
default size may leave out.
"""


def _check_model(
    diffusion: object,
) -> None:
    """
    Raise a :class:`TypeError` unless `diffusion` is :obj:`None` or a model of
    diffusion.

    Parameters
    ----------
    diffusion
        The value of a `diffusion` argument.
    """
    if diffusion is not None and not isinstance(diffusion, AbstractDiffusionModel):
        raise TypeError(
            "`diffusion` must be None or an instance of "
            f"`optika.sensors.diffusion.AbstractDiffusionModel`, got {diffusion!r}."
        )


def _pixel_vector(
    width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
) -> na.AbstractCartesian2dVectorArray:
    """A pixel width as a vector, square if a scalar is given."""
    if not isinstance(width_pixel, na.AbstractCartesian2dVectorArray):
        width_pixel = na.Cartesian2dVectorArray(width_pixel, width_pixel)
    return width_pixel


def _indices_kernel(
    num: int,
    axis_x: str,
    axis_y: str,
) -> na.Cartesian2dVectorArray:
    """The indices of the pixels of a kernel, relative to its center pixel."""
    half = num // 2
    return na.Cartesian2dVectorArray(
        x=na.linspace(-half, half, axis=axis_x, num=num),
        y=na.linspace(-half, half, axis=axis_y, num=num),
    )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDiffusionModel(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    r"""
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

    Notes
    -----

    The quantities averaged over depth, such as :meth:`kernel_average` and
    :meth:`mean_charge_capture`, average the quantity for charge created at
    each depth :math:`z`, weighted by the probability that a photon is
    absorbed there,

    .. math::

        \left\langle f \right\rangle = \frac{\displaystyle \int_0^{z_s} f(z) \, \alpha e^{-\alpha z} dz}
                                            {1 - e^{-\alpha z_s}},

    where :math:`\alpha` is the absorption coefficient and :math:`z_s` is the
    thickness of the light-sensitive region.
    This is the same average that the Monte Carlo simulation of
    :func:`optika.sensors.electrons_measured` draws its samples from,
    and that :func:`optika.sensors.vmr_signal` takes of the noise.

    These averages generally have no closed form, so they are evaluated by
    quadrature, in variables chosen to make the quadrature converge quickly.
    The field-free region, :math:`0 < z < z_f`, and the depletion region,
    :math:`z_f < z < z_s`, are integrated separately.
    Writing the average over each region as an integral over the cumulative
    absorption probability within it distributes the nodes according to where
    photons are actually absorbed, which matters because the optical depth of
    the sensor spans four orders of magnitude across the wavelengths of
    interest.

    The width of the charge cloud typically has a square-root branch point
    where it vanishes:
    at the edge of the depletion region, :math:`z = z_f`,
    for the charge that diffuses across the field-free region,
    and at the gates, :math:`z = z_s`,
    for the charge that spreads as it drifts across the depletion region.
    In the field-free region, substituting :math:`1 - r^2` for the cumulative
    absorption probability puts the branch point at :math:`r = 0`,
    where the width becomes proportional to :math:`r`, which removes it.
    In terms of :math:`r` the optical depth is

    .. math::

        \alpha z = -\log \left( e^{-\alpha z_f} + \left( 1 - e^{-\alpha z_f} \right) r^2 \right),

    which is evaluated in this form where most of the photons absorbed in the
    region are absorbed above the depth, so that it stays accurate when
    :math:`e^{-\alpha z_f}` underflows,
    and as the logarithm of one minus the cumulative probability elsewhere,
    so that it stays accurate when the region is optically thin.
    In the depletion region, of thickness :math:`z_d`, the same substitution
    in terms of :math:`v` puts the branch point at the gates, :math:`v = 0`,
    where the optical depth below the edge of the depletion region is

    .. math::

        \alpha (z - z_f) = -\log \left( e^{-\alpha z_d} + \left( 1 - e^{-\alpha z_d} \right) v^2 \right).

    Gauss-Legendre quadrature with
    :obj:`~optika.sensors.diffusion._quadrature._num_gauss_legendre` nodes is
    then applied to each region, and the average is accurate to about
    one part in :math:`10^6` for optical depths :math:`\alpha z_f` between
    :math:`0.02` and :math:`3000`,
    which covers silicon between 1 and 10000 angstroms;
    a midpoint rule in the cumulative probability needs some thirty times as
    many nodes to reach a hundred times worse accuracy.

    A sensor which absorbs weakly absorbs photons uniformly in depth,
    and the averages for a sensor which does not absorb at all,
    :math:`\alpha = 0`, are that limit.
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
    def kernel(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: None | int = None,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        """
        The fraction of the charge created at a given depth which is collected
        in the pixel the photon was absorbed in and in each of the pixels
        around it, averaged over the position of the photon within its pixel.

        The fractions are not normalized,
        so the charge which lands beyond the kernel is missing from their sum.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            charge was created, between zero and `thickness_substrate`.
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
            If :obj:`None` (the default), the kernel is made just large enough
            to leave out no more than one part in a million of the charge
            created at any depth.
        """

    def width_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The square root of the variance of the charge cloud along one axis
        averaged over the depth at which photons are absorbed.

        This averages the square of :meth:`width` with :meth:`average_depth`.
        Models for which the average has a closed form override it.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons, per unit of perpendicular depth.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """
        s = thickness_substrate

        def variance(depth: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
            return np.square(self.width(depth, s))

        return np.sqrt(self.average_depth(variance, absorption, s))

    def kernel_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: None | int = None,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        """
        The fraction of the charge from each photon collected in the pixel it
        was absorbed in and in each of the pixels around it, averaged over
        the position of the photon within its pixel and over the depth at
        which it was absorbed.

        This averages :meth:`kernel` with :meth:`average_depth`.
        Like :meth:`kernel`, the fractions are not normalized,
        so the center of the kernel is the :meth:`mean_charge_capture` whatever
        its size, and the charge which lands beyond the kernel is missing from
        their sum.

        Each depth is weighted by the probability that a photon is absorbed
        there, so this is the kernel of the charge each photon creates,
        before any of it is lost near the back surface of the sensor.

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
            If :obj:`None` (the default), the kernel is made just large enough
            to leave out no more than one part in a million of the charge
            created at any depth.
        """
        num = self._num_kernel(num, thickness_substrate, width_pixel)

        def kernel(depth: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
            return self.kernel(
                depth=depth,
                thickness_substrate=thickness_substrate,
                width_pixel=width_pixel,
                axis_x=axis_x,
                axis_y=axis_y,
                num=num,
            ).outputs

        result = self.average_depth(kernel, absorption, thickness_substrate)

        return na.FunctionArray(
            inputs=_indices_kernel(num, axis_x, axis_y),
            outputs=result,
        )

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

        This averages the center of :meth:`kernel` with :meth:`average_depth`,
        weighting each depth by the probability that a photon is absorbed
        there, as :meth:`kernel_average` does.

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
        axis_x = "_mean_charge_capture_x"
        axis_y = "_mean_charge_capture_y"

        def capture(depth: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
            kernel = self.kernel(
                depth=depth,
                thickness_substrate=thickness_substrate,
                width_pixel=width_pixel,
                axis_x=axis_x,
                axis_y=axis_y,
                num=1,
            )
            return kernel.outputs.sum(axis=(axis_x, axis_y))

        return self.average_depth(capture, absorption, thickness_substrate)

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

    def _num_kernel(
        self,
        num: None | int,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> int:
        """
        The number of pixels along each axis of a kernel:
        `num` if given, checked to be a positive odd number,
        or otherwise the smallest which leaves out no more than
        :obj:`_tolerance_kernel` of the charge created at any depth.

        The widest charge cloud is the one created at the back surface.
        A kernel which reaches :math:`h` pixels beyond the center pixel
        misses charge from a photon anywhere in that pixel only if the charge
        moves more than :math:`h` pixel widths along one of the axes,
        so the default kernel is the smallest for which the widest cloud
        reaches that far with a probability of at most
        :obj:`_tolerance_kernel`, summed over the two axes.

        Parameters
        ----------
        num
            The number of pixels along each axis of the kernel,
            or :obj:`None` for the default.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
        """
        if num is not None:
            if not isinstance(num, numbers.Integral) or num < 1 or num % 2 != 1:
                raise ValueError(f"`num` must be a positive odd integer, got {num}.")
            return int(num)

        s = thickness_substrate
        width_pixel = _pixel_vector(width_pixel)
        depth = 0 * s

        # Without a cloud, or without pixels to spread it over,
        # the charge stays in the pixel the photon was absorbed in.
        widths = [width_pixel.x, width_pixel.y]
        if np.all(self.width(depth, s) == 0 * u.um):
            return 1
        if all(np.all(w == 0 * w) for w in widths):
            return 1

        def outside(half: int) -> float:
            """The fraction of the widest cloud beyond `half` pixels."""
            result = 0
            for w in widths:
                position = half * w
                inside = self.cdf(position, depth, s) - self.cdf(-position, depth, s)
                fraction = np.where(w > 0 * w, 1 - inside, 0)
                result = result + float(na.as_named_array(fraction).max().ndarray)
            return result

        # Double the half-width until the kernel is large enough,
        # then bisect between the last two half-widths.
        high = 1
        while outside(high) > _tolerance_kernel:
            high = 2 * high
        low = high // 2
        while high - low > 1:
            middle = (low + high) // 2
            if outside(middle) > _tolerance_kernel:
                low = middle
            else:
                high = middle

        return 2 * high + 1

    def average_depth(
        self,
        function: Callable[[u.Quantity | na.AbstractScalar], na.AbstractScalar],
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        depth_break: None | u.Quantity | na.AbstractScalar = None,
    ) -> na.AbstractScalar:
        """
        The average of a function of the depth at which charge is created,
        over the depths at which photons are absorbed in the light-sensitive
        region, evaluated by the quadrature described in the notes of this
        class.

        Parameters
        ----------
        function
            A function of the distance from the back surface of the sensor at
            which the charge was created.
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons, per unit of perpendicular depth.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        depth_break
            An optional depth at which `function` has a kink,
            such as the end of the implant layer,
            where the interval of integration is split so that the quadrature
            still converges quickly.
        """
        s = thickness_substrate
        f = np.maximum(s - self.thickness_depletion, 0 * s)

        absorption = _absorption_positive(absorption, s)

        az_f = (absorption * f).to(u.dimensionless_unscaled).value
        az_s = (absorption * s).to(u.dimensionless_unscaled).value
        az_d = az_s - az_f

        fraction_absorbed = -np.expm1(-az_s)
        fraction_f = -np.expm1(-az_f)
        fraction_d = -np.expm1(-az_d)

        axis = "_diffusion_depth"

        def optical_depth(
            az_region: na.AbstractScalar,
            fraction: na.AbstractScalar,
            t: na.AbstractScalar,
        ) -> na.AbstractScalar:
            """
            The optical depth below the start of a region of optical
            thickness `az_region` at the substituted variable `t`,
            accurate near both ends of the region,
            and equal to the thickness of the region at its end, ``t = 0``,
            even where the transmittance of the region underflows.
            """
            # The probability of a photon being absorbed between the start of
            # the region and the depth.
            p = fraction * (1 - np.square(t))
            near = p < 0.5
            result_near = -np.log1p(-np.where(near, p, 0))
            with np.errstate(divide="ignore"):
                result_far = -np.log(np.exp(-az_region) + fraction * np.square(t))
            result = np.where(near, result_near, result_far)
            return np.minimum(result, az_region)

        def integrand_f(r: na.AbstractScalar) -> na.AbstractScalar:
            az = optical_depth(az_f, fraction_f, r)
            return function(az / absorption) * 2 * r

        def integrand_d(v: na.AbstractScalar) -> na.AbstractScalar:
            az = az_f + optical_depth(az_d, fraction_d, v)
            return function(az / absorption) * 2 * v

        if depth_break is None:
            r_break = v_break = 1
        else:
            az_break = (absorption * depth_break).to(u.dimensionless_unscaled).value

            # Where the break falls in each region; at an end of the region
            # if it is outside it, which leaves a single interval.
            # The probability of a photon being absorbed between the break and
            # the end of each region is written with `expm1` so that it stays
            # accurate in an optically thin region.
            fraction_f_safe = np.where(fraction_f > 0, fraction_f, 1)
            fraction_d_safe = np.where(fraction_d > 0, fraction_d, 1)
            az_break_d = np.maximum(az_break - az_f, 0)
            r_break = np.sqrt(
                np.clip(
                    np.exp(-az_break)
                    * -np.expm1(np.minimum(az_break - az_f, 0))
                    / fraction_f_safe,
                    0,
                    1,
                ),
            )
            v_break = np.sqrt(
                np.clip(
                    np.exp(-az_break_d)
                    * -np.expm1(np.minimum(az_break_d - az_d, 0))
                    / fraction_d_safe,
                    0,
                    1,
                ),
            )

        result_f = _integrate_gauss_legendre(integrand_f, 0, r_break, axis)
        result_d = _integrate_gauss_legendre(integrand_d, 0, v_break, axis)
        if depth_break is not None:
            result_f = result_f + _integrate_gauss_legendre(
                integrand_f, r_break, 1, axis
            )
            result_d = result_d + _integrate_gauss_legendre(
                integrand_d, v_break, 1, axis
            )

        # The fractions of the absorbed photons absorbed in each region.
        weight_f = fraction_f / fraction_absorbed
        weight_d = np.exp(-az_f) * fraction_d / fraction_absorbed

        return weight_f * result_f + weight_d * result_d

    def fit_mean_charge_capture(
        self,
        mcc_measured: MeanChargeCapture,
    ) -> Self:
        """
        A copy of this model with the thickness of its depletion region fitted
        to a measured mean charge capture, holding its other parameters fixed.

        The fit minimizes the root-mean-square difference between the
        measured and the modeled mean charge capture,
        with the thickness of the depletion region between zero and the
        thickness of the light-sensitive region of the sensor that was
        measured.

        Parameters
        ----------
        mcc_measured
            The measured mean charge capture as a function of the vacuum
            wavelength of the incident photons,
            together with the sensor it was measured on.
        """
        thickness_substrate = mcc_measured.thickness_substrate
        width_pixel = mcc_measured.width_pixel
        chemical_substrate = mcc_measured.chemical_substrate

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

    Plot the central 3 by 3 pixels of the kernel of a 13 micron pixel for
    photons of 1403 angstroms, averaged over the depth at which they are
    absorbed.

    .. jupyter-execute::

        # Compute the kernel
        kernel = janesick.kernel_average(
            absorption=optika.chemicals.Chemical("Si").absorption(1403 * u.AA),
            thickness_substrate=thickness_substrate,
            width_pixel=13 * u.um,
            axis_x="x",
            axis_y="y",
            num=3,
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

    where :math:`\alpha` is the absorption coefficient.
    For :math:`\sigma_\text{bs} = x_{ff}` and :math:`\sigma_d = 0` this is the
    average of the width given by :cite:t:`Janesick2001`.
    If the depletion region is thicker than the light-sensitive region,
    there is no field-free region, so :math:`x_{ff}` is zero,
    and the charge created at the back surface crosses only part of the
    depletion region, so the :math:`\alpha x_d` in the second term becomes
    :math:`\alpha x_s`.

    The kernel at each depth, :meth:`kernel`, is that of the Gaussian charge
    cloud created there.
    :meth:`kernel_average` and :meth:`mean_charge_capture` average it over the
    depth at which photons are absorbed,
    as described in the notes of
    :class:`~optika.sensors.diffusion.AbstractDiffusionModel`.
    Since a photon can strike anywhere within its pixel, the Gaussian is
    convolved with a rectangle function the width of a pixel before it is
    integrated over each pixel, so that the mean charge capture is

    .. math::

        P_\text{MCC} = \left\langle \prod_{i \in \{x, y\}} \left\{
            \sqrt{\frac{2}{\pi}} \frac{\sigma(x)}{d_i}
            \left[ \exp \left( -\frac{d_i^2}{2 \sigma^2(x)} \right) - 1 \right]
            + \text{erf} \left( \frac{d_i}{\sqrt{2} \sigma(x)} \right)
        \right\} \right\rangle,

    where :math:`d_i` is the width of a pixel along each axis
    and :math:`\langle \cdot \rangle` is the average over depth.
    This is not the mean charge capture of a single Gaussian whose variance is
    :math:`\overline{\sigma}^2`, which underestimates the charge kept in the
    central pixel when photons are absorbed over a range of depths,
    as weakly absorbed ones are.
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
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: None | int = None,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        num = self._num_kernel(num, thickness_substrate, width_pixel)
        width_pixel = _pixel_vector(width_pixel)
        inputs = _indices_kernel(num, axis_x, axis_y)
        width = self.width(depth, thickness_substrate)
        kx = _kernel_1d(_ratio(width, width_pixel.x), inputs.x)
        ky = _kernel_1d(_ratio(width, width_pixel.y), inputs.y)
        return na.FunctionArray(
            inputs=inputs,
            outputs=kx * ky,
        )

    def _parameters_monte_carlo(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> dict[str, u.Quantity | na.AbstractScalar]:
        width_backsurface = self.width_backsurface
        if width_backsurface is None:
            width_backsurface = np.maximum(
                thickness_substrate - self.thickness_depletion,
                0 * thickness_substrate,
            )
        width_depletion = self.width_depletion
        if width_depletion is None:
            width_depletion = 0 * u.um
        return dict(
            thickness_depletion=self.thickness_depletion,
            width_backsurface=width_backsurface,
            width_depletion=width_depletion,
        )
