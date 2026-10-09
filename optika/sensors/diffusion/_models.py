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
from . import _slab

__all__ = [
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
    "SlabDiffusionModel",
]


_tolerance_kernel = 1e-6
"""
The largest fraction of the charge created at any depth which a kernel of the
default size may leave out.
"""


_monte_carlo_gaussian = 0
"""
The Monte Carlo simulation of :func:`optika.sensors.electrons_measured`
spreads the electrons created at each depth with a Gaussian of the width the
model gives there.
"""

_monte_carlo_slab = 1
"""
The Monte Carlo simulation of :func:`optika.sensors.electrons_measured`
spreads the electrons created in the field-free region with the mixture of
Gaussians of :class:`SlabDiffusionModel` at the depth they were created at,
and the electrons created in the depletion region with a Gaussian.
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


def _fraction_depletion(
    depth: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The fraction of the depletion region that charge created at a given depth
    drifts across, all of it if the depletion region has no thickness.

    Parameters
    ----------
    depth
        The distance from the back surface of the sensor at which the charge
        was created.
    thickness_substrate
        The thickness of the light-sensitive region of the sensor.
    thickness_depletion
        The thickness of the depletion region of the sensor.
    """
    d = thickness_depletion
    crossed = np.minimum(np.maximum(thickness_substrate - depth, 0 * d), d)
    where = d > 0 * d
    return np.where(where, crossed / np.where(where, d, 1 * u.um), 1)


def _sum_product(
    a: na.AbstractScalar,
    b: na.AbstractScalar,
    axis: str,
) -> na.ScalarArray:
    """
    The sum over `axis` of the product of two arrays,
    without forming the product before the sum,
    which for the kernel of a model averaged over a distribution would hold
    every pixel of the kernel at every node of the average.

    Parameters
    ----------
    a
        The first array, which has `axis`.
    b
        The second array, which has `axis`.
    axis
        The logical axis to sum over.
    """
    shape_a = na.shape(a)
    shape_b = na.shape(b)
    only_a = {k: v for k, v in shape_a.items() if k not in shape_b}
    only_b = {k: v for k, v in shape_b.items() if k not in shape_a}
    shared = na.broadcast_shapes(
        {k: v for k, v in shape_a.items() if k in shape_b},
        {k: v for k, v in shape_b.items() if k in shape_a},
    )
    num = shared.pop(axis)

    a = na.broadcast_to(a, shared | {axis: num} | only_a).ndarray
    b = na.broadcast_to(b, shared | {axis: num} | only_b).ndarray
    a = a.reshape(a.shape[: len(shared) + 1] + (-1,))
    b = b.reshape(b.shape[: len(shared) + 1] + (-1,))

    result = np.einsum("...ki,...kj->...ij", a, b)
    result = result.reshape(
        tuple(shared.values()) + tuple(only_a.values()) + tuple(only_b.values())
    )

    return na.ScalarArray(
        ndarray=result,
        axes=tuple(shared) + tuple(only_a) + tuple(only_b),
    )


def _check_num(
    num: int,
) -> int:
    """
    The number of pixels along each axis of a kernel,
    checked to be a positive odd integer,
    so that the kernel is centered on the pixel the photon was absorbed in.

    Parameters
    ----------
    num
        The number of pixels along each axis of the kernel.
    """
    if not isinstance(num, numbers.Integral) or num < 1 or num % 2 != 1:
        raise ValueError(f"`num` must be a positive odd integer, got {num}.")
    return int(num)


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


def _kernel_mixture(
    variance: na.AbstractScalar,
    weight: na.AbstractScalar,
    axis: str,
    width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    inputs: na.Cartesian2dVectorArray,
) -> na.ScalarArray:
    """
    The kernel of a mixture of Gaussians, which are independent along the
    two axes of the kernel,
    averaged over the position of the photon within its pixel.

    Parameters
    ----------
    variance
        The variance of each Gaussian along one axis.
    weight
        The weight of each Gaussian.
    axis
        The logical axis of the Gaussians.
    width_pixel
        The width of a pixel.
    inputs
        The indices of the pixels of the kernel, from :func:`_indices_kernel`.
    """
    width_pixel = _pixel_vector(width_pixel)
    width = np.sqrt(variance)
    kx = weight * _kernel_1d(_ratio(width, width_pixel.x), inputs.x)
    ky = _kernel_1d(_ratio(width, width_pixel.y), inputs.y)
    return _sum_product(kx, ky, axis)


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
    Each model gives the charge cloud that arrives at the gates as a mixture
    of Gaussians, :meth:`mixture`, which depends on the depth at which the
    charge was created,
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
    def mixture(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        axis: str,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        """
        The charge cloud created at a given depth as a mixture of Gaussians:
        the variance of each Gaussian along one axis, including the spread
        acquired in the depletion region, and the weight of each Gaussian.

        Each electron is drawn from the mixture independently of the other
        electrons created by the same photon,
        and given the Gaussian it is drawn from,
        its offsets along the two axes are independent of each other.
        :meth:`cdf`, :meth:`kernel`, and :meth:`probability_same_pixel` are
        computed from the mixture,
        which can also be used directly where the charge cloud is evaluated
        too many times to go through them,
        such as in a fit of the model to the charge clouds of particle tracks.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            charge was created, between zero and `thickness_substrate`.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        axis
            The logical axis of the Gaussians.

        Returns
        -------
        variance
            The variance of each Gaussian along one axis.
        weight
            The weight of each Gaussian, which sum to one along `axis`.
        """

    def _mixture_pairs(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        axis: str,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        """
        The separation along one axis of two electrons created at a given
        depth by the same photon, as a mixture of Gaussians:
        the variance of the separation for each pair of Gaussians of
        :meth:`mixture` the electrons may be drawn from, and the weight of
        the pair.

        Given the Gaussians they are drawn from, the separation of the two
        electrons is Gaussian with the sum of their variances.
        Exchanging the electrons gives the same pair, so each pair of
        different Gaussians is counted once, with twice the weight.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            electrons were created.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        axis
            The logical axis of the pairs of Gaussians.
        """
        axis_mixture = "_diffusion_mixture"
        variance, weight = self.mixture(depth, thickness_substrate, axis_mixture)
        num = na.broadcast_shapes(na.shape(variance), na.shape(weight))[axis_mixture]
        i, j = np.triu_indices(num)
        i = na.ScalarArray(i, axes=axis)
        j = na.ScalarArray(j, axes=axis)
        variance = variance[{axis_mixture: i}] + variance[{axis_mixture: j}]
        weight = weight[{axis_mixture: i}] * weight[{axis_mixture: j}]
        weight = weight * np.where(i == j, 1, 2)
        return variance, weight

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
        axis = "_diffusion_mixture"
        variance, weight = self.mixture(depth, thickness_substrate, axis)
        width = np.sqrt(variance)
        where = width > 0 * u.um
        width = np.where(where, width, 1 * u.um)
        t = (position / (np.sqrt(2) * width)).to(u.dimensionless_unscaled).value
        result = (1 + scipy.special.erf(t)) / 2
        result = np.where(where, result, position >= 0 * u.um)
        return (weight * result).sum(axis)

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
        axis = "_diffusion_mixture_pair"
        width_pixel = _pixel_vector(width_pixel)
        variance, weight = self._mixture_pairs(depth, thickness_substrate, axis)

        # The separation of the electrons has twice the variance of a single
        # electron with the mean of their variances.
        width = np.sqrt(variance / 2)
        x = _probability_same_pixel(_ratio(width, width_pixel.x))
        y = _probability_same_pixel(_ratio(width, width_pixel.y))

        # Averaging the probability of not sharing a pixel keeps the result
        # exactly one where the charge does not spread.
        return 1 - (weight * (1 - x * y)).sum(axis)

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
        num = self._num_kernel(num, thickness_substrate, width_pixel)
        inputs = _indices_kernel(num, axis_x, axis_y)
        axis = "_diffusion_mixture"
        variance, weight = self.mixture(depth, thickness_substrate, axis)
        return na.FunctionArray(
            inputs=inputs,
            outputs=_kernel_mixture(variance, weight, axis, width_pixel, inputs),
        )

    def kernel_pair(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        axis_x: str,
        axis_y: str,
        num: None | int = None,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
        r"""
        The probability that two electrons created at a given depth by the
        same photon are collected in pixels a given offset apart,
        averaged over the position of the photon within its pixel.

        Its center is :meth:`probability_same_pixel`,
        and it sums to one, except for the pairs of electrons which land
        farther apart than the kernel reaches.
        Under uniform illumination, it is the shape of the covariance that
        charge diffusion gives the noise of neighboring pixels;
        see :func:`optika.sensors.covariance_signal`.

        Given the Gaussians of :meth:`mixture` the two electrons are drawn
        from, their separation along each axis is Gaussian with the sum of
        their variances.
        In units of the width of a pixel, a separation :math:`\delta` puts
        the electrons :math:`\Delta` pixels apart with probability
        :math:`(1 - |\delta - \Delta|)_+`
        for a photon anywhere within its pixel,
        which is also the probability that a single electron displaced by
        :math:`\delta` lands :math:`\Delta` pixels away.
        So along each axis, a pair of Gaussians has the kernel of a single
        Gaussian with the sum of their variances,
        and the kernels of the pairs are averaged with their weights.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            electrons were created, between zero and `thickness_substrate`.
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
            which must be odd so that the kernel is centered on zero offset.
            If :obj:`None` (the default), the kernel is made just large enough
            to leave out no more than one part in a million of the pairs of
            electrons created at any depth,
            which takes about :math:`\sqrt{2}` times as many pixels beyond the
            center as :meth:`kernel`.
        """
        num = self._num_kernel(num, thickness_substrate, width_pixel, pair=True)
        inputs = _indices_kernel(num, axis_x, axis_y)
        axis = "_diffusion_mixture_pair"
        variance, weight = self._mixture_pairs(depth, thickness_substrate, axis)
        return na.FunctionArray(
            inputs=inputs,
            outputs=_kernel_mixture(variance, weight, axis, width_pixel, inputs),
        )

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

    @property
    @abc.abstractmethod
    def _kind_monte_carlo(self) -> int:
        """
        How the Monte Carlo simulation of
        :func:`optika.sensors.electrons_measured` spreads the charge:
        :obj:`_monte_carlo_gaussian` or :obj:`_monte_carlo_slab`.
        """

    def _num_kernel(
        self,
        num: None | int,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
        pair: bool = False,
    ) -> int:
        """
        The number of pixels along each axis of a kernel:
        `num` if given, checked to be a positive odd number,
        or otherwise the smallest which leaves out no more than
        :obj:`_tolerance_kernel` of the charge created at any depth,
        or of the pairs of electrons if `pair` is :obj:`True`.

        The widest charge cloud is the one created at the back surface.
        A kernel which reaches :math:`h` pixels beyond the center pixel
        misses charge from a photon anywhere in that pixel only if the charge
        moves more than :math:`h` pixel widths along one of the axes,
        and the kernel of :meth:`kernel_pair` misses a pair of electrons only
        if they are separated by more than that,
        so the default kernel is the smallest for which the displacements of
        the widest cloud, or the separations of its pairs of electrons,
        reach that far with a probability of at most
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
        pair
            Whether the kernel is that of :meth:`kernel_pair`.
        """
        if num is not None:
            return _check_num(num)

        s = thickness_substrate
        width_pixel = _pixel_vector(width_pixel)
        depth = 0 * s

        axis = "_diffusion_num_kernel"
        if pair:
            variance, weight = self._mixture_pairs(depth, s, axis)
        else:
            variance, weight = self.mixture(depth, s, axis)
        width = np.sqrt(variance)

        # Without a cloud, or without pixels to spread it over,
        # the charge stays in the pixel the photon was absorbed in.
        widths = [width_pixel.x, width_pixel.y]
        if np.all(width == 0 * u.um):
            return 1
        if all(np.all(w == 0 * w) for w in widths):
            return 1

        where = width > 0 * u.um
        width = np.where(where, width, 1 * u.um)

        def outside(half: int) -> float:
            """
            The fraction of the displacements, or of the separations,
            beyond `half` pixels.
            """
            result = 0
            for w in widths:
                t = (half * w / (np.sqrt(2) * width)).to(u.dimensionless_unscaled)
                beyond = np.where(where, scipy.special.erfc(t.value), 0)
                fraction = np.where(w > 0 * w, (weight * beyond).sum(axis), 0)
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

    The charge cloud at each depth is a single Gaussian,
    so its :meth:`mixture` has one component,
    and the kernel at each depth, :meth:`kernel`, is that of the Gaussian.
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

    def _variance(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The variance of the charge cloud along one axis for charge created at
        a given depth.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            charge was created.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """
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
            crossed = _fraction_depletion(x, s, d)
            variance = variance + np.square(self.width_depletion) * crossed

        return variance.to(u.um**2)

    def width(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return np.sqrt(self._variance(depth, thickness_substrate))

    def mixture(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        axis: str,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        variance = na.add_axes(self._variance(depth, thickness_substrate), axis)
        weight = na.ScalarArray(np.ones(1), axes=axis)
        return variance, weight

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

    @property
    def _kind_monte_carlo(self) -> int:
        return _monte_carlo_gaussian


@dataclasses.dataclass(eq=False, repr=False)
class SlabDiffusionModel(
    AbstractDiffusionModel,
):
    r"""
    The charge cloud of charge diffusing across a field-free region,
    reflected at the back surface, until it reaches the depletion region,
    with an optional spread acquired in the depletion region.

    Examples
    --------

    Plot the width of the charge cloud against the depth at which the charge
    was created, for this model and for the model of :cite:t:`Janesick2001`
    with the same depletion region.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the thickness of the light-sensitive region of the sensor
        thickness_substrate = 14 * u.um

        # Define this model and the model of Janesick (2001)
        slab = optika.sensors.diffusion.SlabDiffusionModel(
            thickness_depletion=8.7 * u.um,
        )
        janesick = optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=slab.thickness_depletion,
        )

        # Define a grid of depths through the light-sensitive region
        depth = na.linspace(0, thickness_substrate, axis="depth", num=1001)

        # Plot the width of the charge cloud against depth
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                depth,
                slab.width(depth, thickness_substrate),
                ax=ax,
                label="slab",
            )
            na.plt.plot(
                depth,
                janesick.width(depth, thickness_substrate),
                ax=ax,
                label="Janesick (2001)",
            )
            ax.set_xlabel(f"depth ({ax.get_xlabel()})")
            ax.set_ylabel(f"charge diffusion ({ax.get_ylabel()})")
            ax.legend()

    Plot the fraction of the charge created at the back surface that lands
    farther than a given distance along one axis,
    which falls off exponentially rather than as a Gaussian.

    .. jupyter-execute::

        # Define a grid of distances along one axis
        position = na.linspace(0, 40, axis="position", num=101) * u.um

        def beyond(model):
            cdf = model.cdf(position, 0 * u.um, thickness_substrate)
            return 2 * (1 - cdf)

        # Plot the fraction beyond each distance
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(position, beyond(slab), ax=ax, label="slab")
            na.plt.plot(position, beyond(janesick), ax=ax, label="Janesick (2001)")
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 1)
            ax.set_xlabel(f"distance ({ax.get_xlabel()})")
            ax.set_ylabel("fraction of the charge beyond")
            ax.legend()

    Fit both models to the mean charge capture measured by
    :cite:t:`Stern2004`, and plot them against it.

    .. jupyter-execute::

        # Load the measurement
        measured = optika.sensors.diffusion.mcc_stern2004("thick")

        # Fit the thickness of the depletion region of each model
        slab_fit = slab.fit_mean_charge_capture(measured)
        janesick_fit = janesick.fit_mean_charge_capture(measured)

        # Define a grid of wavelengths
        wavelength = na.geomspace(1, 100, axis="wavelength", num=101) * u.AA
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

        def mcc(model):
            # on the CCD the measurement was made on
            return model.mean_charge_capture(
                absorption=absorption,
                thickness_substrate=measured.thickness_substrate,
                width_pixel=measured.width_pixel,
            )

        # Plot the fits against the measurement
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                measured.inputs,
                measured.outputs,
                ax=ax,
                color="black",
                label="Stern et al. (2004)",
            )
            na.plt.plot(wavelength, mcc(slab_fit), ax=ax, label="slab")
            na.plt.plot(
                wavelength,
                mcc(janesick_fit),
                ax=ax,
                label="Janesick (2001)",
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel("mean charge capture")
            ax.legend()

    Notes
    -----

    Charge created in the field-free region, at a distance :math:`x` from the
    back surface, diffuses until it reaches the edge of the depletion region,
    a distance :math:`x_{ff} = x_s - x_d` from the back surface,
    where :math:`x_s` is the thickness of the light-sensitive region and
    :math:`x_d` the thickness of the depletion region.
    The back surface reflects it.
    Its motion across the sensor is independent of its motion along it,
    so given the time :math:`T` it takes to reach the depletion region,
    its offset along each axis is Gaussian with variance :math:`2 D T`,
    where :math:`D` is the diffusion coefficient.
    The charge cloud is the average of these Gaussians over the distribution
    of :math:`T`, whose Fourier transform is

    .. math::

        \left\langle e^{i \mathbf{k} \cdot \mathbf{r}} \right\rangle
            = \frac{\cosh k x}{\cosh k x_{ff}},

    where :math:`k` is the magnitude of :math:`\mathbf{k}`.
    Its standard deviation along each axis is

    .. math::

        \sigma_\text{ff}(x) = \sqrt{x_{ff}^2 - x^2},

    which is never smaller than that of the model of :cite:t:`Janesick2001`,
    :math:`\sqrt{x_{ff} (x_{ff} - x)}`, and is equal to it at the back surface
    and at the depletion region,
    and the fraction of it that arrives at less than a position :math:`y`
    along one axis is

    .. math::

        \frac{1}{2} + \frac{1}{\pi} \arctan \left(
            \frac{\sinh \left( \pi y / 2 x_{ff} \right)}
                 {\cos \left( \pi x / 2 x_{ff} \right)}
        \right),

    so it falls off exponentially, as :math:`e^{-\pi |y| / 2 x_{ff}}`,
    rather than as a Gaussian.
    The cloud is not the product of its profiles along the two axes,
    since an electron which takes a long time to reach the depletion region
    spreads along both.

    Charge that reaches the depletion region still diffuses as it drifts to
    the gates, which adds a Gaussian spread
    :math:`\sigma_d^2 \, g(x)` to the variance,
    as in :class:`~optika.sensors.diffusion.JanesickDiffusionModel`.
    The electrons created by one photon start from the same place but diffuse
    independently, so the probability that two of them are collected in the
    same pixel depends on the sum of their variances.

    The average over :math:`T` has no closed form,
    so it is taken by quadrature over the quantiles of :math:`T`
    with eight nodes,
    which makes the charge cloud at each depth a mixture of eight Gaussians
    whose variances are tabulated once against depth.
    Each electron is drawn from the mixture independently,
    so two electrons of the same photon may be drawn from different
    Gaussians.
    The mixture is given by :meth:`mixture`.
    Its weights are the same at every depth,
    and the variance each Gaussian acquires crossing the field-free region is
    the square of the thickness of that region times a function of the
    fraction of the way across it the charge was created,
    so the mixture for one thickness of the field-free region follows from
    that for another by scaling the variances,
    before the spread of the depletion region is added.
    Averaged over the depth at which the charge was created,
    the kernel and the probability that two electrons share a pixel agree
    with the exact cloud to about one part in :math:`10^5`;
    at a single depth they agree to about one part in :math:`10^3`.
    The variance of the mixture is that of the exact cloud, which
    :meth:`width` gives, to about a percent farther than a tenth of
    :math:`x_{ff}` from the depletion region.
    Closer than that, the rare electrons which wander back toward the back
    surface set the variance, and only the widest Gaussian holds them.
    Likewise, the tail of the mixture follows the exponential tail of the
    exact cloud to within ten percent until less than about :math:`10^{-5}`
    of the charge lies beyond it, and then falls off as the widest Gaussian.
    The Monte Carlo simulation of :func:`optika.sensors.electrons_measured`
    samples the same mixture, so it agrees with the averages of this model,
    and its cost does not grow with the number of electrons per photon.

    Near the depletion region, the few electrons that wander back toward the
    back surface give the charge cloud a heavy tail,
    so the averages over depth converge more slowly than those of
    :class:`~optika.sensors.diffusion.JanesickDiffusionModel`:
    to a few parts in :math:`10^6` where a pixel is at least three quarters
    as wide as the field-free region is thick,
    and to about two parts in :math:`10^5` where it is a quarter as wide.
    """

    thickness_depletion: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The thickness of the depletion region of the sensor."""

    width_depletion: None | u.Quantity | na.AbstractScalar = None
    """
    The standard deviation acquired by charge drifting across the full
    thickness of the depletion region.

    If :obj:`None` (the default), charge does not spread in the depletion
    region.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            na.shape(self.thickness_depletion),
            na.shape(self.width_depletion),
        )

    def _thickness_field_free(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The thickness of the field-free region, zero if the depletion region
        fills the light-sensitive region.

        Parameters
        ----------
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """
        s = thickness_substrate
        return np.maximum(s - self.thickness_depletion, 0 * s)

    def _variance_depletion(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        The variance acquired along each axis by charge drifting across the
        depletion region.

        Parameters
        ----------
        depth
            The distance from the back surface of the sensor at which the
            charge was created.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        """
        if self.width_depletion is None:
            return 0 * u.um**2
        crossed = _fraction_depletion(
            depth=depth,
            thickness_substrate=thickness_substrate,
            thickness_depletion=self.thickness_depletion,
        )
        return np.square(self.width_depletion) * crossed

    def mixture(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        axis: str,
    ) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        f = self._thickness_field_free(thickness_substrate)
        where = f > 0 * f
        f_safe = np.where(where, f, 1 * u.um)

        # The distance from the depletion region in units of the thickness of
        # the field-free region, zero in the depletion region.
        delta = ((f - depth) / f_safe).to(u.dimensionless_unscaled).value
        delta = np.where(where, np.clip(delta, 0, 1), 0)

        s, weight = _slab._transit(delta, axis)

        variance = s * np.square(f)
        variance = variance + self._variance_depletion(depth, thickness_substrate)

        return variance, weight

    def width(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        f = self._thickness_field_free(thickness_substrate)
        x = depth
        variance = np.maximum(f - x, 0 * f) * (f + x)
        variance = variance + self._variance_depletion(depth, thickness_substrate)
        return np.sqrt(variance).to(u.um)

    def _parameters_monte_carlo(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> dict[str, u.Quantity | na.AbstractScalar]:
        width_depletion = self.width_depletion
        if width_depletion is None:
            width_depletion = 0 * u.um
        return dict(
            thickness_depletion=self.thickness_depletion,
            width_backsurface=self._thickness_field_free(thickness_substrate),
            width_depletion=width_depletion,
        )

    @property
    def _kind_monte_carlo(self) -> int:
        return _monte_carlo_slab
