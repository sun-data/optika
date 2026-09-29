import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.sensors.materials._diffusion import _probability_same_pixel

__all__ = [
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
]


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
    Each model gives the standard deviation of the resulting charge cloud
    along one axis of the sensor, :meth:`width`, as a function of the depth
    at which the charge was created, and the quantities which depend on it.
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
    def width_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The square root of the variance of the charge cloud averaged over
        the depth at which photons are absorbed.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons.
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
    def mean_charge_capture(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The fraction of the charge from each photon collected in the pixel it
        was absorbed in, averaged over the position of the photon within the
        pixel; see :func:`optika.sensors.mean_charge_capture`.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
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

    @abc.abstractmethod
    def kernel(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar,
        axis_x: str,
        axis_y: str,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
        """
        The fraction of the charge from each photon collected in the pixel it
        was absorbed in and in each of its eight neighbors, averaged over the
        position of the photon within the pixel;
        see :func:`optika.sensors.kernel_diffusion`.

        Parameters
        ----------
        absorption
            The absorption coefficient of the light-sensitive region for the
            incident photons.
        thickness_substrate
            The thickness of the light-sensitive region of the sensor.
        width_pixel
            The width of a pixel.
        axis_x
            The name of the horizontal axis of the kernel.
        axis_y
            The name of the vertical axis of the kernel.
        """


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
        depleted = janesick.replace(width_depleted=0.8 * u.um)

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
    First, the width at the back surface, :math:`\sigma_\text{max}`,
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

        \sigma^2(x) = \sigma_\text{max}^2 \left( 1 - \frac{x}{x_{ff}} \right)_+
                      + \sigma_d^2 \, g(x),

    where :math:`(\cdot)_+` is zero for negative arguments and

    .. math::

        g(x) = \min \left( \frac{x_s - x}{x_d}, 1 \right)

    is the fraction of the depletion region crossed by charge created at
    :math:`x`.
    The average over the depth at which photons are absorbed is given in
    closed form by :func:`optika.sensors.charge_diffusion`.

    The charge cloud is taken to be Gaussian at every depth,
    so the kernel and the mean charge capture are those of a Gaussian whose
    variance is the variance averaged over depth.
    """

    thickness_depletion: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The thickness of the depletion region of the sensor."""

    width_max: None | u.Quantity | na.AbstractScalar = None
    """
    The standard deviation of the charge cloud of a photon absorbed at the
    back surface, before it crosses the depletion region.

    If :obj:`None` (the default), the thickness of the field-free region,
    as in the model of :cite:t:`Janesick2001`.
    """

    width_depleted: None | u.Quantity | na.AbstractScalar = None
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
            na.shape(self.width_max),
            na.shape(self.width_depleted),
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

        width_max = self.width_max
        if width_max is None:
            width_max = f

        # The fraction of the field-free region between the charge and the edge
        # of the depletion region, zero if there is no field-free region.
        remaining = np.maximum(f - x, 0 * f) / np.where(f > 0 * f, f, 1 * u.um)

        variance = np.square(width_max) * remaining

        if self.width_depleted is not None:
            # The fraction of the depletion region the charge drifts across,
            # all of it if the depletion region has no thickness.
            crossed = np.minimum(np.maximum(s - x, 0 * d), d)
            crossed = np.where(d > 0 * d, crossed / np.where(d > 0 * d, d, 1 * u.um), 1)
            variance = variance + np.square(self.width_depleted) * crossed

        return np.sqrt(variance).to(u.um)

    def width_average(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return optika.sensors.charge_diffusion(
            absorption=absorption,
            thickness_substrate=thickness_substrate,
            thickness_depletion=self.thickness_depletion,
            width_max=self.width_max,
            width_depleted=self.width_depleted,
        )

    def probability_same_pixel(
        self,
        depth: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        if not isinstance(width_pixel, na.AbstractCartesian2dVectorArray):
            width_pixel = na.Cartesian2dVectorArray(width_pixel, width_pixel)

        width = self.width(depth, thickness_substrate)

        def ratio(w: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
            where = w > 0 * w
            w = np.where(where, w, 1 * u.um)
            r = (width / w).to(u.dimensionless_unscaled).value
            return np.where(where, r, 0)

        return _probability_same_pixel(ratio(width_pixel.x)) * _probability_same_pixel(
            ratio(width_pixel.y)
        )

    def mean_charge_capture(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return optika.sensors.mean_charge_capture(
            width_diffusion=self.width_average(absorption, thickness_substrate),
            width_pixel=width_pixel,
        )

    def kernel(
        self,
        absorption: u.Quantity | na.AbstractScalar,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        width_pixel: u.Quantity | na.AbstractScalar,
        axis_x: str,
        axis_y: str,
    ) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
        return optika.sensors.kernel_diffusion(
            width_diffusion=self.width_average(absorption, thickness_substrate),
            width_pixel=width_pixel,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _parameters_monte_carlo(
        self,
        thickness_substrate: u.Quantity | na.AbstractScalar,
    ) -> dict[str, u.Quantity | na.AbstractScalar]:
        width_max = self.width_max
        if width_max is None:
            width_max = thickness_substrate - self.thickness_depletion
        width_depleted = self.width_depleted
        if width_depleted is None:
            width_depleted = 0 * u.um
        return dict(
            thickness_depletion=self.thickness_depletion,
            width_max=width_max,
            width_depleted=width_depleted,
        )


def _model_or_janesick(
    model_diffusion: None | AbstractDiffusionModel,
    thickness_depletion: None | u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
) -> AbstractDiffusionModel:
    """
    The model of diffusion given to a function, or the model of
    :cite:t:`Janesick2001` with the given depletion region if there is none.

    Parameters
    ----------
    model_diffusion
        The model of diffusion given to the function, if any.
    thickness_depletion
        The thickness of the depletion region given to the function, if any.
        If :obj:`None`, the thickness of the substrate, so that there is no
        field-free region.
    thickness_substrate
        The thickness of the light-sensitive region of the sensor.
    """
    if model_diffusion is None:
        if thickness_depletion is None:
            thickness_depletion = thickness_substrate
        return JanesickDiffusionModel(thickness_depletion=thickness_depletion)
    if thickness_depletion is not None:
        raise ValueError(
            "Give either `thickness_depletion` or `model_diffusion`, not both, "
            "since the model has its own depletion region."
        )
    return model_diffusion
