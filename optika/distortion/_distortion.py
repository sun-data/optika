import abc
import dataclasses
import functools
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.visualization
import named_arrays as na
import optika

__all__ = [
    "AbstractDistortionModel",
    "AbstractInterpolatedDistortionModel",
    "PolynomialDistortionModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDistortionModel(
    optika.mixins.Printable,
):
    """
    An interface describing an arbitrary distortion model,
    which maps scene coordinates to sensor coordinates (and vice versa).

    A distortion model carries the wavelength along with the position, since
    the mapping from a point in the scene to a point on the sensor generally
    depends on wavelength (for example, the dispersion of a spectrograph).
    As a result :meth:`distort` and :meth:`undistort` are inverses of one
    another only up to the accuracy of the model.
    """

    @abc.abstractmethod
    def distort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        """
        Convert scene coordinates to sensor coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and position of each point in the scene.
        """

    @abc.abstractmethod
    def undistort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        """
        Convert sensor coordinates to scene coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and sensor position of each point.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInterpolatedDistortionModel(
    AbstractDistortionModel,
):
    """
    A distortion model defined by interpolating between known scene/sensor
    coordinates.

    This class has two main members, :attr:`coordinates_scene` and
    :attr:`coordinates_sensor`, the calibration points between which subclasses
    interpolate.
    """

    @property
    @abc.abstractmethod
    def coordinates_scene(self) -> na.AbstractSpectralPositionalVectorArray:
        """
        The wavelength and position of each calibration point in the scene.
        """

    @property
    @abc.abstractmethod
    def coordinates_sensor(self) -> na.AbstractCartesian2dVectorArray:
        """
        The position of each calibration point mapped onto the sensor.
        """

    @property
    @abc.abstractmethod
    def axis_wavelength(self) -> str:
        """The logical axis corresponding to changing wavelength."""

    @property
    @abc.abstractmethod
    def axis_field(self) -> tuple[str, str]:
        """The logical axes corresponding to changing position in the scene."""


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialDistortionModel(
    AbstractInterpolatedDistortionModel,
):
    """
    A distortion model which fits a polynomial to known scene/sensor coordinates.

    The forward model (:meth:`distort`) is a polynomial fit mapping scene
    position to sensor position as a function of wavelength.
    The inverse model (:meth:`undistort`) is a *separate* polynomial fit in the
    opposite direction, so the round trip is exact only to the accuracy of the
    two fits.

    Examples
    --------

    Plot the fit residual of a distortion model with a deliberately
    underfit (linear) polynomial.

    .. jupyter-execute::

        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import optika

        scene = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-1 * u.deg,
                stop=+1 * u.deg,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=13,
            ),
        )
        sensor = na.Cartesian2dVectorArray(
            x=scene.position.x * (10 * u.mm / u.deg)
            + scene.position.x**2 * (1 * u.mm / u.deg**2),
            y=scene.position.y * (10 * u.mm / u.deg)
            + scene.position.y**2 * (1 * u.mm / u.deg**2),
        )

        model = optika.distortion.PolynomialDistortionModel(
            coordinates_scene=scene,
            coordinates_sensor=sensor,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=1,
        )

        fig, ax = model.plot_residual()
        na.plt.set_aspect("equal", ax=ax);
    """

    coordinates_scene: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """The wavelength and position of each calibration point in the scene."""

    coordinates_sensor: na.AbstractCartesian2dVectorArray = dataclasses.MISSING
    """The position of each calibration point mapped onto the sensor."""

    axis_wavelength: str = dataclasses.MISSING
    """The logical axis corresponding to changing wavelength."""

    axis_field: tuple[str, str] = dataclasses.MISSING
    """The logical axes corresponding to changing position in the scene."""

    degree: int = 1
    """The degree of the polynomial used to model the distortion."""

    where: bool | na.AbstractScalar = True
    """A boolean mask selecting which calibration points to use for fitting."""

    @property
    def _axis_scene(self) -> tuple[str, ...]:
        """The logical axes over which the calibration points are distributed."""
        return (self.axis_wavelength, *self.axis_field)

    @functools.cached_property
    def fit(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping scene position to sensor position."""
        scene = self.coordinates_scene
        return na.PolynomialFitFunctionArray(
            inputs=scene,
            outputs=self.coordinates_sensor,
            center=scene.mean(self._axis_scene),
            degree=self.degree,
            where_polynomial=self.where,
        )

    @functools.cached_property
    def fit_inverse(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping sensor position back to scene position."""
        scene = self.coordinates_scene
        inputs = na.SpectralPositionalVectorArray(
            wavelength=scene.wavelength,
            position=self.coordinates_sensor,
        )
        return na.PolynomialFitFunctionArray(
            inputs=inputs,
            outputs=scene.position,
            center=inputs.mean(self._axis_scene),
            degree=self.degree,
            where_polynomial=self.where,
        )

    def distort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        return na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=self.fit(coordinates).outputs,
        )

    def undistort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        return na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=self.fit_inverse(coordinates).outputs,
        )

    def plot_residual(
        self,
        figsize: None | tuple[float, float] = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, na.ScalarArray]:
        """
        Plot the residual of the forward :attr:`fit` as a function of field
        angle, with a separate subplot for each wavelength.

        The residual is the magnitude of the difference between the calibration
        sensor positions, :attr:`coordinates_sensor`, and the positions
        predicted by the forward polynomial fit.

        Parameters
        ----------
        figsize
            The size of the returned figure in inches.
            If :obj:`None`, the size is chosen automatically from the number
            of wavelengths and the aspect ratio of the field of view.
        cmap
            The colormap used to map the residual magnitude to colors.
        vmin
            The residual value mapped to the lowest color.
            If :obj:`None`, defaults to zero.
        vmax
            The residual value mapped to the highest color.
            If :obj:`None`, defaults to the maximum residual.
        kwargs
            Additional keyword arguments passed to
            :func:`named_arrays.plt.pcolormesh`.
        """
        scene = self.coordinates_scene
        position = scene.position
        wavelength = na.as_named_array(scene.wavelength)
        axis_wavelength = self.axis_wavelength

        residual = (self.coordinates_sensor - self.fit.predictions).length
        unit = na.unit(residual)

        if vmin is None:
            vmin = 0 * unit
        if vmax is None:
            vmax = residual.max()

        ncols = na.shape(wavelength).get(axis_wavelength, 1)

        if figsize is None:
            # shape each subplot to the field-of-view aspect ratio, and widen
            # the figure to fit one subplot per wavelength
            height_subplot = 3
            aspect = (position.x.ptp() / position.y.ptp()).ndarray.value
            figsize = (
                ncols * height_subplot * aspect + 1.5,
                height_subplot + 1,
            )

        with astropy.visualization.quantity_support():
            fig, ax = na.plt.subplots(
                axis_cols=axis_wavelength,
                ncols=ncols,
                sharex=True,
                sharey=True,
                squeeze=False,
                figsize=figsize,
                constrained_layout=True,
            )

            colorizer = plt.Colorizer(
                cmap=cmap,
                norm=plt.Normalize(
                    vmin=na.as_named_array(vmin).ndarray.to_value(unit),
                    vmax=na.as_named_array(vmax).ndarray.to_value(unit),
                ),
            )

            na.plt.pcolormesh(
                position,
                C=residual,
                ax=ax,
                colorizer=colorizer,
                **kwargs,
            )

            na.plt.set_xlabel(f"field $x$ ({na.unit(position.x):latex_inline})", ax=ax)
            na.plt.set_ylabel(
                f"field $y$ ({na.unit(position.y):latex_inline})",
                ax=ax[{axis_wavelength: 0}],
            )
            na.plt.set_title(wavelength.to_string_array(), ax=ax)

            plt.colorbar(
                mappable=matplotlib.cm.ScalarMappable(colorizer=colorizer),
                ax=ax.ndarray,
                label=f"residual ({unit:latex_inline})",
            )

        return fig, ax
