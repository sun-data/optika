import abc
import dataclasses
import functools
import numpy as np
import matplotlib.axes
import matplotlib.cm
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import named_arrays as na
import optika

__all__ = [
    "AbstractDistortionModel",
    "AbstractLinearDistortionModel",
    "SimpleDistortionModel",
    "AbstractInterpolatedDistortionModel",
    "PolynomialDistortionModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDistortionModel(
    optika.mixins.Printable,
    optika.mixins.Shaped,
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
class AbstractLinearDistortionModel(
    AbstractDistortionModel,
):
    r"""
    A distortion model which is an affine transformation of the scene
    coordinates,

    .. math::

        \text{distort}(\vec{c}) = \mathbf{M} \, (\vec{c} - \vec{c}_0) + \vec{b},

    where :math:`\mathbf{M}` is :attr:`matrix`, :math:`\vec{c}_0` is
    :attr:`center`, and :math:`\vec{b}` is :attr:`intercept`.
    Since the transformation is linear, :meth:`undistort` is its *exact*
    inverse (unlike a polynomial fit).
    """

    @property
    @abc.abstractmethod
    def matrix(self) -> na.AbstractSpectralPositionalMatrixArray:
        """The linear part of the affine transformation."""

    @property
    @abc.abstractmethod
    def center(self) -> na.AbstractSpectralPositionalVectorArray:
        """The reference point subtracted from the coordinates before
        applying :attr:`matrix`."""

    @property
    @abc.abstractmethod
    def intercept(self) -> na.AbstractSpectralPositionalVectorArray:
        """The constant offset added after applying :attr:`matrix`."""

    def distort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        return self.matrix @ (coordinates - self.center) + self.intercept

    def undistort(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.SpectralPositionalVectorArray:
        return self.matrix.inverse @ (coordinates - self.intercept) + self.center


@dataclasses.dataclass(eq=False, repr=False)
class SimpleDistortionModel(
    AbstractLinearDistortionModel,
):
    r"""
    A simple analytic distortion model consisting of a rotation of the field,
    an isotropic plate scale, and a linear spectral dispersion along the
    rotated :math:`x` axis.

    This captures the distortion of an idealized spectrograph: the field
    center at the :attr:`reference` wavelength maps to the :attr:`reference`
    position on the sensor, and other wavelengths are displaced along the
    dispersion direction.

    Examples
    --------

    Distort a grid of scene coordinates and plot the result on the sensor,
    colored by wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        model = optika.distortion.SimpleDistortionModel(
            plate_scale=1 * u.arcsec / u.pix,
            dispersion=2 * u.nm / u.pix,
            angle=15 * u.deg,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(0, 0) * u.pix,
            ),
        )

        scene = na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
            ),
        )

        sensor = model.distort(scene)

        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_aspect("equal")
        for wavelength in scene.wavelength.ndarray:
            na.plt.scatter(
                sensor.position.x,
                sensor.position.y,
                where=scene.wavelength == wavelength,
                label=f"{wavelength}",
                ax=ax,
            )
        ax.set_xlabel(f"detector $x$ ({na.unit(sensor.position.x):latex_inline})")
        ax.set_ylabel(f"detector $y$ ({na.unit(sensor.position.y):latex_inline})")
        ax.legend();
    """

    plate_scale: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The spatial plate scale, in units such as :math:`\\text{arcsec} / \\text{pix}`."""

    dispersion: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The magnitude of the spectral dispersion, in units such as :math:`\\text{nm} / \\text{pix}`."""

    angle: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The angle of the dispersion direction with respect to the scene."""

    reference: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """The reference wavelength and the sensor position that the field center
    maps to at that wavelength."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.plate_scale),
            optika.shape(self.dispersion),
            optika.shape(self.angle),
            optika.shape(self.reference),
        )

    @functools.cached_property
    def matrix(self) -> na.SpectralPositionalMatrixArray:
        cos = np.cos(self.angle)
        sin = np.sin(self.angle)
        plate_scale = self.plate_scale
        dispersion = self.dispersion
        unit_wavelength = na.unit(self.reference.wavelength)
        return na.SpectralPositionalMatrixArray(
            wavelength=na.SpectralPositionalVectorArray(
                wavelength=1,
                position=na.Cartesian2dVectorArray(
                    x=0 * unit_wavelength / u.arcsec,
                    y=0 * unit_wavelength / u.arcsec,
                ),
            ),
            position=na.Cartesian2dMatrixArray(
                x=na.SpectralPositionalVectorArray(
                    wavelength=1 / dispersion,
                    position=na.Cartesian2dVectorArray(
                        x=cos / plate_scale,
                        y=-sin / plate_scale,
                    ),
                ),
                y=na.SpectralPositionalVectorArray(
                    wavelength=0 / dispersion,
                    position=na.Cartesian2dVectorArray(
                        x=sin / plate_scale,
                        y=cos / plate_scale,
                    ),
                ),
            ),
        )

    @property
    def center(self) -> na.SpectralPositionalVectorArray:
        return na.SpectralPositionalVectorArray(
            wavelength=self.reference.wavelength,
            position=na.Cartesian2dVectorArray(0, 0) * u.arcsec,
        )

    @property
    def intercept(self) -> na.AbstractSpectralPositionalVectorArray:
        return self.reference


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
    def shape(self) -> dict[str, int]:
        shape = na.broadcast_shapes(
            optika.shape(self.coordinates_scene),
            optika.shape(self.coordinates_sensor),
        )
        return {ax: n for ax, n in shape.items() if ax not in self._axis_scene}

    @property
    def _axis_scene(self) -> tuple[str, ...]:
        """The logical axes over which the calibration points are distributed."""
        return (self.axis_wavelength, *self.axis_field)

    @functools.cached_property
    def fit(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping scene position to sensor position."""
        scene = self.coordinates_scene
        return na.PolynomialFitFunctionArray.from_degree(
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
        return na.PolynomialFitFunctionArray.from_degree(
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

        # exclude the calibration points that were not used by the fit
        residual = np.where(self.where, residual, np.nan * unit)

        if vmin is None:
            vmin = 0 * unit
        if vmax is None:
            vmax = np.nanmax(residual)

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
                C=residual.value,
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
