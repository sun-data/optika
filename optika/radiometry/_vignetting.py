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
    "AbstractVignettingModel",
    "AbstractInterpolatedVignettingModel",
    "PolynomialVignettingModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractVignettingModel(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """
    An interface describing an arbitrary vignetting model, which maps scene
    coordinates to the relative illumination of the optical system (the spatial
    response normalized to one at the center of the field of view).
    """

    @abc.abstractmethod
    def __call__(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the relative illumination for the given scene coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and position of each point in the scene.
        """

    def inverse(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        r"""
        Compute the inverse of the illumination, :math:`1 / I`, the factor
        which corrects for the vignetting at the given scene coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and position of each point in the scene.
        """
        return 1 / self(coordinates)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInterpolatedVignettingModel(
    AbstractVignettingModel,
):
    """
    A vignetting model defined by interpolating between known scene coordinates
    and their measured illumination.

    This class has two main members, :attr:`coordinates_scene` and
    :attr:`illumination`, the calibration points between which subclasses
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
    def illumination(self) -> na.AbstractScalar:
        """
        The relative illumination at each calibration point.
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
class PolynomialVignettingModel(
    AbstractInterpolatedVignettingModel,
):
    """
    A vignetting model which fits a polynomial to the measured illumination at
    known scene coordinates.

    Examples
    --------

    Build a vignetting model with a radial illumination falloff fit by a
    deliberately underfit (linear) polynomial, then plot the illumination and
    the fit residual.

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
        illumination = 1 - 0.1 * (scene.position.length / u.deg) ** 2

        model = optika.radiometry.PolynomialVignettingModel(
            coordinates_scene=scene,
            illumination=illumination,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=1,
        )

        fig, ax = model.plot()
        na.plt.set_aspect("equal", ax=ax);

        fig, ax = model.plot_residual()
        na.plt.set_aspect("equal", ax=ax);
    """

    coordinates_scene: na.AbstractSpectralPositionalVectorArray = dataclasses.MISSING
    """The wavelength and position of each calibration point in the scene."""

    illumination: na.AbstractScalar = dataclasses.MISSING
    """The relative illumination at each calibration point."""

    axis_wavelength: str = dataclasses.MISSING
    """The logical axis corresponding to changing wavelength."""

    axis_field: tuple[str, str] = dataclasses.MISSING
    """The logical axes corresponding to changing position in the scene."""

    degree: int = 1
    """The degree of the polynomial used to model the vignetting."""

    where: bool | na.AbstractScalar = True
    """A boolean mask selecting which calibration points to use for fitting."""

    @property
    def shape(self) -> dict[str, int]:
        shape = na.broadcast_shapes(
            optika.shape(self.coordinates_scene),
            optika.shape(self.illumination),
        )
        return {ax: n for ax, n in shape.items() if ax not in self._axis_scene}

    @property
    def _axis_scene(self) -> tuple[str, ...]:
        """The logical axes over which the calibration points are distributed."""
        return (self.axis_wavelength, *self.axis_field)

    @functools.cached_property
    def fit(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping scene coordinates to illumination."""
        scene = self.coordinates_scene
        return na.PolynomialFitFunctionArray.from_degree(
            inputs=scene,
            outputs=self.illumination,
            center=scene.mean(self._axis_scene),
            degree=self.degree,
            where_polynomial=self.where,
        )

    def __call__(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        return self.fit(coordinates).outputs

    def plot_residual(
        self,
        figsize: None | tuple[float, float] = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, na.ScalarArray]:
        """
        Plot the residual of the :attr:`fit` as a function of field angle, with
        a separate subplot for each wavelength.

        The residual is the absolute difference between the calibration
        :attr:`illumination` and the illumination predicted by the polynomial
        fit.

        Parameters
        ----------
        figsize
            The size of the returned figure in inches.
            If :obj:`None`, the size is chosen automatically from the number
            of wavelengths and the aspect ratio of the field of view.
        cmap
            The colormap used to map the residual to colors.
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
        return self._plot(
            abs(self.illumination - self.fit.predictions),
            label="illumination residual",
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def plot(
        self,
        figsize: None | tuple[float, float] = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, na.ScalarArray]:
        """
        Plot the calibration :attr:`illumination` as a function of field angle,
        with a separate subplot for each wavelength.

        Parameters
        ----------
        figsize
            The size of the returned figure in inches.
            If :obj:`None`, the size is chosen automatically from the number
            of wavelengths and the aspect ratio of the field of view.
        cmap
            The colormap used to map the illumination to colors.
        vmin
            The illumination value mapped to the lowest color.
            If :obj:`None`, defaults to zero.
        vmax
            The illumination value mapped to the highest color.
            If :obj:`None`, defaults to the maximum illumination.
        kwargs
            Additional keyword arguments passed to
            :func:`named_arrays.plt.pcolormesh`.
        """
        return self._plot(
            self.illumination,
            label="illumination",
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _plot(
        self,
        values: na.AbstractScalar,
        label: str,
        figsize: None | tuple[float, float] = None,
        cmap: None | str | matplotlib.colors.Colormap = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, na.ScalarArray]:
        """
        Plot a scalar quantity as a function of field angle, with a separate
        subplot for each wavelength.
        """
        scene = self.coordinates_scene
        position = scene.position
        wavelength = na.as_named_array(scene.wavelength)
        axis_wavelength = self.axis_wavelength

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = values.max()

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
                    vmin=na.as_named_array(vmin).ndarray,
                    vmax=na.as_named_array(vmax).ndarray,
                ),
            )

            na.plt.pcolormesh(
                position,
                C=values,
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
                label=label,
            )

        return fig, ax
