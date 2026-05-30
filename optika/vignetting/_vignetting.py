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
):
    """
    An interface describing an arbitrary vignetting model, which maps scene
    coordinates to the fraction of light transmitted by the optical system.
    """

    @abc.abstractmethod
    def __call__(
        self,
        coordinates: optika.vectors.SceneVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the fraction of light transmitted for the given scene
        coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and field position of each point in the scene.
        """

    def inverse(
        self,
        coordinates: optika.vectors.SceneVectorArray,
    ) -> na.AbstractScalar:
        r"""
        Compute the inverse of the transmission, :math:`1 / T`, the factor
        which corrects for the vignetting at the given scene coordinates.

        Parameters
        ----------
        coordinates
            The wavelength and field position of each point in the scene.
        """
        return 1 / self(coordinates)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInterpolatedVignettingModel(
    AbstractVignettingModel,
):
    """
    A vignetting model defined by interpolating between known scene coordinates
    and their measured transmission.

    This class has two main members, :attr:`coordinates_scene` and
    :attr:`transmission`, the calibration points between which subclasses
    interpolate.
    """

    @property
    @abc.abstractmethod
    def coordinates_scene(self) -> optika.vectors.SceneVectorArray:
        """
        The wavelength and field position of each calibration point in the scene.
        """

    @property
    @abc.abstractmethod
    def transmission(self) -> na.AbstractScalar:
        """
        The fraction of light transmitted at each calibration point.
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
    A vignetting model which fits a polynomial to the measured transmission at
    known scene coordinates.

    Examples
    --------

    Plot the fit residual of a vignetting model with a radial transmission
    falloff and a deliberately underfit (linear) polynomial.

    .. jupyter-execute::

        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import optika

        scene = optika.vectors.SceneVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1 * u.deg,
                stop=+1 * u.deg,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=13,
            ),
        )
        transmission = 1 - 0.1 * (scene.field.length / u.deg) ** 2

        model = optika.vignetting.PolynomialVignettingModel(
            coordinates_scene=scene,
            transmission=transmission,
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=1,
        )

        fig, ax = model.plot_residual()
        na.plt.set_aspect("equal", ax=ax);
    """

    coordinates_scene: optika.vectors.SceneVectorArray = dataclasses.MISSING
    """The wavelength and field position of each calibration point in the scene."""

    transmission: na.AbstractScalar = dataclasses.MISSING
    """The fraction of light transmitted at each calibration point."""

    axis_wavelength: str = dataclasses.MISSING
    """The logical axis corresponding to changing wavelength."""

    axis_field: tuple[str, str] = dataclasses.MISSING
    """The logical axes corresponding to changing position in the scene."""

    degree: int = 1
    """The degree of the polynomial used to model the vignetting."""

    where: bool | na.AbstractScalar = True
    """A boolean mask selecting which calibration points to use for fitting."""

    @property
    def _axis_scene(self) -> tuple[str, ...]:
        """The logical axes over which the calibration points are distributed."""
        return (self.axis_wavelength, *self.axis_field)

    @functools.cached_property
    def fit(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping scene coordinates to transmission."""
        scene = self.coordinates_scene
        return na.PolynomialFitFunctionArray(
            inputs=scene,
            outputs=self.transmission,
            center=scene.mean(self._axis_scene),
            degree=self.degree,
            where_polynomial=self.where,
        )

    def __call__(
        self,
        coordinates: optika.vectors.SceneVectorArray,
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
        :attr:`transmission` and the transmission predicted by the polynomial
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
        scene = self.coordinates_scene
        field = scene.field
        wavelength = na.as_named_array(scene.wavelength)
        axis_wavelength = self.axis_wavelength

        residual = abs(self.transmission - self.fit.predictions)

        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = residual.max()

        ncols = na.shape(wavelength).get(axis_wavelength, 1)

        if figsize is None:
            # shape each subplot to the field-of-view aspect ratio, and widen
            # the figure to fit one subplot per wavelength
            height_subplot = 3
            aspect = (field.x.ptp() / field.y.ptp()).ndarray.value
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
                field,
                C=residual,
                ax=ax,
                colorizer=colorizer,
                **kwargs,
            )

            na.plt.set_xlabel(f"field $x$ ({na.unit(field.x):latex_inline})", ax=ax)
            na.plt.set_ylabel(
                f"field $y$ ({na.unit(field.y):latex_inline})",
                ax=ax[{axis_wavelength: 0}],
            )
            na.plt.set_title(wavelength.to_string_array(), ax=ax)

            plt.colorbar(
                mappable=matplotlib.cm.ScalarMappable(colorizer=colorizer),
                ax=ax.ndarray,
                label="transmission residual",
            )

        return fig, ax
