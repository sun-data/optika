import abc
import dataclasses
import named_arrays as na
import optika

__all__ = [
    "AbstractEffectiveAreaModel",
    "InterpolatedEffectiveAreaModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractEffectiveAreaModel(
    optika.mixins.Printable,
):
    """
    An interface describing the effective area of an optical system as a
    function of wavelength.
    """

    @abc.abstractmethod
    def __call__(
        self,
        wavelength: na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The effective area of the optical system at the given wavelength.

        Parameters
        ----------
        wavelength
            The wavelength at which to evaluate the effective area.
        """


@dataclasses.dataclass(eq=False, repr=False)
class InterpolatedEffectiveAreaModel(
    AbstractEffectiveAreaModel,
):
    """
    An effective area model which linearly interpolates between measured
    calibration points.

    Linear interpolation is used (rather than a polynomial fit) because the
    effective area of a real system has sharp features --- absorption edges,
    filter cutoffs, and multilayer reflectivity peaks --- that a polynomial
    would fit poorly.

    Examples
    --------

    Interpolate a measured effective area curve and plot the result.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # A coarse set of calibration measurements
        wavelength = na.linspace(100, 1000, axis="wavelength", num=10) * u.AA
        area = 10 * np.exp(-(((wavelength - 500 * u.AA) / (150 * u.AA)) ** 2))
        area = area * u.cm**2

        model = optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=wavelength,
            area=area,
            axis_wavelength="wavelength",
        )

        # Evaluate the model on a finer grid
        wavelength_fit = na.linspace(100, 1000, axis="wavelength", num=201) * u.AA

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.scatter(wavelength, area, ax=ax, label="calibration")
        na.plt.plot(wavelength_fit, model(wavelength_fit), ax=ax, label="interpolated")
        ax.set_xlabel(f"wavelength ({na.unit(wavelength):latex_inline})")
        ax.set_ylabel(f"effective area ({na.unit(area):latex_inline})")
        ax.legend();
    """

    wavelength: na.AbstractScalar = dataclasses.MISSING
    """The wavelength of each calibration point."""

    area: na.AbstractScalar = dataclasses.MISSING
    """The measured effective area at each calibration point."""

    axis_wavelength: str = dataclasses.MISSING
    """The logical axis corresponding to changing wavelength."""

    def __call__(
        self,
        wavelength: na.AbstractScalar,
    ) -> na.AbstractScalar:
        return na.interp(
            wavelength,
            self.wavelength,
            self.area,
            axis=self.axis_wavelength,
        )
