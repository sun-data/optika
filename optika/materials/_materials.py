from __future__ import annotations
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._layers import Layer

__all__ = [
    "AbstractMaterial",
    "Vacuum",
    "AbstractMirror",
    "Mirror",
    "MeasuredMirror",
    "Glass",
    "Dielectric",
    "MeasuredFilter",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMaterial(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    """An interface describing a generalized optical material."""

    @abc.abstractmethod
    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the index of refraction of this material for the given input rays

        Parameters
        ----------
        rays
            input rays used to evaluate the index of refraction
        """

    @abc.abstractmethod
    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the attenuation coefficient of the given rays

        Parameters
        ----------
        rays
            input rays to calculate the attenuation coefficient for
        """

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        The fraction of light that passes through the interface.

        Parameters
        ----------
        rays
            the input rays to calculate the efficiency for
        normal
            the vector perpendicular to the optical surface
        """

    @property
    @abc.abstractmethod
    def is_mirror(self) -> bool:
        """
        flag controlling whether this material reflects or transmits light
        """


@dataclasses.dataclass(eq=False, repr=False)
class Vacuum(
    AbstractMaterial,
):
    """Empty space, the default material."""

    @property
    def shape(self) -> dict[str, int]:
        return dict()

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 0 / u.mm

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMirror(
    AbstractMaterial,
):
    """An interface describing a generalized reflective mirror."""

    @property
    @abc.abstractmethod
    def substrate(self) -> None | Layer:
        """
        A layer representing the substrate supporting the reflective surface.
        """

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.index_refraction

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.attenuation

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return True


@dataclasses.dataclass(eq=False, repr=False)
class Mirror(
    AbstractMirror,
):
    """A perfect mirror material."""

    substrate: None | Layer = None
    """A layer representing the substrate supporting the reflective surface."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.substrate),
        )


def _interp_efficiency_measured(
    measurement: na.FunctionArray[na.SpectralDirectionalVectorArray, na.AbstractScalar],
    rays: optika.rays.RayVectorArray,
) -> na.ScalarLike:
    """
    Interpolate a measured efficiency onto the wavelengths of the given rays.

    Parameters
    ----------
    measurement
        A function array mapping wavelength and incidence angle to the
        measured efficiency.
    rays
        The rays at which to evaluate the efficiency.
    """

    wavelength = measurement.inputs.wavelength
    direction = measurement.inputs.direction
    efficiency = measurement.outputs

    if direction.size != 1:  # pragma: nocover
        raise ValueError(
            "Interpolating over different incidence angles is not supported."
        )

    if wavelength.ndim != 1:  # pragma: nocover
        raise ValueError(
            f"wavelength must be one dimensional, got shape {wavelength.shape}"
        )

    return na.interp(
        x=rays.wavelength,
        xp=wavelength,
        fp=efficiency,
    )


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredMirror(
    AbstractMirror,
):
    """
    A mirror where the reflectivity has been measured by an external source as
    a function of wavelength.

    Examples
    --------

    Create a mirror where the reflectivity is a Gaussian centered
    at 304 Angstroms.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the mean and standard deviation of the reflectivity peak.
        center = 304 * u.AA
        width = 10 * u.AA

        # Define a grid of wavelengths
        wavelength_min = center - 3 * width
        wavelength_max = center + 3 * width
        wavelength = na.linspace(
            start=wavelength_min,
            stop=wavelength_max,
            axis="wavelength",
            num=11,
        )

        # Define an array of simulated reflectivity measurements
        efficiency = na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            outputs=np.exp(-np.square((wavelength - center) / width) / 2),
        )

        # Create an instance of a MeasuredMirror object
        mirror = optika.materials.MeasuredMirror(efficiency)

        # Define a new grid of wavelengths at which to evaluate the interpolated
        # reflectivity
        wavelength_interp = na.linspace(
            start=wavelength_min,
            stop=wavelength_max,
            axis="wavelength",
            num=1001,
        )

        # Evaluate the interpolated reflectivity
        efficiency_interp = mirror.efficiency(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength_interp,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            normal=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Plot the interpolated reflectivity vs the measured reflectivity
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(wavelength_interp, efficiency_interp, label="interpolated");
            na.plt.scatter(efficiency.inputs.wavelength, efficiency.outputs, label="measured");
            ax.set_xlabel(f"wavelength ({ax.xaxis.get_label().get_text()})");
            ax.set_ylabel(f"reflectivity");
            ax.legend();
    """

    efficiency_measured: na.FunctionArray[
        na.SpectralDirectionalVectorArray, na.AbstractScalar
    ] = dataclasses.MISSING
    """
    A function array that maps wavelengths and incidence angles to the
    measured reflectivity.
    """

    substrate: None | Layer = None
    """A layer representing the substrate supporting the reflective surface."""

    serial_number: None | str | na.AbstractArray = None
    """A unique number associated with this material"""

    @property
    def shape(self) -> dict[str, int]:
        axis_wavelength = self.efficiency_measured.inputs.wavelength.axes
        shape = optika.shape(self.efficiency_measured.outputs)
        for ax in axis_wavelength:
            shape.pop(ax, None)
        return na.broadcast_shapes(
            shape,
            optika.shape(self.substrate),
            optika.shape(self.serial_number),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:

        return _interp_efficiency_measured(self.efficiency_measured, rays)


@dataclasses.dataclass(eq=False, repr=False)
class Glass(
    AbstractMaterial,
):
    r"""
    A transparent, refractive material whose index of refraction follows the
    three-term Sellmeier dispersion equation,

    .. math::

        n^2(\lambda) = 1
            + \frac{B_1 \lambda^2}{\lambda^2 - C_1}
            + \frac{B_2 \lambda^2}{\lambda^2 - C_2}
            + \frac{B_3 \lambda^2}{\lambda^2 - C_3},

    where :math:`\lambda` is the vacuum wavelength of the light, and
    :math:`B_i` (dimensionless) and :math:`C_i` (square length) are the
    Sellmeier coefficients of the glass.

    Unlike :class:`Vacuum` and :class:`Mirror`, this material changes the index
    of refraction of a transmitted ray, so a curved surface made of it has
    optical power and bends light according to Snell's law.

    Examples
    --------

    Plot the index of refraction of N-BK7 and F2 glass across the visible
    spectrum.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        wavelength = na.linspace(380, 750, axis="wavelength", num=101) * u.nm

        glasses = {
            "N-BK7": optika.materials.Glass.n_bk7(),
            "F2": optika.materials.Glass.f2(),
        }

        fig, ax = plt.subplots(constrained_layout=True)
        for name, glass in glasses.items():
            rays = optika.rays.RayVectorArray(wavelength=wavelength)
            na.plt.plot(
                wavelength,
                glass.index_refraction(rays),
                ax=ax,
                label=name,
            )
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("index of refraction");
        ax.legend();
    """

    b1: float | na.AbstractScalar = 0
    """The first dimensionless Sellmeier coefficient."""

    b2: float | na.AbstractScalar = 0
    """The second dimensionless Sellmeier coefficient."""

    b3: float | na.AbstractScalar = 0
    """The third dimensionless Sellmeier coefficient."""

    c1: u.Quantity | na.AbstractScalar = 0 * u.um**2
    """The first Sellmeier resonance (units of square length)."""

    c2: u.Quantity | na.AbstractScalar = 0 * u.um**2
    """The second Sellmeier resonance (units of square length)."""

    c3: u.Quantity | na.AbstractScalar = 0 * u.um**2
    """The third Sellmeier resonance (units of square length)."""

    @classmethod
    def n_bk7(cls) -> Glass:
        """
        SCHOTT N-BK7 borosilicate crown glass
        (:math:`n_d \\approx 1.5168`, :math:`V_d \\approx 64.2`).
        """
        return cls(
            b1=1.03961212,
            b2=0.231792344,
            b3=1.01046945,
            c1=0.00600069867 * u.um**2,
            c2=0.0200179144 * u.um**2,
            c3=103.560653 * u.um**2,
        )

    @classmethod
    def f2(cls) -> Glass:
        """
        SCHOTT F2 flint glass
        (:math:`n_d \\approx 1.6200`, :math:`V_d \\approx 36.4`).
        """
        return cls(
            b1=1.34533359,
            b2=0.209073176,
            b3=0.937357162,
            c1=0.00997743871 * u.um**2,
            c2=0.0470450767 * u.um**2,
            c3=111.886764 * u.um**2,
        )

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.b1),
            optika.shape(self.b2),
            optika.shape(self.b3),
            optika.shape(self.c1),
            optika.shape(self.c2),
            optika.shape(self.c3),
        )

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        w2 = np.square(rays.wavelength)
        n2 = 1 + (
            self.b1 * w2 / (w2 - self.c1)
            + self.b2 * w2 / (w2 - self.c2)
            + self.b3 * w2 / (w2 - self.c3)
        )
        return np.sqrt(n2)

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 0 / u.mm

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class Dielectric(
    AbstractMaterial,
):
    """
    A transparent, refractive material whose complex index of refraction is
    taken from the tabulated optical constants of a chemical.

    Like :class:`Glass`, this material refracts transmitted rays according to
    Snell's law, so a curved surface made of it has optical power.
    Unlike :class:`Glass`, its index of refraction comes from tables of
    measured optical constants (:cite:t:`Palik1997`, :cite:t:`Henke1993`, etc.)
    instead of a Sellmeier fit, so it can be used in the ultraviolet where the
    Sellmeier fits of common glasses are not valid, and it absorbs light
    according to the imaginary part of the index.

    Note that the tabulated absorption of many transparent crystals is zero
    throughout their transmission window, since the intrinsic absorption is
    below the sensitivity of the measurement, even though real samples of
    ultraviolet-grade material can absorb significantly near the band edge
    because of impurities and defects.

    Examples
    --------

    Plot the index of refraction and the internal transmission of a 2 mm
    thick window of magnesium fluoride in the vacuum ultraviolet.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the material
        mgf2 = optika.materials.Dielectric("MgF2")

        # Define the wavelengths at which to sample the material
        wavelength = na.linspace(100, 300, axis="wavelength", num=201) * u.nm
        rays = optika.rays.RayVectorArray(wavelength=wavelength)

        # Compute the index of refraction and the attenuation coefficient
        n = mgf2.index_refraction(rays)
        attenuation = mgf2.attenuation(rays)

        # Compute the internal transmission of a 2 mm thick window
        thickness = 2 * u.mm
        transmission = np.exp(-(attenuation * thickness).to(u.dimensionless_unscaled))

        fig, ax = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
        na.plt.plot(wavelength, n, ax=ax[0]);
        na.plt.plot(wavelength, transmission, ax=ax[1]);
        ax[0].set_ylabel("index of refraction");
        ax[1].set_ylabel("internal transmission");
        ax[1].set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
    """

    chemical: str | optika.chemicals.AbstractChemical = dataclasses.MISSING
    """
    The chemical composing this material, either as an empirical formula
    such as ``"MgF2"`` or as an instance of :class:`optika.chemicals.Chemical`.
    """

    @property
    def _chemical(self) -> optika.chemicals.AbstractChemical:
        result = self.chemical
        if not isinstance(result, optika.chemicals.AbstractChemical):
            result = optika.chemicals.Chemical(result)
        return result

    @property
    def shape(self) -> dict[str, int]:
        return optika.shape(self.chemical)

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        wavelength = na.as_named_array(rays.wavelength)
        return self._chemical.index_refraction(wavelength)

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        wavelength = na.as_named_array(rays.wavelength)
        return self._chemical.absorption(wavelength)

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredFilter(
    AbstractMaterial,
):
    """
    A transmissive material where the transmissivity has been measured by an
    external source as a function of wavelength.

    This is the transmissive counterpart of :class:`MeasuredMirror`.
    The measured transmissivity is applied once, at the interface, while
    refraction and absorption inside the material are delegated to
    :attr:`medium`.
    A coated window is therefore modeled as a pair of surfaces:
    a front surface made of this material, with :attr:`medium` set to the
    window material, and a back surface made of :class:`Vacuum`.

    A transmissivity can only be measured through the substrate of the coating,
    so by default (:attr:`is_medium_measured` is :obj:`True`) the measurement
    is taken to include the absorption of :attr:`medium` and the reflection
    from its back face, and the attenuation of :attr:`medium` is ignored to
    avoid counting it twice.
    Set :attr:`is_medium_measured` to :obj:`False` if the measurement has
    been reduced to the transmissivity of the interface alone,
    in which case the attenuation of :attr:`medium` is applied along the path
    of the transmitted rays, for example to model a window of a different
    thickness than the measured one.

    Examples
    --------

    Model a 2 mm thick magnesium fluoride window with a coating whose
    transmissivity was measured on a witness sample, and trace an oblique
    ray through it.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.linspace(120, 250, axis="wavelength", num=14) * u.nm

        # Define an array of simulated transmissivity measurements
        efficiency = na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=wavelength,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            outputs=0.2 * np.exp(-np.square((wavelength - 170 * u.nm) / (50 * u.nm))),
        )

        # Create the coated window material
        material = optika.materials.MeasuredFilter(
            efficiency_measured=efficiency,
            medium=optika.materials.Dielectric("MgF2"),
        )

        # Define the window as a pair of surfaces
        thickness = 2 * u.mm
        window = [
            optika.surfaces.Surface(
                name="front",
                material=material,
            ),
            optika.surfaces.Surface(
                name="back",
                material=optika.materials.Vacuum(),
                transformation=na.transformations.Cartesian3dTranslation(
                    z=thickness,
                ),
            ),
        ]

        # Define a ray incident on the window at 30 degrees
        angle = 30 * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=na.linspace(120, 250, axis="wavelength", num=131) * u.nm,
            position=na.Cartesian3dVectorArray(0, 0, -1) * u.mm,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )

        # Trace the ray through the window
        for surface in window:
            rays = surface.propagate_rays(rays)

        # Plot the transmissivity of the window and the lateral displacement
        # of the transmitted ray relative to an undeviated ray
        displacement = rays.position.x - thickness * np.tan(angle)
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
            na.plt.plot(rays.wavelength, rays.intensity, ax=ax[0], label="interpolated");
            na.plt.scatter(wavelength, efficiency.outputs, ax=ax[0], label="measured");
            na.plt.plot(rays.wavelength, displacement, ax=ax[1]);
            ax[0].set_ylabel("transmissivity");
            ax[0].legend();
            ax[1].set_ylabel(f"displacement ({displacement.unit:latex_inline})");
            ax[1].set_xlabel(f"wavelength ({rays.wavelength.unit:latex_inline})");
    """

    efficiency_measured: na.FunctionArray[
        na.SpectralDirectionalVectorArray, na.AbstractScalar
    ] = dataclasses.MISSING
    """
    A function array that maps wavelengths and incidence angles to the
    measured transmissivity.
    """

    medium: AbstractMaterial = dataclasses.field(default_factory=Vacuum)
    """
    The transmissive material behind the interface,
    which determines the refraction and absorption of the transmitted light.
    """

    is_medium_measured: bool = True
    """
    Whether :attr:`efficiency_measured` already includes the absorption of
    :attr:`medium`.
    If :obj:`True`, only the index of refraction of :attr:`medium` is used
    and its attenuation is ignored.
    If :obj:`False`, the attenuation of :attr:`medium` is applied along the
    path of the transmitted rays.
    """

    serial_number: None | str | na.AbstractArray = None
    """A unique number associated with this material"""

    @property
    def shape(self) -> dict[str, int]:
        axis_wavelength = self.efficiency_measured.inputs.wavelength.axes
        shape = optika.shape(self.efficiency_measured.outputs)
        for ax in axis_wavelength:
            shape.pop(ax, None)
        return na.broadcast_shapes(
            shape,
            optika.shape(self.medium),
            optika.shape(self.serial_number),
        )

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return self.medium.index_refraction(rays)

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        if self.is_medium_measured:
            return 0 / u.mm
        else:
            return self.medium.attenuation(rays)

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        result = _interp_efficiency_measured(self.efficiency_measured, rays)
        return result * self.medium.efficiency(rays, normal)

    @property
    def is_mirror(self) -> bool:
        return False
