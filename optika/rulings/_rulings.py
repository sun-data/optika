import abc
import dataclasses
from dataclasses import MISSING
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractRulingSpacing

__all__ = [
    "incident_effective",
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
    "SinusoidalRulings",
    "SquareRulings",
    "SawtoothRulings",
    "TriangularRulings",
    "RectangularRulings",
]


def incident_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    index_refraction: float | na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
    diffraction_order: int,
    spacing_rulings: u.Quantity | na.AbstractScalar,
    normal_rulings: na.AbstractCartesian3dVectorArray,
) -> na.Cartesian3dVectorArray:
    r"""
    The effective propagation direction of some rays incident on a diffraction
    grating.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The propagation direction of the incident light.
    index_refraction
        The index of refraction of the current medium.
    normal
        A unit vector perpendicular to the surface on which the rulings are
        inscribed.
    diffraction_order
        The diffraction order of the reflected or transmitted rays.
    spacing_rulings
        The distance between the parallel planes defining the rulings.
    normal_rulings
        A unit vector perpendicular to the parallel planes defining the rulings.

    Notes
    -----

    Our goal is to find the effective propagation direction of a light ray
    incident on a diffraction grating.
    This effective, incident ray can be used in Snell's law to find the
    direction of the diffracted rays.
    To start, consider the Dirichlet boundary conditions given in
    Equation :eq:`boundary-condition` of the :func:`~optika.materials.snells_law`
    notes.

    .. math::

        A_1 \exp\left[i \mathbf{k}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
        = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]

    To include the ruling pattern, we model it as a phase shift of the wave at the interface,

    .. math::
        :label: phase-shift

        \phi(x, y) = i \boldsymbol{\kappa} \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})

    where

    .. math::

        \boldsymbol{\kappa} = -\frac{2 \pi m}{d} \hat{\boldsymbol{\kappa}},

    :math:`m` is the diffraction order,
    :math:`d` is the groove spacing,
    and :math:`\hat{\boldsymbol{\kappa}}` is a unit vector normal to the
    planes of the rulings.

    With the inclusion of Equation :eq:`phase-shift`, Equation :eq:`boundary-condition` becomes:

    .. math::
        :label: boundary-condition-shifted

         A_1 \exp\left[i (\mathbf{k}_1 + \boldsymbol{\kappa}) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
        = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right].

    By following a similar procedure to the one described in the notes of
    :func:`~optika.materials.snells_law`,
    we find that everything is exactly the same if we replace every instance
    of :math:`\mathbf{k}_1` with an effective incident wavevector:

    .. math::

        \boxed{\mathbf{k}_\text{e} = \hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1}.
    """

    unit = spacing_rulings.unit

    w = wavelength.to(unit).value  # noqa: F841
    a = direction  # noqa: F841
    n = index_refraction  # noqa: F841
    m = diffraction_order  # noqa: F841
    d = spacing_rulings.value  # noqa: F841
    g = normal_rulings  # noqa: F841

    ax = a.x  # noqa: F841
    ay = a.y  # noqa: F841
    az = a.z  # noqa: F841

    ux = normal.x  # noqa: F841
    uy = normal.y  # noqa: F841
    uz = normal.z  # noqa: F841

    result = na.numexpr.evaluate(
        "a + sign(ax * ux + ay * uy + az * uz) * m * w * g / (n * d)"
    )

    return result


def _sinc(x: na.ScalarLike) -> na.ScalarLike:
    """
    The normalized sinc function, :math:`\\sin(\\pi x) / (\\pi x)`, which is
    one at zero.
    """
    x_safe = np.where(x == 0, 1, x)
    return np.where(
        x == 0,
        1,
        np.sin(np.pi * x_safe * u.rad) / (np.pi * x_safe),
    )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    """
    Interface for the interaction of a ruled surface with incident light
    """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> int | na.AbstractScalar:
        """
        the diffraction order to simulate
        """

    @property
    @abc.abstractmethod
    def spacing(
        self,
    ) -> u.Quantity | na.AbstractScalar | AbstractRulingSpacing:
        """
        Spacing between adjacent rulings at the given position.
        """

    @property
    def spacing_(self) -> AbstractRulingSpacing:
        """
        A normalized version of :attr:`spacing` that is guaranteed to be
        an instance of :class:`optika.rulings.AbstractRulingSpacing`.
        """
        spacing = self.spacing
        if not isinstance(spacing, optika.rulings.AbstractRulingSpacing):
            spacing = optika.rulings.ConstantRulingSpacing(
                constant=spacing,
                normal=na.Cartesian3dVectorArray(1, 0, 0),
            )
        return spacing

    def incident_effective(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:
        """
        Compute the effective propagation direction of the given rays
        using :func:`~optika.rulings.incident_effective`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. Ignored.
        is_mirror
            Whether the rulings are on a reflective surface. Ignored.
        """

        kappa = self.spacing_(
            position=rays.position,
            normal=normal,
        )

        spacing = kappa.length

        direction = incident_effective(
            wavelength=rays.wavelength,
            direction=rays.direction,
            index_refraction=rays.index_refraction,
            normal=normal,
            diffraction_order=self.diffraction_order,
            spacing_rulings=spacing,
            normal_rulings=kappa / spacing,
        )

        return rays.replace(direction=direction)

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light that is diffracted into a given order.

        Parameters
        ----------
        rays
            The light rays incident on the rulings, before diffraction.
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface, in which case
            the diffracted light propagates back into the incident medium.

        Notes
        -----
        The thin-grating efficiencies implemented by the subclasses of this
        class are taken from :cite:t:`Magnusson1978`, who consider a
        *volume* phase grating: a slab of thickness :math:`d` whose index
        of refraction is modulated by :math:`n_1 f(x)`, so that a ray
        crossing it at angle :math:`\theta` accumulates a phase
        proportional to :math:`n_1 d / \cos \theta`, the more the more
        obliquely it crosses.

        The rulings here are instead a *relief* on a surface, a groove
        profile of height :math:`h(x)` between a medium of index
        :math:`n_1` and one of index :math:`n_2`. Under the scalar
        (Kirchhoff) approximation, the phase a ray picks up from the
        relief is the path difference across the height of the profile,

        .. math::

            \phi(x) = \frac{2 \pi}{\lambda} h(x) (n_1 \cos \alpha - n_2 \cos \beta),

        where :math:`\alpha` and :math:`\beta` are the angles of incidence
        and diffraction and :math:`\lambda` is the free-space wavelength.
        A reflection grating is the case :math:`n_2 = -n_1`, for which the
        factor becomes :math:`n_1 (\cos \alpha + \cos \beta)`, twice the
        height times the cosine at normal incidence, and which falls with
        obliquity rather than rising. A transmission relief grating at
        normal incidence has the factor :math:`n_1 - n_2`, the index
        contrast.

        So the formulas of :cite:t:`Magnusson1978` are used with their
        normalized amplitude :math:`\gamma` replaced by

        .. math::

            \gamma = \frac{\pi h_1}{\lambda}
                \frac{|n_1 \cos \alpha - n_2 \cos \beta|}{2},

        where :math:`h_1` is the amplitude of the fundamental Fourier
        component of the groove profile, which each profile relates to its
        :attr:`depth`. Orders which are evanescent, for which no real
        :math:`\beta` exists, have zero efficiency.
        """

    def _gamma(
        self,
        depth: u.Quantity | na.AbstractScalar,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar,
        is_mirror: bool | na.AbstractScalar,
    ) -> na.AbstractScalar:
        r"""
        The normalized amplitude of the phase modulation,
        :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`,
        as defined in the notes of :meth:`efficiency`.

        :obj:`numpy.nan` for orders which are evanescent.

        Parameters
        ----------
        depth
            The amplitude of the fundamental Fourier component of the
            groove profile, :math:`h_1`.
        rays
            The light rays incident on the rulings, before diffraction.
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in, or :obj:`None` for that of the incident light.
        is_mirror
            Whether the rulings are on a reflective surface.
        """
        n1 = rays.index_refraction
        if index_refraction_new is None:
            n2 = n1
        else:
            n2 = index_refraction_new

        # the diffracted light of a mirror goes back into the incident
        # medium, which is the same as transmitting into its negative
        n2 = np.where(is_mirror, -n1, n2)

        direction = rays.direction
        cos_alpha = -(direction @ normal)

        # the sine of the diffracted angle, in the incident medium, is the
        # in-plane component of the effective incident direction
        direction_effective = self.incident_effective(rays, normal).direction
        sin_beta_1 = direction_effective - (direction_effective @ normal) * normal
        sin_beta_1 = sin_beta_1.length

        # Snell's law carries it into the diffracted medium
        cos_beta_squared = 1 - np.square(n1 * sin_beta_1 / n2)
        cos_beta = np.sqrt(np.where(cos_beta_squared < 0, np.nan, cos_beta_squared))

        obliquity = np.abs(n1 * cos_alpha - n2 * cos_beta) / 2

        result = np.pi * depth * obliquity / rays.wavelength
        result = na.as_named_array(result).to(u.dimensionless_unscaled)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class Rulings(
    AbstractRulings,
):
    """
    An idealized set of rulings which have perfect efficiency in all diffraction
    orders.
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float:
        return 1


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredRulings(
    AbstractRulings,
):
    """
    A set of rulings where the efficiency has been measured or calculated
    by an independent source.

    Examples
    --------

    Define rulings whose efficiency was calculated at three angles of
    incidence, and interpolate it between them.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # The wavelengths and angles of incidence of the calculation
        wavelength = na.linspace(120, 190, axis="wavelength", num=8) * u.nm
        angle = na.ScalarArray([10, 14, 18] * u.deg, axes="angle")

        # An efficiency which falls with wavelength and rises with angle
        efficiency = (
            0.35
            - 0.001 * (wavelength.to(u.nm).value - 120)
            + 0.002 * (angle.to(u.deg).value - 10)
        )

        # Define the rulings
        rulings = optika.rulings.MeasuredRulings(
            spacing=1 / (2200 / u.mm),
            diffraction_order=1,
            efficiency_measured=na.FunctionArray(
                inputs=na.SpectralDirectionalVectorArray(
                    wavelength=wavelength,
                    direction=angle,
                ),
                outputs=efficiency,
            ),
            axis_angle="angle",
        )

        # Evaluate the efficiency at 150 nm over a range of angles of incidence
        angle_rays = na.linspace(5, 23, axis="angle_rays", num=91) * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=150 * u.nm,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle_rays),
                y=0,
                z=np.cos(angle_rays),
            ),
        )
        efficiency_rays = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the interpolated efficiency against the calculation
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(angle_rays, efficiency_rays, ax=ax, label="interpolated");
            na.plt.scatter(
                angle,
                efficiency[dict(wavelength=3)],
                ax=ax,
                label="calculated",
            );
            ax.set_xlabel(f"angle of incidence ({ax.get_xlabel()})");
            ax.set_ylabel("efficiency at 150 nm");
            ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    efficiency_measured: na.FunctionArray[
        na.SpectralDirectionalVectorArray,
        na.AbstractScalar,
    ] = MISSING
    """
    A function array that maps wavelengths and incidence angles to the
    measured efficiency.

    See :attr:`axis_angle` for measurements at more than one angle of
    incidence.
    """

    axis_angle: None | str = None
    """
    The logical axis of :attr:`efficiency_measured` along which the angle
    of incidence varies.

    If :obj:`None`, the efficiency was measured at a single angle and is
    used at every angle of incidence.
    Otherwise, the directions of :attr:`efficiency_measured` must be angles
    of incidence, measured from the surface normal and increasing along
    this axis, and the efficiency is interpolated linearly in both
    wavelength and angle of incidence, holding the nearest measurement
    outside the measured range.
    Each angle may have its own wavelength samples.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.diffraction_order),
            optika._util._shape_efficiency_measured(
                measurement=self.efficiency_measured,
                axis_angle=self.axis_angle,
            ),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> na.AbstractScalar:

        return optika._util._interp_efficiency_measured(
            measurement=self.efficiency_measured,
            rays=rays,
            normal=normal,
            axis_angle=self.axis_angle,
        )


@dataclasses.dataclass(eq=False, repr=False)
class SinusoidalRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a sinusoidal wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of sinusoidal rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SinusoidalRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), sinusoidal rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = J_i^2(2 \gamma)

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`J_i(x)` is a Bessel function of the first kind,
        :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`
        is the normalized amplitude of the phase modulation, in which
        :math:`h_1` is the amplitude of the fundamental Fourier component
        of the groove profile, :math:`\lambda` is the free-space
        wavelength, :math:`n_1` and :math:`n_2` are the indices of
        refraction of the incident and diffracted media, and :math:`\alpha`
        and :math:`\beta` are the angles of incidence and diffraction.
        See the notes of :meth:`AbstractRulings.efficiency` for where this
        factor comes from.
        """

        d = self.depth
        i = self.diffraction_order

        gamma = self._gamma(
            depth=d,
            rays=rays,
            normal=normal,
            index_refraction_new=index_refraction_new,
            is_mirror=is_mirror,
        )

        result = np.square(scipy.special.jv(i, 2 * gamma))

        # orders which are evanescent carry no light
        result = np.where(np.isfinite(gamma), result, 0)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class SquareRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a square wave with a 50% duty cycle.

    Examples
    --------

    Compute the 1st-order groove efficiency of square rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SquareRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), square rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                \cos^2(\pi \gamma / 2) & i = 0 \\
                0 & i = \text{even} \\
                (2 / i \pi)^2 \sin^2 (\pi \gamma / 2) & i = \text{odd}, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`
        is the normalized amplitude of the phase modulation, in which
        :math:`h_1` is the amplitude of the fundamental Fourier component
        of the groove profile, :math:`\lambda` is the free-space
        wavelength, :math:`n_1` and :math:`n_2` are the indices of
        refraction of the incident and diffracted media, and :math:`\alpha`
        and :math:`\beta` are the angles of incidence and diffraction.
        See the notes of :meth:`AbstractRulings.efficiency` for where this
        factor comes from.
        """

        amplitude = np.pi / 4
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = self._gamma(
            depth=d,
            rays=rays,
            normal=normal,
            index_refraction_new=index_refraction_new,
            is_mirror=is_mirror,
        )

        result = np.where(
            i % 2 == 0,
            0,
            np.square(2 * np.sin(np.pi * gamma / 2 * u.rad) / (i * np.pi)),
        )
        result = np.where(
            i == 0,
            np.square(np.cos(np.pi * gamma / 2 * u.rad)),
            result,
        )

        # orders which are evanescent carry no light
        result = np.where(np.isfinite(gamma), result, 0)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class SawtoothRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a sawtooth wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of sawtooth rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SawtoothRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=-1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), sawtooth rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = [\pi (\gamma + i)]^{-2} \sin^2(\pi \gamma)

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`
        is the normalized amplitude of the phase modulation, in which
        :math:`h_1` is the amplitude of the fundamental Fourier component
        of the groove profile, :math:`\lambda` is the free-space
        wavelength, :math:`n_1` and :math:`n_2` are the indices of
        refraction of the incident and diffracted media, and :math:`\alpha`
        and :math:`\beta` are the angles of incidence and diffraction.
        See the notes of :meth:`AbstractRulings.efficiency` for where this
        factor comes from.
        """

        amplitude = np.pi / 2
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = self._gamma(
            depth=d,
            rays=rays,
            normal=normal,
            index_refraction_new=index_refraction_new,
            is_mirror=is_mirror,
        )

        # Since sin(pi gamma) = (-1)^i sin(pi (gamma + i)), this is the
        # squared sinc of gamma + i, which is finite when the profile is
        # a whole number of waves deep and the formula above is 0 / 0.
        result = np.square(_sinc(gamma + i))

        # orders which are evanescent carry no light
        result = np.where(np.isfinite(gamma), result, 0)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class TriangularRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a triangle wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of triangular rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.TriangularRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), triangular rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                \{\gamma / [(\pi \gamma / 2)^2 - i^2]\}^2 \sin^2 (\pi^2 \gamma / 4) & i = \text{even} \\
                \{\gamma / [(\pi \gamma / 2)^2 - i^2]\}^2 \cos^2 (\pi^2 \gamma / 4) & i = \text{odd}, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`
        is the normalized amplitude of the phase modulation, in which
        :math:`h_1` is the amplitude of the fundamental Fourier component
        of the groove profile, :math:`\lambda` is the free-space
        wavelength, :math:`n_1` and :math:`n_2` are the indices of
        refraction of the incident and diffracted media, and :math:`\alpha`
        and :math:`\beta` are the angles of incidence and diffraction.
        See the notes of :meth:`AbstractRulings.efficiency` for where this
        factor comes from.
        """

        amplitude = np.square(np.pi) / 8
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = self._gamma(
            depth=d,
            rays=rays,
            normal=normal,
            index_refraction_new=index_refraction_new,
            is_mirror=is_mirror,
        )

        # Writing beta = pi gamma / 2 and k = |i|, the trigonometric factor
        # for either parity is sin(pi (beta - k) / 2) up to sign, so the
        # formula above is the squared sinc of (beta - k) / 2 divided by
        # beta + k, which is finite when the profile is a whole number of
        # waves deep and the formula above is 0 / 0.
        beta = np.pi * gamma / 2
        k = np.abs(i)
        denominator = beta + k
        denominator_safe = np.where(denominator == 0, 1, denominator)

        result = np.pi * gamma * _sinc((beta - k) / 2) / (2 * denominator_safe)
        result = np.square(result)

        # zero depth in the zeroth order, where the limit is one
        result = np.where(denominator == 0, 1, result)

        # orders which are evanescent carry no light
        result = np.where(np.isfinite(gamma), result, 0)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class RectangularRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a rectangular wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of rectangular rulings with a groove
    density of 2500 grooves/mm, a groove depth of 15 nm, and a duty cycle of
    30 percent.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.RectangularRulings(
            spacing=1 / density,
            depth=depth,
            ratio_duty=0.3,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    ratio_duty: u.Quantity | na.AbstractScalar = MISSING
    """The duty cycle of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.ratio_duty),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
        index_refraction_new: None | float | na.AbstractScalar = None,
        is_mirror: bool | na.AbstractScalar = True,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        index_refraction_new
            The index of refraction of the medium the diffracted light
            propagates in. If :obj:`None`, it is taken to be the same as
            that of the incident light. Ignored if ``is_mirror`` is true.
        is_mirror
            Whether the rulings are on a reflective surface.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), rectangular rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                1 - [(2 a / \pi) - (a / \pi)^2] \sin^2 \left\{ \pi \gamma / [2 (1 - \cos a)]^{1/2} \right\} & i = 0 \\
                [2 / (i \pi)^2](1 - \cos i a) \sin^2 \left\{ \pi \gamma / [2 (1 - \cos a)]^{1/2} \right\} & i \ne 0, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order :math:`i`,
        :math:`a` :math:`(0 < a < 2 \pi)` is the duty cycle of the rectangular wave,
        :math:`\gamma = \pi h_1 |n_1 \cos \alpha - n_2 \cos \beta| / 2 \lambda`
        is the normalized amplitude of the phase modulation, in which
        :math:`h_1` is the amplitude of the fundamental Fourier component
        of the groove profile, :math:`\lambda` is the free-space
        wavelength, :math:`n_1` and :math:`n_2` are the indices of
        refraction of the incident and diffracted media, and :math:`\alpha`
        and :math:`\beta` are the angles of incidence and diffraction.
        See the notes of :meth:`AbstractRulings.efficiency` for where this
        factor comes from.
        """

        a = 2 * np.pi * self.ratio_duty
        amplitude = np.pi / (2 * np.sqrt(2 * (1 - np.cos(a))))
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = self._gamma(
            depth=d,
            rays=rays,
            normal=normal,
            index_refraction_new=index_refraction_new,
            is_mirror=is_mirror,
        )

        b = np.sin(np.pi * gamma / np.sqrt(2 * (1 - np.cos(a * u.rad))) * u.rad)
        b = np.square(b)

        result = np.where(
            i == 0,
            1 - ((2 * a / np.pi) - np.square(a / np.pi)) * b,
            (2 / np.square(i * np.pi)) * (1 - np.cos(i * a * u.rad)) * b,
        )

        # orders which are evanescent carry no light
        result = np.where(np.isfinite(gamma), result, 0)

        return result
