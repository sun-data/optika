import abc
import functools
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "energy_bandgap",
    "energy_electron_hole",
    "quantum_yield_ideal",
    "quantum_efficiency_effective",
    "AbstractImagingSensorMaterial",
    "AbstractCCDMaterial",
    "AbstractBackilluminatedCCDMaterial",
    "AbstractStern1994BackilluminatedCCDMaterial",
]

energy_bandgap = 1.12 * u.eV
"""the bandgap energy of silicon"""

energy_electron_hole = 3.65 * u.eV
"""
the high-energy limit of the energy required to create an electron-hole pair
in silicon at room temperature
"""


def quantum_yield_ideal(
    wavelength: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    Calculate the ideal quantum yield of a silicon detector for a given
    wavelength.

    Parameters
    ----------
    wavelength
        the wavelength of the incident photons

    Notes
    -----
    The quantum yield is the number of electron-hole pairs produced per photon.

    The ideal quantum yield is given in :cite:t:`Janesick2001` as:

    .. math::

        \text{QY}(\epsilon) = \begin{cases}
            0, & \epsilon < E_\text{g}\\
            1, &  E_\text{g} \leq \epsilon < E_\text{e-h} \\
            E_\text{e-h} / \epsilon, & E_\text{e-h} \leq \epsilon,
        \end{cases},

    where :math:`\epsilon` is the energy of the incident photon,
    :math:`E_\text{g} = 1.12\;\text{eV}` is the bandgap energy of silicon,
    and :math:`E_\text{e-h} = 3.65\;\text{eV}` is the energy required to
    generate 1 electron-hole pair in silicon at room temperature.

    Examples
    --------

    Plot the quantum yield vs wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths
        wavelength = na.geomspace(100, 100000, axis="wavelength", num=101) << u.AA

        # Compute the quantum yield
        qy = optika.sensors.quantum_yield_ideal(wavelength)

        # Plot the quantum yield vs wavelength
        fig, ax = plt.subplots()
        na.plt.plot(wavelength, qy, ax=ax);
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("quantum yield");
    """
    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    result = energy / energy_electron_hole
    result = np.where(energy > energy_electron_hole, result, 1)
    result = np.where(energy > energy_bandgap, result, 0)

    return result


def quantum_efficiency_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
    formula_oxide: str = "SiO2",
    n_ambient: complex | na.AbstractScalar = 1,
    n_substrate: None | complex | na.AbstractScalar = None,
    normal: None | na.AbstractCartesian3dVectorArray = None,
) -> na.AbstractScalar:
    r"""
    Calculate the effective quantum efficiency of a back-illuminated detector.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light.
    direction
        The propagation direction of the incident light.
    thickness_oxide
        The thickness of the oxide layer on the back surface of the sensor.
    thickness_implant
        The thickness of the implant layer.
    thickness_substrate
        The thickness of the silicon substrate.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
    formula_oxide
        The chemical formula of the oxide layer.
        Default is silicon dioxide.
    n_ambient
        Optional complex refractive index of the ambient propagation medium.
        Default is :math:`1`, the refractive index of vacuum.
    n_substrate
        Optional complex refractive index of the implant region and substrate.
        If :obj:`None`, then the refractive index of silicon is used.
    normal
        The vector perpendicular to the surface of the sensor.
        If :obj:`None`, then the normal is assumed to be :math:`-\hat{z}`

    Examples
    --------
    Reproduce Figure 12 from :cite:t:`Stern1994`, the modeled quantum efficiency
    of a Tektronix TK512CB :math:`512 \times 512` pixel backilluminated CCD.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths with which to sample the EQE
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Assume normal incidence
        direction = na.Cartesian3dVectorArray(0, 0, 1)

        # Store the fit parameters found in Stern 1994
        thickness_oxide = 50 * u.AA
        thickness_implant = 2317 * u.AA
        thickness_substrate = 7 * u.um
        cce_backsurface = 0.21

        # Compute the effective quantum efficiency
        eqe = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            cce_backsurface=cce_backsurface,
        )

        # Compute the maximum theoretical quantum efficiency
        eqe_max = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            cce_backsurface=1,
        )

        # Plot the effective and maximum quantum efficiency
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            eqe,
            ax=ax,
            label="effective quantum efficiency",
        );
        na.plt.plot(
            wavelength,
            eqe_max,
            ax=ax,
            label="maximum quantum efficiency",
        );
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("efficiency");
        ax.legend();

    Plot the EQE as a function of wavelength for normal and oblique incidence

    .. jupyter-execute::

        angle = na.linspace(0, 30, axis="angle", num=2) * u.deg
        direction = na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        )

        eqe = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            cce_backsurface=cce_backsurface,
        )

        # Plot the results
        fig, ax = plt.subplots(constrained_layout=True)
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            eqe,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("efficiency");
        ax.legend();

    Notes
    -----
    Our goal is to recover Equation 11 in :cite:t:`Stern1994`, along wtih the
    correction for photons lost by transmission through the entire CCD substrate.
    From inspecting Equations 6 and 9 in :cite:t:`Stern1994`,
    we can see that the effective quantum efficiency is:

    .. math::
        :label: eqe-definition

        \text{EQE} = T_\lambda \int_0^\infty \alpha \eta(x) e^{-\alpha x} \; dx

    where :math:`T_\lambda` is the net transmission of photons through the backsurface
    oxide layer (accounting for both absorption and reflections) calculated using
    :func:`optika.materials.multilayer_efficiency`,
    :math:`\alpha` is the absorption coefficient of silicon,
    :math:`x` is the distance from the backsurface,
    and :math:`\eta(x)` is the differential charge collection efficiency (CCE).

    :cite:t:`Stern1994` assumes that the differential CCE takes the following
    linear form,

    .. math::
        :label: differential-cce

        \eta(x) = \begin{cases}
            \eta_0 + (1 - \eta_0) x / W, & 0 < x < W \\
            1, & W < x < D, \\
            0, & D < x
        \end{cases}

    where :math:`\eta_0` is the differential CCE at the backsurface,
    :math:`W` is the thickness of the implant region,
    and :math:`D` is the total thickness of the silicon substrate.

    Plugging Equation :eq:`differential-cce` into Equation :eq:`eqe-definition`
    and integrating yields

    .. math::
        :label: eqe

        \text{EQE} &= \alpha T_\lambda \left\{
                        \int_0^W \left[ \eta_0 + \left( \frac{1 - \eta_0}{W} \right) x \right] e^{-\alpha x} \; dx
                        + \int_W^D e^{-\alpha x} \; dx
        \right\} \\
        &= \alpha T_\lambda \left\{
            \eta_0 \int_0^W e^{-\alpha x} \; dx
            + \left( \frac{1 - \eta_0}{W} \right) \int_0^W x e^{-\alpha x} \; dx
            + \int_W^D e^{-\alpha x} \; dx
        \right\} \\
        &= \alpha T_\lambda \left\{
            -\left[ \frac{\eta_0}{\alpha} e^{-\alpha x} \right|_0^W
            - \left( \frac{1 - \eta_0}{W} \right) \left[ \left( \frac{\alpha x + 1}{\alpha^2} \right) e^{-\alpha x} \right|_0^W
            - \left[ \frac{1}{\alpha} e^{-\alpha x} \right|_W^D
        \right\} \\
        &= T_\lambda \left\{
            - \left[ \eta_0 (e^{-\alpha W} - 1) \right]
            - \left( \frac{1 - \eta_0}{\alpha W} \right) \left[ (\alpha W + 1) e^{-\alpha W} - 1 \right]
            - \left[ e^{-\alpha D} - e^{-\alpha W} \right]
        \right\} \\
        &= T_\lambda \left\{
            - \eta_0 e^{-\alpha W}
            + \eta_0
            - e^{-\alpha W}
            + \eta_0 e^{-\alpha W}
            + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
            - e^{-\alpha D}
            + e^{-\alpha W}
        \right\} \\
        &= T_\lambda \left\{
            \eta_0
            + \left( \frac{1 - \eta_0}{\alpha W} \right) (1 - e^{-\alpha W})
            - e^{-\alpha D}
        \right\} \\

    Equation :eq:`eqe` is equivalent to Equation 11 in :cite:t:`Stern1994`,
    with the addition of an :math:`e^{-\alpha W}-e^{-\alpha D}` term which represents photons
    that traveled all the way through the silicon substrate without interacting.

    Equation :eq:`eqe` is only valid for normally-incident light.
    We can generalize it to obliquely-incident light by making the substitution

    .. math::
        :label: x-oblique

        x \rightarrow \frac{x}{\cos \theta}

    where :math:`\theta` is the angle between the propagation direction
    inside the silicon substrate and the normal vector.

    Substituting :eq:`x-oblique` into Equation :eq:`eqe` and solving yields

    .. math::
        :label: eqe-oblique

        \text{EQE} = T_\lambda \left\{
            \eta_0
            + \left( \frac{1 - \eta_0}{\alpha W \sec \theta} \right) (1 - e^{-\alpha W \sec \theta})
            - e^{-\alpha D \sec \theta}
        \right\} \\

    """

    if n_substrate is None:
        substrate = optika.chemicals.Chemical("Si")
        n_substrate = substrate.n(wavelength)

    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    n = optika.chemicals.Chemical(formula_oxide).n(wavelength)
    n = n.broadcast_to(na.broadcast_shapes(n.shape, dict(_layer=1)))

    reflectivity, transmissivity = optika.materials.multilayer_efficiency(
        n=n,
        thickness=na.stack([thickness_oxide], axis="_layer"),
        axis="_layer",
        wavelength_ambient=wavelength,
        direction_ambient=direction,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        normal=normal,
    )

    wavenumber_substrate = np.imag(n_substrate)
    absorption_substrate = 4 * np.pi * wavenumber_substrate / wavelength

    direction_substrate = optika.materials.snells_law(
        wavelength=wavelength,
        direction=direction,
        index_refraction=np.real(n_ambient),
        index_refraction_new=np.real(n_substrate),
        normal=normal,
    )

    cos_theta = -direction_substrate @ normal

    z0 = absorption_substrate * thickness_implant / cos_theta
    exp_z0 = np.exp(-z0)

    term_1 = cce_backsurface
    term_2 = ((1 - cce_backsurface) / z0) * (1 - exp_z0)
    term_3 = -np.exp(-absorption_substrate * thickness_substrate / cos_theta)

    result = transmissivity * (term_1 + term_2 + term_3)

    return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImagingSensorMaterial(
    optika.materials.AbstractMaterial,
):
    """
    An interface representing the light-sensitive material of an imaging sensor.
    """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractCCDMaterial(
    AbstractImagingSensorMaterial,
):
    """
    An interface representing the light-sensitive material of a CCD sensor.
    """

    @property
    def transformation(self) -> None:
        return None

    @functools.cached_property
    def _chemical(self) -> optika.chemicals.Chemical:
        return optika.chemicals.Chemical("Si")

    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        result = self._chemical.index_refraction(rays.wavelength)
        return result

    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        result = self._chemical.wavenumber(rays.wavelength)
        result = 4 * np.pi * result / rays.wavelength
        return result

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class AbstractBackilluminatedCCDMaterial(
    AbstractCCDMaterial,
):
    """
    An interface representing the light-sensitive material of a backilluminated
    CCD sensor.
    """

    @property
    @abc.abstractmethod
    def thickness_oxide(self) -> u.Quantity | na.AbstractScalar:
        """
        The thickness of the oxide layer on the back surface of the CCD sensor.
        """

    @property
    @abc.abstractmethod
    def thickness_implant(self) -> u.Quantity | na.AbstractScalar:
        """
        The thickness of the implant layer of the CCD sensor.
        """

    @property
    @abc.abstractmethod
    def thickness_substrate(self) -> u.Quantity | na.AbstractScalar:
        """the thickness of the entire CCD silicon substrate"""

    @property
    @abc.abstractmethod
    def cce_backsurface(self) -> float | na.AbstractScalar:
        """
        The charge collection efficiency on the backsurface of the CCD sensor.
        """

    def quantum_yield_ideal(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> u.Quantity | na.AbstractScalar:
        """
        Compute the ideal quantum yield of this CCD sensor material using
        :func:`optika.sensors.quantum_yield_ideal`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light
        """
        return quantum_yield_ideal(wavelength)

    def quantum_efficiency_effective(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the effective quantum efficiency of this CCD material using
        :func:`optika.sensors.quantum_efficiency_effective`.

        Parameters
        ----------
        rays
            The light rays incident on the CCD surface
        normal
            The vector perpendicular to the surface of the CCD.
        """
        k_ambient = rays.attenuation * rays.wavelength / (4 * np.pi)
        n_ambient = rays.index_refraction + k_ambient * 1j

        k_substrate = self.attenuation(rays) * rays.wavelength / (4 * np.pi)
        n_substrate = self.index_refraction(rays) + k_substrate * 1j

        return quantum_efficiency_effective(
            wavelength=rays.wavelength,
            direction=rays.direction,
            thickness_oxide=self.thickness_oxide,
            thickness_implant=self.thickness_implant,
            thickness_substrate=self.thickness_substrate,
            cce_backsurface=self.cce_backsurface,
            n_ambient=n_ambient,
            n_substrate=n_substrate,
            normal=normal,
        )

    def transmissivity(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return self.quantum_efficiency_effective(
            rays=rays,
            normal=normal,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractStern1994BackilluminatedCCDMaterial(
    AbstractBackilluminatedCCDMaterial,
):
    """
    A CCD material that is uses the method of :cite:t:`Stern1994` to compute
    the quantum efficiency.
    """

    @property
    @abc.abstractmethod
    def quantum_efficiency_measured(self) -> na.FunctionArray:
        """
        The measured quantum efficiency that will be fit by the function
        :func:`optika.sensors.quantum_efficiency_effective`.
        """

    @functools.cached_property
    def _quantum_efficiency_fit(self) -> dict[str, float | u.Quantity]:
        qe_measured = self.quantum_efficiency_measured

        unit_thickness_oxide = u.AA
        unit_thickness_implant = u.AA
        unit_thickness_substrate = u.um
        unit_cce_backsurface = u.dimensionless_unscaled

        def eqe_rms_difference(x: tuple[float, float, float, float]):
            (
                thickness_oxide,
                thickness_implant,
                thickness_substrate,
                cce_backsurface,
            ) = x
            qe_fit = quantum_efficiency_effective(
                wavelength=qe_measured.inputs,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
                thickness_oxide=thickness_oxide << unit_thickness_oxide,
                thickness_implant=thickness_implant << unit_thickness_implant,
                thickness_substrate=thickness_substrate << unit_thickness_substrate,
                cce_backsurface=cce_backsurface << unit_cce_backsurface,
            )

            return np.sqrt(np.mean(np.square(qe_measured.outputs - qe_fit))).ndarray

        thickness_oxide_guess = 50 * u.AA
        thickness_implant_guess = 2317 * u.AA
        thickness_substrate_guess = 7 * u.um
        cce_backsurface_guess = 0.21 * u.dimensionless_unscaled

        fit = scipy.optimize.minimize(
            fun=eqe_rms_difference,
            x0=[
                thickness_oxide_guess.to_value(unit_thickness_oxide),
                thickness_implant_guess.to_value(unit_thickness_implant),
                thickness_substrate_guess.to_value(unit_thickness_substrate),
                cce_backsurface_guess.to_value(unit_cce_backsurface),
            ],
            method="nelder-mead",
        )

        thickness_oxide, thickness_implant, thickness_substrate, cce_backsurface = fit.x
        thickness_oxide = thickness_oxide << unit_thickness_oxide
        thickness_implant = thickness_implant << unit_thickness_implant
        thickness_substrate = thickness_substrate << unit_thickness_substrate
        cce_backsurface = cce_backsurface << unit_cce_backsurface

        return dict(
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            cce_backsurface=cce_backsurface,
        )

    @property
    def thickness_oxide(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_oxide"]

    @property
    def thickness_implant(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_implant"]

    @property
    def thickness_substrate(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_substrate"]

    @property
    def cce_backsurface(self) -> float:
        return self._quantum_efficiency_fit["cce_backsurface"]
