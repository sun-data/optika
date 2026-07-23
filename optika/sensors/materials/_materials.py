from typing import Literal
from typing_extensions import Self
import abc
import functools
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
from ._stern_1994 import (
    _thickness_oxide,
    _thickness_implant,
    _thickness_substrate,
    _width_pixel,
    _cce_backsurface,
)
from ._ramanathan_2020 import (
    energy_bandgap,
    energy_pair,
    energy_pair_inf,
    quantum_yield_ideal,
    fano_factor,
    fano_factor_inf,
    electrons_measured,
)
from .depletion import AbstractDepletionModel

__all__ = [
    "energy_bandgap",
    "energy_pair",
    "energy_pair_inf",
    "quantum_yield_ideal",
    "fano_factor",
    "fano_factor_inf",
    "absorption_effective",
    "transmittance",
    "absorbance",
    "charge_collection_efficiency",
    "quantum_efficiency_effective",
    "probability_measurement",
    "electrons_measured",
    "electrons_measured_approx",
    "signal",
    "vmr_signal",
    "AbstractSensorMaterial",
    "IdealSensorMaterial",
    "AbstractSiliconSensorMaterial",
    "AbstractBackIlluminatedSiliconSensorMaterial",
    "BackIlluminatedSiliconSensorMaterial",
]


def absorption_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    n_substrate: complex | na.AbstractScalar,
    direction_substrate: complex | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The absorption coefficient per unit perpendicular depth for a wave
    refracted into the (absorbing) light-sensitive region at oblique incidence.

    This is the rigorous generalization of :math:`4 \pi \, \text{Im}(n) / \lambda`
    to oblique incidence: it uses the complex refracted cosine
    ``direction_substrate`` so that the geometric :math:`\sec\theta` path-length
    factor is no longer needed downstream. Reduces to the normal-incidence
    coefficient when ``direction_substrate`` is 1.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    n_substrate
        The complex index of refraction of the light-sensitive region.
    direction_substrate
        The (generally complex) cosine of the refracted angle inside the
        light-sensitive region, e.g. from :func:`optika.materials.snells_law_scalar`.

    Notes
    -----
    For a wave refracted from a transparent ambient medium into an absorbing
    medium, the planes of constant amplitude remain parallel to the interface,
    so the intensity attenuates along the surface normal with coefficient

    .. math::

        \alpha_\perp = \frac{4 \pi}{\lambda} \, \text{Im}(n_2 \cos\theta_2),

    where :math:`n_2` is the complex index of the light-sensitive region and
    :math:`\cos\theta_2` is its (complex) refracted cosine.
    """
    return 4 * np.pi * np.imag(n_substrate * direction_substrate) / wavelength


def transmittance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: complex | na.AbstractScalar = 1,
    thickness_oxide: u.Quantity | na.AbstractScalar = _thickness_oxide,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    chemical_oxide: str | optika.chemicals.AbstractChemical = "SiO2",
    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    roughness_oxide: u.Quantity | na.AbstractScalar = 0 * u.nm,
    roughness_substrate: u.Quantity | na.AbstractScalar = 0 * u.nm,
) -> optika.vectors.PolarizationVectorArray:
    """
    The fraction of incident energy transmitted through the oxide layer into
    the light-sensitive material.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The cosine of the incidence angle.
        Default is normal incidence.
    n
        The complex index of refraction in the ambient medium.
    thickness_oxide
        The thickness of the oxide layer on the illuminated surface of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    thickness_substrate
        The thickness of the light-sensitive substrate layer.
        Default is the value given in :cite:t:`Stern1994`.
    chemical_oxide
        The chemical formula of the oxide layer on the illuminated surface of the sensor.
        Default is silicon dioxide.
    chemical_substrate
        The chemical formula of the light-sensitive portion of the sensor.
        Default is silicon.
    roughness_oxide
        The RMS roughness the oxide layer surface.
    roughness_substrate
        The RMS roughness of the substrate surface.

    Examples
    --------

    Plot the transmittance as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the transmittance vs wavelength
        transmittance = optika.sensors.transmittance(
            wavelength=wavelength,
        )

        # Plot the average transmittance vs. wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            transmittance.average,
            ax=ax,
        );
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("incident energy fraction");
    """

    if not isinstance(chemical_oxide, optika.chemicals.AbstractChemical):
        chemical_oxide = optika.chemicals.Chemical(chemical_oxide)

    if not isinstance(chemical_substrate, optika.chemicals.AbstractChemical):
        chemical_substrate = optika.chemicals.Chemical(chemical_substrate)

    reflection, transmission = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=[
            optika.materials.Layer(
                chemical=chemical_oxide,
                thickness=thickness_oxide,
                interface=optika.materials.profiles.ErfInterfaceProfile(
                    width=roughness_oxide,
                ),
            ),
        ],
        substrate=optika.materials.Layer(
            chemical=chemical_substrate,
            thickness=thickness_substrate,
            interface=optika.materials.profiles.ErfInterfaceProfile(
                width=roughness_substrate,
            ),
        ),
    )

    return transmission


def absorbance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: complex | na.AbstractScalar = 1,
    thickness_oxide: u.Quantity | na.AbstractScalar = _thickness_oxide,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    chemical_oxide: str | optika.chemicals.AbstractChemical = "SiO2",
    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    roughness_oxide: u.Quantity | na.AbstractScalar = 0 * u.nm,
    roughness_substrate: u.Quantity | na.AbstractScalar = 0 * u.nm,
    method: Literal["exact", "Beer-Lambert"] = "Beer-Lambert",
) -> optika.vectors.PolarizationVectorArray:
    """
    The fraction of incident energy absorbed by the light-sensitive
    region of the sensor

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The cosine of the incidence angle.
        Default is normal incidence.
    n
        The complex index of refraction in the ambient medium.
    thickness_oxide
        The thickness of the oxide layer on the illuminated surface of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    thickness_substrate
        The thickness of the light-sensitive substrate layer.
        Default is the value given in :cite:t:`Stern1994`.
    chemical_oxide
        The chemical formula of the oxide layer on the illuminated surface of the sensor.
        Default is silicon dioxide.
    chemical_substrate
        The chemical formula of the light-sensitive portion of the sensor.
        Default is silicon.
    roughness_oxide
        The RMS roughness the oxide layer surface.
    roughness_substrate
        The RMS roughness of the substrate surface.
    method
        The method to use to compute the absorbance.
        If ``exact``, this method allows thin-film interference effects
        inside the light-sensitive region.
        If ``Beer-Lambert``, this method assumes no interference effects.
        These methods only differ in the infrared, where the wavelength is
        commensurate with the thickness of the light-sensitive region.

    Examples
    --------

    Plot the absorbance as a function of wavelength and compare it to the
    transmittance.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the transmittance vs wavelength
        transmittance = optika.sensors.transmittance(
            wavelength=wavelength,
        )

        # Compute the absorbance vs wavelength
        absorbance_exact = optika.sensors.absorbance(
            wavelength=wavelength,
            method="exact",
        )

        absorbance_beer = optika.sensors.absorbance(
            wavelength=wavelength,
            method="Beer-Lambert",
        )

        # Plot the average absorbance vs. wavelength
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            transmittance.average,
            ax=ax,
            label="transmittance",
        );
        na.plt.plot(
            wavelength,
            absorbance_exact.average,
            ax=ax,
            label="exact absorbance",
        );
        na.plt.plot(
            wavelength,
            absorbance_beer.average,
            ax=ax,
            label="Beer-Lambert absorbance",
        );
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("incident energy fraction");
        ax.legend();
    """
    if not isinstance(chemical_oxide, optika.chemicals.AbstractChemical):
        chemical_oxide = optika.chemicals.Chemical(chemical_oxide)

    if not isinstance(chemical_substrate, optika.chemicals.AbstractChemical):
        chemical_substrate = optika.chemicals.Chemical(chemical_substrate)

    if method == "exact":

        result = optika.materials.layer_absorbance(
            index=1,
            wavelength=wavelength,
            direction=direction,
            n=n,
            layers=[
                optika.materials.Layer(
                    chemical=chemical_oxide,
                    thickness=thickness_oxide,
                    interface=optika.materials.profiles.ErfInterfaceProfile(
                        width=roughness_oxide,
                    ),
                ),
                optika.materials.Layer(
                    chemical=chemical_substrate,
                    thickness=thickness_substrate,
                    interface=optika.materials.profiles.ErfInterfaceProfile(
                        width=roughness_substrate,
                    ),
                ),
            ],
        )

    elif method == "Beer-Lambert":

        _transmittance = transmittance(
            wavelength=wavelength,
            direction=direction,
            n=n,
            thickness_oxide=thickness_oxide,
            thickness_substrate=thickness_substrate,
            chemical_oxide=chemical_oxide,
            chemical_substrate=chemical_substrate,
            roughness_oxide=roughness_oxide,
            roughness_substrate=roughness_substrate,
        )

        n_substrate = chemical_substrate.n(wavelength)

        direction_substrate = optika.materials.snells_law_scalar(
            cos_incidence=direction,
            index_refraction=n,
            index_refraction_new=n_substrate,
        )

        absorption = absorption_effective(
            wavelength=wavelength,
            n_substrate=n_substrate,
            direction_substrate=direction_substrate,
        )

        _transmittance_total = _transmittance * np.exp(
            -absorption * thickness_substrate
        )

        result = _transmittance - _transmittance_total

    else:  # pragma: nocover
        raise ValueError(f"Method {method} not implemented")

    return np.real(result)


def charge_collection_efficiency(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
) -> na.AbstractScalar:
    r"""
    Compute the average charge collection efficiency using the differential
    charge collection efficiency profile described in :cite:t:`Stern1994`.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive material per unit
        perpendicular depth.
        For oblique incidence, supply the effective coefficient from
        :func:`absorption_effective`, which folds in the refracted angle, so no
        separate angle argument is needed.
    thickness_implant
        The thickness of the implant layer, the layer where recombination can
        occur.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.

    Examples
    --------

    Plot the charge collection efficiency as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the absorption coefficient for silicon
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

        # Compute the CCE vs wavelength
        cce = optika.sensors.charge_collection_efficiency(
            absorption=absorption,
        )

        # Plot the effective and maximum quantum efficiency
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            cce,
            ax=ax,
        );
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("charge collection efficiency");

    Notes
    -----

    The charge collection efficiency is the fraction of photoelectrons that
    are measured by the sensor :cite:p:`Janesick2001`,
    and is an important component of the quantum efficiency of the sensor

    In :cite:t:`Stern1994`, the authors define a differential charge collection
    efficiency, :math:`\eta(z)`, which is the probability that a photoelectron
    resulting from a photon absorbed at a depth :math:`z` will be measured by
    the sensor.
    In principle, :math:`\eta(z)` depends on the exact implant profile on the
    backsurface of the sensor, however :cite:t:`Stern1994` and :cite:t:`Boerner2012`
    have shown that a piecewise-linear approximation of :math:`\eta(z)`,

    .. math::
        :label: differential-cce

        \eta(z) = \begin{cases}
            \eta_0 + (1 - \eta_0) z / W, & 0 < z < W \\
            1, & W < z < D,
        \end{cases}

    is sufficient, given the uncertainties in the optical constants involved.

    The total charge collection efficiency is then the average value of
    :math:`\eta(z)` weighted by the probability of absorbing a photon at a
    depth :math:`z`,

    .. math::
        :label: cce-definition

        \text{CCE}(\lambda) = \frac{\int_0^\infty \eta(z) e^{-\alpha z} \, dz}
                               {\int_0^\infty e^{-\alpha z} \, dz}.

    Plugging Equation :eq:`differential-cce` into Equation :eq:`cce-definition`
    and integrating yields

    .. math::
        :label: cce

        \text{CCE}(\lambda) = \eta_0 + \left( \frac{1 - \eta_0}{\alpha W} \right)(1 - e^{-\alpha W}).

    Equation :eq:`cce` is equivalent to the term in curly braces of Equation 11 in :cite:t:`Stern1994`,
    with the addition of an :math:`e^{-\alpha W}` term which represents photons
    absorbed inside the epitaxial layer but outside the implant layer.

    In Equation :eq:`cce`, :math:`\alpha` and :math:`z` are measured along the
    surface normal. Oblique incidence is therefore handled entirely by the
    absorption coefficient: passing the perpendicular-depth coefficient
    :math:`\alpha_\perp` from :func:`absorption_effective` (which folds in the
    refracted angle, and reduces to :math:`\alpha` at normal incidence) makes
    Equation :eq:`cce` valid at any incidence angle without further modification.
    """
    z0 = absorption * thickness_implant
    exp_z0 = np.exp(-z0)

    term_1 = cce_backsurface
    term_2 = ((1 - cce_backsurface) / z0) * (1 - exp_z0)

    return term_1 + term_2


def quantum_efficiency_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: complex | na.AbstractScalar = 1,
    thickness_oxide: u.Quantity | na.AbstractScalar = _thickness_oxide,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    chemical_oxide: str | optika.chemicals.AbstractChemical = "SiO2",
    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    roughness_oxide: u.Quantity | na.AbstractScalar = 0 * u.nm,
    roughness_substrate: u.Quantity | na.AbstractScalar = 0 * u.nm,
) -> na.AbstractScalar:
    r"""
    Calculate the effective quantum efficiency of a backilluminated detector.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The cosine of the incidence angle.
        Default is normal incidence.
    n
        The complex index of refraction of the ambient medium.
    thickness_oxide
        The thickness of the oxide layer on the back surface of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    thickness_implant
        The thickness of the implant layer.
        Default is the value given in :cite:t:`Stern1994`.
    thickness_substrate
        The thickness of the silicon substrate.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    chemical_oxide
        The chemical composition of the oxide layer.
        The default is to assume the oxide layer is silicon dioxide.
    chemical_substrate
        Optional complex refractive index of the implant region and substrate.
        The default is to assume the substrate is made from silicon.
    roughness_oxide
        The RMS roughness the oxide layer surface.
    roughness_substrate
        The RMS roughness of the substrate surface.

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

        # Compute the effective quantum efficiency
        eqe = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
        )

        # Compute the maximum theoretical quantum efficiency
        eqe_max = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
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

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths with which to sample the EQE
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Define the cosines of the incidence angles
        angle = na.linspace(0, 30, axis="angle", num=2) * u.deg
        direction = np.cos(angle)

        eqe = optika.sensors.quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
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
    :cite:t:`Stern1994` defines the effective quantum efficiency as

    .. math::
        :label: eqe

        \text{EQE}(\lambda) = A(\lambda) \times \text{CCE}(\lambda),

    where :math:`A(\lambda)` is the absorbtivity of the epitaxial silicon layer
    for a given wavelength :math:`\lambda`,
    and :math:`\text{CCE}(\lambda)` is the charge collection efficiency
    (computed by :func:`charge_collection_efficiency`).
    """

    if not isinstance(chemical_oxide, optika.chemicals.AbstractChemical):
        chemical_oxide = optika.chemicals.Chemical(chemical_oxide)

    if not isinstance(chemical_substrate, optika.chemicals.AbstractChemical):
        chemical_substrate = optika.chemicals.Chemical(chemical_substrate)

    absorbance_substrate = absorbance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
        chemical_oxide=chemical_oxide,
        chemical_substrate=chemical_substrate,
        roughness_oxide=roughness_oxide,
        roughness_substrate=roughness_substrate,
    )

    n_substrate = chemical_substrate.n(wavelength)

    direction_substrate = optika.materials.snells_law_scalar(
        cos_incidence=direction,
        index_refraction=n,
        index_refraction_new=n_substrate,
    )

    absorption_substrate = absorption_effective(
        wavelength=wavelength,
        n_substrate=n_substrate,
        direction_substrate=direction_substrate,
    )

    cce = charge_collection_efficiency(
        absorption=absorption_substrate,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
    )

    result = absorbance_substrate.average * cce

    return result


def probability_measurement(
    iqy: u.Quantity | na.AbstractScalar = 1 * u.electron / u.photon,
    cce: float | na.AbstractScalar = 1,
) -> na.AbstractScalar:
    r"""
    The probability that a photon absorbed in the epitaxial silicon layer results
    in at least one photoelectron measured by the sensor.

    For most of the electromagnetic spectrum, this quantity is nearly unity,
    but in the ultraviolet, there is a significant chance that all the
    photoelectrons associated with a photon recombine before being measured.

    Parameters
    ----------
    iqy
        The ideal quantum yield of the sensor in electrons per photon,
        calculated by :func:`quantum_yield_ideal`.
    cce
        The charge collection efficiency of the detector computed using
        :func:`charge_collection_efficiency`.

    Examples
    --------

    Plot the probability of measuring an absorbed photon vs the charge collection
    efficiency

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the ideal quantum yield of silicon for these wavelengths
        iqy = optika.sensors.quantum_yield_ideal(wavelength)

        # Compute the charge collection efficiency for each wavelength
        cce = optika.sensors.charge_collection_efficiency(
            absorption=optika.chemicals.Chemical("Si").absorption(wavelength),
        )

        # Compute the probability of measuring an absorbed photon
        # vs the charge collection efficiency
        p_m = optika.sensors.probability_measurement(iqy, cce)

        # Plot the results
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                wavelength,
                cce,
                ax=ax,
                label="charge collection efficiency",
            )
            na.plt.plot(
                wavelength,
                p_m,
                ax=ax,
                label="probability of measurement",
            )
            ax.set_xscale("log");
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})");
            ax.set_ylabel("probability");
            ax.legend();

    Notes
    -----

    The probability that `all` the electrons recombine before
    being measured is

    .. math::

        P_r = (1 - \text{CCE})^\text{IQY}

    Where :math:`\text{CCE}` is the charge collection efficiency,
    and :math:`\text{IQY}` is the ideal quantum yield of the sensor.
    So then the probability of a photon being measured is just the compliment
    of :math:`P_r`,

    .. math::

        P_m = 1 - P_r.
    """
    iqy = iqy.to(u.electron / u.photon).value
    p_r = (1 - cce) ** iqy
    p_m = 1 - p_r
    return p_m


def _discrete_gamma(
    mean: float | na.ScalarArray,
    vmr: float | na.ScalarArray,
    shape_random: None | dict[str, int] = None,
) -> na.ScalarArray:
    x = na.random.gamma(
        shape=mean / vmr,
        scale=vmr,
        shape_random=shape_random,
    )

    x = np.where(
        vmr != 0,
        x,
        mean,
    )

    unit_x = x.unit
    if unit_x is not None:
        x = x.value

    x_frac, x_int = np.modf(x)
    x_frac = na.random.binomial(
        n=1,
        p=x_frac,
        shape_random=shape_random,
    )
    x = x_int + x_frac

    if unit_x is not None:
        x = x << unit_x

    return x


_fano_factor = fano_factor


def electrons_measured_approx(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    absorption: None | u.Quantity | na.AbstractScalar = None,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
    iqy: None | u.Quantity | na.AbstractScalar = None,
    fano_factor: None | u.Quantity | na.AbstractScalar = None,
    shape_random: None | dict[str, int] = None,
) -> na.AbstractScalar:
    r"""
    A random sample from an approximate distribution of measured electrons
    given the number of photons absorbed by the light-sensitive layer
    of the sensor.

    This function accounts for both Fano noise and recombination noise due to
    partial-charge collection.

    Parameters
    ----------
    photons_absorbed
        The number of photons absorbed by the light-sensitive layer
        of the sensor.
    wavelength
        The vacuum wavelength of the absorbed photons.
    absorption
        The absorption coefficient of silicon per unit perpendicular depth.
        For oblique incidence, supply the effective coefficient from
        :func:`optika.sensors.absorption_effective`, which folds in the
        refracted angle, so no separate angle argument is needed.
    thickness_implant
        The thickness of the implant layer, where partial-charge collection occurs.
    thickness_substrate
        The thickness of the entire light-sensitive region of the device.
        The absorbed photons are distributed within this region, which sets
        the fraction that land in the implant layer.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
    temperature
        The temperature of the silicon detector.
        Default is room temperature.
    iqy
        The ideal quantum yield of the sensor in electrons per photon.
        If :obj:`None` (the default), the result of :func:`quantum_yield_ideal`
        is used.
    fano_factor
        The `Fano factor <https://en.wikipedia.org/wiki/Fano_factor>`_
        (ratio of the variance to the mean) of the Fano noise for this
        sensor material in units of electrons per photon.
        If :obj:`None` (the default), the result of :func:`fano_factor`
        is used.
    shape_random
        Additional shape used to specify the number of samples to draw.

    Examples
    --------

    Plot the energy spectrum of twenty 6 keV photons emitted from an Fe-55
    radioactive source and compare it to the exact spectrum

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the number of experiments to perform
        num_experiments = 100000

        # Define the expected number of photons
        # for each experiment
        photons_absorbed = (20 * u.photon).astype(int)

        # Define the wavelength at which to sample the distribution
        wavelength = 5.9 * u.keV
        wavelength = wavelength.to(u.AA, equivalencies=u.spectral())

        # Compute the actual number of electrons measured for each experiment
        electrons_exact = optika.sensors.electrons_measured(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            shape_random=dict(experiment=num_experiments),
        )

        # Compute the approximate number of electrons measured for each experiment
        electrons_approx = optika.sensors.electrons_measured_approx(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            shape_random=dict(experiment=num_experiments),
        )

        # Define the histogram bins
        step = 10
        bins = na.arange(
            electrons_exact.value.min()-step/2,
            electrons_exact.value.max()+step/2,
            step=step,
            axis="bin",
        ) * u.electron

        # Compute a histogram of exact energy spectrum
        hist_exact = na.histogram(
            electrons_exact,
            bins=bins,
            axis="experiment",
        )

        # Compute a histogram of approximate energy spectrum
        hist_approx = na.histogram(
            electrons_approx,
            bins=bins,
            axis="experiment",
        )

        # Plot the histogram
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.stairs(
                hist_exact.inputs,
                hist_exact.outputs,
                ax=ax,
                label="exact",
            );
            na.plt.stairs(
                hist_approx.inputs,
                hist_approx.outputs,
                ax=ax,
                label="approx",
            );
            ax.legend();
    """

    if absorption is None:
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

    if iqy is None:
        iqy = quantum_yield_ideal(wavelength, temperature)

    if fano_factor is None:
        fano_factor = _fano_factor(wavelength, temperature)

    if shape_random is None:
        shape_random = dict()

    shape = na.shape_broadcasted(
        photons_absorbed,
        absorption,
        iqy,
        thickness_implant,
        thickness_substrate,
        cce_backsurface,
        fano_factor,
    )
    shape = na.broadcast_shapes(shape, shape_random)

    f = fano_factor

    a = absorption
    W = thickness_implant
    D = thickness_substrate
    n0 = cce_backsurface
    aW = (a * W).to(u.dimensionless_unscaled).value
    aD = (a * D).to(u.dimensionless_unscaled).value

    # The absorbed photons are distributed within the substrate `[0, D)`, so the
    # depth follows an exponential truncated to that region (matching the exact
    # `electrons_measured` kernel). Conditioned on absorption within the
    # substrate, a photon lands in the implant region (partial charge collection)
    # with probability (1 - exp(-alpha * W)) / (1 - exp(-alpha * D)); otherwise
    # it is absorbed deeper in the substrate (complete charge collection).
    fraction_partial = (1 - np.exp(-aW)) / (1 - np.exp(-aD))
    photons_absorbed_partial = na.random.binomial(
        n=photons_absorbed,
        p=fraction_partial,
        shape_random=shape,
    )
    photons_absorbed_complete = photons_absorbed - photons_absorbed_partial

    mean_p = n0 + (1 - n0) / (aW) + (1 - n0) / (1 - np.exp(aW))
    var_p = np.square(n0 - 1) * (4 / np.square(aW) - 1 / np.square(np.sinh(aW / 2))) / 4

    mean_n = iqy
    var_n = f * mean_n

    mean_p2 = np.square(mean_p)
    mean_n2 = np.square(mean_n)
    mean_i = mean_n * mean_p
    var_exp = (var_n * var_p) + (var_n * mean_p2) + (var_p * mean_n2)
    exp_var = mean_n * (mean_p - (var_p + mean_p2)) * u.electron / u.photon
    var_i = var_exp + exp_var

    mean_partial = photons_absorbed_partial * mean_i
    vmr_partial = var_i / mean_i * u.photon

    electrons_partial = _discrete_gamma(
        mean=mean_partial,
        vmr=np.maximum(vmr_partial - 1 / (6 * mean_partial) * u.electron**2, f * u.ph),
        shape_random=shape,
    )
    electrons_complete = _discrete_gamma(
        mean=photons_absorbed_complete * iqy,
        vmr=f * u.photon,
        shape_random=shape,
    )
    result = electrons_complete + electrons_partial

    return result


_transmittance = transmittance
_absorbance = absorbance


def signal(
    photons_expected: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    direction: float | na.AbstractScalar = 1,
    n: complex | na.AbstractScalar = 1,
    n_substrate: None | complex | na.AbstractScalar = None,
    absorbance: None | float | na.AbstractScalar = None,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    thickness_depletion: u.Quantity | na.AbstractScalar = _thickness_substrate,
    thickness_substrate: None | na.AbstractScalar = _thickness_substrate,
    width_pixel: (
        u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
    ) = _width_pixel,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
    method: Literal["monte-carlo", "expected"] = "monte-carlo",
    axis_xy: None | tuple[str, str] = None,
    wrap: bool = False,
    shape_random: None | dict[str, int] = None,
) -> na.AbstractScalar:
    r"""
    A random sample from the distribution of measured electrons
    given the expected number of photons incident on the front surface of
    the sensor.

    This function adds shot noise to the expected number of photons,
    and then adds Fano noise and recombination noise using
    :func:`electrons_measured`.

    Parameters
    ----------
    photons_expected
        The `expected` number of photons incident on the detector surface.
    wavelength
        The vacuum wavelength of the absorbed photons.
    direction
        The cosine of the incidence angle.
    n
        The complex index of refraction of the ambient medium.
    n_substrate
        The complex index of refraction of the light-sensitive material
        If :obj:`None` (the default), the result of
        :meth:`optika.chemicals.Chemical.n` for silicon will be used.
    absorbance
        The fraction of incident energy absorbed by the light-sensitive region
        of the detector.
        If :obj:`None` (the default), the result of :func:`absorbance`
        called with default values will be used.
    thickness_implant
        The thickness of the implant layer.
        Default is the value given in :cite:t:`Stern1994`.
    thickness_depletion
        The thickness of the depletion region, the region with significant electric
        field.
        If :obj:`None` (the default), this is set to the same value as
        `thickness_substrate`.
    thickness_substrate
        The thickness of the entire light-sensitive region of the device.
        The default
    width_pixel
        The size of a single pixel on the sensor.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    temperature
        The temperature of the light-sensitive silicon layer.
    method
        The method used to generate samples of measured electrons.
        The `monte-carlo` method draws a random sample by simulating every
        photon using :func:`electrons_measured`, including shot, Fano, and
        recombination noise as well as charge diffusion.
        The `expected` method adds no noise and just returns the expected
        number of electrons in each pixel; since it is a per-pixel expectation,
        it does not apply charge diffusion.
    axis_xy
        The two logical axes corresponding to the pixel grid of the sensor
        along which electrons will diffuse.
        If :obj:`None` (the default), there is no charge diffusion.
    wrap
        Controls how diffused charge is treated at the edges of the pixel grid.
        If :obj:`False` (the default), charge that diffuses past the edge of the
        grid is lost, as it would be at the physical edge of a sensor.
        If :obj:`True`, the grid is treated as periodic and the charge re-enters
        the opposite edge (a toroidal boundary).
    shape_random
        Additional shape used to specify the number of samples to draw.

    Examples
    --------

    Plot the variance-to-mean ratio of the number of electrons measured by the sensor
    as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the number of experiments to perform
        num_experiments = 1000

        # Define the expected number of photons
        # for each experiment
        photons_expected = 100 * u.photon

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the actual number of electrons measured for each experiment
        electrons = optika.sensors.signal(
            photons_expected=photons_expected,
            wavelength=wavelength,
            shape_random=dict(experiment=num_experiments),
        )

        # Plot the variance-to-mean ratio of the result
        # as a function of wavelength.
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            electrons.vmr("experiment"),
            ax=ax,
        );
        ax.set_xscale("log");
        ax.set_yscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"variance-to-mean ratio ({electrons.unit:latex_inline})");
    """

    if absorbance is None:
        absorbance = _absorbance(
            wavelength=wavelength,
            direction=direction,
            n=n,
            thickness_substrate=thickness_substrate,
        ).average

    if n_substrate is None:
        n_substrate = optika.chemicals.Chemical("Si").n(wavelength)

    direction_substrate = optika.materials.snells_law_scalar(
        cos_incidence=direction,
        index_refraction=n,
        index_refraction_new=n_substrate,
    )

    absorption = absorption_effective(
        wavelength=wavelength,
        n_substrate=n_substrate,
        direction_substrate=direction_substrate,
    )

    if method == "expected":
        iqy = quantum_yield_ideal(
            wavelength=wavelength,
            temperature=temperature,
        )

        cce = charge_collection_efficiency(
            absorption=absorption,
            thickness_implant=thickness_implant,
            cce_backsurface=cce_backsurface,
        )
        return iqy * absorbance * cce * photons_expected.to(u.ph)

    elif method == "monte-carlo":

        photons = na.random.poisson(
            lam=absorbance * photons_expected.to(u.ph),
            shape_random=shape_random,
        ).astype(int)

        return electrons_measured(
            photons_absorbed=photons,
            wavelength=wavelength,
            absorption=absorption,
            thickness_implant=thickness_implant,
            thickness_depletion=thickness_depletion,
            thickness_substrate=thickness_substrate,
            width_pixel=width_pixel,
            cce_backsurface=cce_backsurface,
            temperature=temperature,
            axis_xy=axis_xy,
            wrap=wrap,
            shape_random=shape_random,
        )

    else:  # pragma: nocover
        raise ValueError(f"Unrecognized method: {method}")


def vmr_signal(
    wavelength: u.Quantity | na.ScalarArray,
    direction: float | na.AbstractScalar = 1,
    n: complex | na.AbstractScalar = 1,
    n_substrate: None | complex | na.AbstractScalar = None,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
    shot: bool = True,
    fano: bool = True,
    pcc: bool = True,
) -> na.ScalarArray:
    r"""
    Compute the variance-to-mean ratio (VMR) of the number of electrons measured by
    the sensor using an analytic expression.

    Parameters
    ----------
    wavelength
        The vacuum wavelength of the absorbed photons.
    direction
        The cosine of the incidence angle.
    n
        The complex index of refraction of the ambient medium.
    n_substrate
        The complex index of refraction of the light-sensitive material.
        If :obj:`None` (the default), the result of
        :meth:`optika.chemicals.Chemical.n` for silicon will be used.
    thickness_implant
        The thickness of the implant layer.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    temperature
        The temperature of the light-sensitive silicon layer.
    shot
        Whether to include shot noise in the result.
    fano
        Whether to include the Fano noise in the result.
    pcc
        Whether to include noise due to partial charge collection in the result.

    Examples
    --------

    Compute the VMR of the signal analytically and compare to a Monte Carlo approximation

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the number of experiments to perform
        num_experiments = 1000

        # Define the expected number of photons
        # for each experiment
        photons_expected = 100 * u.photon

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the variance-to-mean ratio of the signal analytically
        vmr_signal = optika.sensors.vmr_signal(wavelength)

        # Compute the actual number of electrons measured for each experiment
        signal = optika.sensors.signal(
            photons_expected=photons_expected,
            wavelength=wavelength,
            shape_random=dict(experiment=num_experiments),
        )

        # Plot the variance-to-mean ratio of the result
        # as a function of wavelength.
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            signal.vmr("experiment"),
            ax=ax,
            label="Monte Carlo",
        );
        na.plt.plot(
            wavelength,
            vmr_signal,
            ax=ax,
            label="analytic"
        );
        ax.set_xscale("log");
        ax.set_yscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"variance-to-mean ratio ({signal.unit:latex_inline})");
        ax.legend();

    Notes
    -----
    The VMR of the measured electrons is given by

    .. math::

        F(N_e'') = 1 - \overline{\eta} - F(\eta) + \overline{n} \, \overline{\eta} + \overline{\eta} \mathcal{F} + \overline{n} F(\eta) + \mathcal{F} F(\eta)

    where :math:`N_e''` is the number of measured electrons,
    :math:`\overline{\eta}` is the average charge-collection efficiency,
    :math:`\overline{n}` is the average quantum yield,
    and :math:`\mathcal{F}` is the Fano factor.
    The VMR of the charge-collection efficiency (CCE) is

    .. math::

        F(\eta) = \frac{2 e^{-\alpha W}}{\overline{\eta}} \left( \frac{1 - \eta_0}{\alpha W} \right)^2 \bigl( \sinh(\alpha W) - \alpha W \bigr)

    where :math:`\alpha` is the absorption coefficient,
    :math:`W` is the thickness of the implant region,
    and :math:`\eta_0` is the CCE at the back surface.
    """

    if n_substrate is None:
        n_substrate = optika.chemicals.Chemical("Si").n(wavelength)

    direction_substrate = optika.materials.snells_law_scalar(
        cos_incidence=direction,
        index_refraction=n,
        index_refraction_new=n_substrate,
    )

    absorption = absorption_effective(
        wavelength=wavelength,
        n_substrate=n_substrate,
        direction_substrate=direction_substrate,
    )

    iqy = quantum_yield_ideal(
        wavelength=wavelength,
        temperature=temperature,
    )

    cce = charge_collection_efficiency(
        absorption=absorption,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
    )

    F = fano_factor(wavelength)

    result = 0

    if shot:
        F_shot = iqy * cce

        result = result + F_shot

    if fano:
        F_fano = cce * F

        result = result + F_fano

    if pcc:
        n0 = cce_backsurface
        aW = absorption * thickness_implant
        aW = aW.to(u.dimensionless_unscaled).value
        F_cce = 2 * np.exp(-aW) * np.square((n0 - 1) / aW) * (np.sinh(aW) - aW) / cce

        unit = u.electron / u.photon

        F_pcc = 1 * unit - cce * unit - F_cce * unit + iqy * F_cce + F * F_cce

        result = result + F_pcc

    return result * u.photon


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSensorMaterial(
    optika.materials.AbstractMaterial,
):
    """
    An interface representing the light-sensitive material of an imaging sensor.
    """

    @abc.abstractmethod
    def signal(
        self,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
        width_pixel: (
            u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
        ) = 0
        * u.um,
        axis_xy: None | tuple[str, str] = None,
        noise: bool = True,
        wrap: bool = False,
    ) -> na.AbstractScalar:
        """
        Given the photons incident on each pixel, compute the number of
        electrons measured by the sensor using :func:`signal`.

        Parameters
        ----------
        photons
            The number of photons incident on each pixel.
        wavelength
            An assumed grid of wavelengths for the incident photons.
        direction
            The cosine of the refracted angle inside the light-sensitive region,
            as produced by :meth:`direction_refracted`.
        width_pixel
            The physical size of each pixel, used by the charge-diffusion model.
        axis_xy
            The two logical axes corresponding to the pixel grid of the sensor.
            If provided, charge diffusion will occur along these two axes.
            If :obj:`None` (the default), no diffusion is performed.
        noise
            Whether to add noise to the result.
        wrap
            Controls how diffused charge is treated at the edges of the pixel grid.
            If :obj:`False` (the default), charge that diffuses past the edge of
            the grid is lost, as it would be at the physical edge of a sensor.
            If :obj:`True`, the grid is treated as periodic and the charge
            re-enters the opposite edge (a toroidal boundary).
        """

    @abc.abstractmethod
    def direction_refracted(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> complex | na.AbstractScalar:
        """
        The cosine of the refracted propagation angle inside the light-sensitive
        region of the sensor.

        This is the quantity :meth:`signal` expects as its ``direction``
        argument. Performing the refraction here lets the sensor's ``collect``
        method fold the ambient index of refraction into the per-pixel cosine,
        so :meth:`signal` does not need a separate ambient-index argument.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the sensor.
        """

    @abc.abstractmethod
    def photons_incident(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Given the number of electrons measured by the sensor,
        and a grid of wavelengths, compute the expected number of
        photons incident on the sensor.

        Parameters
        ----------
        electrons
            The number of electrons measured by each pixel.
        wavelength
            An assumed grid of wavelengths for the incident photons.
        direction
            An assumed propagation direction for the incident photons.
        normal
            The vector perpendicular to the surface of the sensor.
        """

    @abc.abstractmethod
    def photons_absorbed(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
    ) -> na.AbstractScalar:
        """
        Given the number of electrons measured by the sensor, compute the
        expected number of photons *absorbed* by the light-sensitive region.

        This is the inverse of :meth:`signal`: it divides out only the quantum
        yield and the charge collection efficiency, not the absorbance. The
        absorbance is deliberately excluded because it is usually accounted for
        elsewhere (for example in the effective area of the optical system).

        Parameters
        ----------
        electrons
            The number of electrons measured by each pixel.
        wavelength
            The vacuum wavelength of the absorbed photons.
        direction
            The cosine of the refracted angle inside the light-sensitive region,
            as produced by :meth:`direction_refracted`.
        """


@dataclasses.dataclass(eq=False, repr=False)
class IdealSensorMaterial(
    optika.materials.Vacuum,
    AbstractSensorMaterial,
):
    """
    An idealized sensor material with a quantum efficiency of unity,
    no charge diffusion, and a noise model which consists of only shot noise.
    """

    def signal(
        self,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
        width_pixel: (
            u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
        ) = 0
        * u.um,
        axis_xy: None | tuple[str, str] = None,
        noise: bool = True,
        wrap: bool = False,
    ) -> na.AbstractScalar:

        if not photons.unit.is_equivalent(u.photon):
            h = astropy.constants.h
            c = astropy.constants.c
            photons = photons / (h * c / wavelength) * u.photon

        if noise:
            photons = na.random.poisson(photons.to(u.ph)).astype(int)

        electrons = photons * u.electron / u.photon
        electrons = electrons.to(u.electron)

        return electrons

    def direction_refracted(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> complex | na.AbstractScalar:
        # An ideal material does not refract, so the refracted cosine is just
        # the cosine of the incidence angle.
        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)
        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)
        return -direction @ normal

    def photons_incident(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        return electrons * u.photon / u.electron

    def photons_absorbed(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
    ) -> na.AbstractScalar:
        return electrons * u.photon / u.electron


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSiliconSensorMaterial(
    AbstractSensorMaterial,
):
    """
    An interface representing the light-sensitive material of a silicon sensor.
    """

    temperature: u.Quantity | na.AbstractScalar = 300 * u.K
    """The temperature of this sensor."""

    @property
    def transformation(self) -> None:
        return None

    @functools.cached_property
    def _chemical(self) -> optika.chemicals.Chemical:
        return optika.chemicals.Chemical("Si")

    @functools.cached_property
    def _chemical_oxide(self) -> optika.chemicals.Chemical:
        return optika.chemicals.Chemical("SiO2_llnl_cxro_rodriguez")

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
        return quantum_yield_ideal(
            wavelength=wavelength,
            temperature=self.temperature,
        )

    def fano_factor(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.ScalarArray:
        """
        The `Fano factor <https://en.wikipedia.org/wiki/Fano_factor>`_
        (ratio of the variance to the mean) of the Fano noise for this
        sensor material.

        The method uses the equivalent function, :func:`optika.sensors.fano_factor,
        along with the :attr:`temperature` attribute to compute the Fano factor
        for this material
        """
        return fano_factor(
            wavelength=wavelength,
            temperature=self.temperature,
        )

    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        return rays.index_refraction

    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        return rays.attenuation

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class AbstractBackIlluminatedSiliconSensorMaterial(
    AbstractSiliconSensorMaterial,
):
    """
    An interface representing the light-sensitive material of a backilluminated
    silicon sensor.
    """

    @property
    @abc.abstractmethod
    def thickness_oxide(self) -> u.Quantity | na.AbstractScalar:
        """
        The thickness of the oxide layer on the illuminated side of the sensor.
        """

    @property
    @abc.abstractmethod
    def thickness_implant(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of the ion implant layer."""

    @property
    @abc.abstractmethod
    def thickness_substrate(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of the light-sensitive silicon substrate."""

    @property
    @abc.abstractmethod
    def roughness_oxide(self):
        """The RMS roughness of the oxide layer surface."""

    @property
    @abc.abstractmethod
    def roughness_substrate(self):
        """The RMS roughness of the silicon substrate surface."""

    @property
    @abc.abstractmethod
    def cce_backsurface(self) -> float | na.AbstractScalar:
        """
        The charge collection efficiency on the illuminated surface of the sensor.
        """

    @property
    @abc.abstractmethod
    def depletion(self) -> AbstractDepletionModel:
        """
        A model of this sensor's depletion region.
        """

    def width_charge_diffusion(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The standard deviation of the charge diffusion kernel for this sensor.
        Calculated using :func:`optika.sensors.charge_diffusion`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        """
        return optika.sensors.charge_diffusion(
            self._chemical.absorption(wavelength),
            thickness_substrate=self.thickness_substrate,
            thickness_depletion=self.depletion.thickness,
        )

    def transmittance(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> optika.vectors.PolarizationVectorArray:
        r"""
        Compute the fraction of energy transmitted to the light-sensitive region
        of the sensor.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the CCD sensor.
        """
        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)

        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)

        return transmittance(
            wavelength=wavelength,
            direction=-direction @ normal,
            n=n,
            thickness_oxide=self.thickness_oxide,
            thickness_substrate=self.thickness_substrate,
            chemical_oxide=self._chemical_oxide,
            chemical_substrate=self._chemical,
            roughness_oxide=self.roughness_oxide,
            roughness_substrate=self.roughness_substrate,
        )

    def absorbance(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> optika.vectors.PolarizationVectorArray:
        r"""
        Compute the fraction of energy absorbed by the light-sensitive region
        of the sensor.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the CCD sensor.
        """
        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)

        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)

        return absorbance(
            wavelength=wavelength,
            direction=-direction @ normal,
            n=n,
            thickness_oxide=self.thickness_oxide,
            thickness_substrate=self.thickness_substrate,
            chemical_oxide=self._chemical_oxide,
            chemical_substrate=self._chemical,
            roughness_oxide=self.roughness_oxide,
            roughness_substrate=self.roughness_substrate,
        )

    def charge_collection_efficiency(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> na.AbstractScalar:
        r"""
        Compute the charge collection efficiency of this CCD sensor material
        using :func:`charge_collection_efficiency`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the CCD sensor.
        """

        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)

        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)

        direction = -direction @ normal

        n_substrate = self._chemical.n(wavelength)

        direction_substrate = optika.materials.snells_law_scalar(
            cos_incidence=direction,
            index_refraction=n,
            index_refraction_new=n_substrate,
        )

        return charge_collection_efficiency(
            absorption=absorption_effective(
                wavelength=wavelength,
                n_substrate=n_substrate,
                direction_substrate=direction_substrate,
            ),
            thickness_implant=self.thickness_implant,
            cce_backsurface=self.cce_backsurface,
        )

    def quantum_efficiency_effective(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> na.AbstractScalar:
        r"""
        Compute the effective quantum efficiency of this CCD material using
        :func:`quantum_efficiency_effective`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the CCD.
        """
        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)

        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)

        direction = -direction @ normal

        return quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
            n=n,
            thickness_oxide=self.thickness_oxide,
            thickness_implant=self.thickness_implant,
            thickness_substrate=self.thickness_substrate,
            cce_backsurface=self.cce_backsurface,
            chemical_oxide=self._chemical_oxide,
            chemical_substrate=self._chemical,
            roughness_oxide=self.roughness_oxide,
            roughness_substrate=self.roughness_substrate,
        )

    def quantum_efficiency(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> na.AbstractScalar:
        r"""
        Compute the quantum efficiency of this CCD material using
        :meth:`quantum_efficiency_effective` and
        :meth:`quantum_yield_ideal`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the CCD.
        """
        iqy = self.quantum_yield_ideal(wavelength)
        eqe = self.quantum_efficiency_effective(
            wavelength=wavelength,
            direction=direction,
            n=n,
            normal=normal,
        )
        return iqy * eqe

    def probability_measurement(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> na.AbstractScalar:
        r"""
        Compute the probability of measuring an absorbed photon for this sensor
        using :func:`probability_measurement`.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in the ambient medium.
            If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
        normal
            The vector perpendicular to the surface of the CCD.
        """
        return probability_measurement(
            iqy=self.quantum_yield_ideal(wavelength),
            cce=self.charge_collection_efficiency(
                wavelength=wavelength,
                direction=direction,
                normal=normal,
            ),
        )

    def direction_refracted(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> complex | na.AbstractScalar:
        if direction is None:
            direction = na.Cartesian3dVectorArray(0, 0, 1)

        if normal is None:
            normal = na.Cartesian3dVectorArray(0, 0, -1)

        return optika.materials.snells_law_scalar(
            cos_incidence=-direction @ normal,
            index_refraction=n,
            index_refraction_new=self._chemical.n(wavelength),
        )

    def electrons_measured(
        self,
        photons_absorbed: na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
        width_pixel: (
            u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
        ) = 0
        * u.um,
        axis_xy: None | tuple[str, str] = None,
        wrap: bool = False,
    ) -> na.AbstractScalar:
        """
        Randomly sample the number of measured electrons given the number of
        absorbed photons using :func:`electrons_measured`.
        """

        direction_substrate = self.direction_refracted(
            wavelength=wavelength,
            direction=direction,
            n=n,
            normal=normal,
        )

        return electrons_measured(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            absorption=absorption_effective(
                wavelength=wavelength,
                n_substrate=self._chemical.n(wavelength),
                direction_substrate=direction_substrate,
            ),
            thickness_implant=self.thickness_implant,
            thickness_depletion=self.depletion.thickness,
            thickness_substrate=self.thickness_substrate,
            width_pixel=width_pixel,
            cce_backsurface=self.cce_backsurface,
            axis_xy=axis_xy,
            wrap=wrap,
            temperature=self.temperature,
        )

    def signal(
        self,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
        width_pixel: (
            u.Quantity | na.AbstractScalar | na.AbstractCartesian2dVectorArray
        ) = 0
        * u.um,
        axis_xy: None | tuple[str, str] = None,
        noise: bool = True,
        wrap: bool = False,
    ) -> na.AbstractScalar:

        if not photons.unit.is_equivalent(u.photon):
            h = astropy.constants.h
            c = astropy.constants.c
            photons = photons / (h * c / wavelength) * u.photon

        if noise:
            method = "monte-carlo"
        else:
            method = "expected"

        # `direction` is the cosine of the refracted angle *inside* the substrate
        # (computed by the sensor's `collect` method), so use equal ambient and
        # substrate indices to make the Snell refraction inside `signal` a no-op
        # while still computing the effective absorption with the correct index.
        n_substrate = self._chemical.n(wavelength)

        return signal(
            photons_expected=photons,
            wavelength=wavelength,
            direction=direction,
            n=n_substrate,
            n_substrate=n_substrate,
            absorbance=1,
            thickness_implant=self.thickness_implant,
            thickness_depletion=self.depletion.thickness,
            thickness_substrate=self.thickness_substrate,
            width_pixel=width_pixel,
            cce_backsurface=self.cce_backsurface,
            temperature=self.temperature,
            method=method,
            axis_xy=axis_xy,
            wrap=wrap,
        )

    def photons_incident(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray = None,
        n: complex | na.AbstractScalar = 1,
        normal: None | na.AbstractCartesian3dVectorArray = None,
    ) -> na.AbstractScalar:
        """
        Compute the expected number of incident photons for a given number
        of electrons.

        Parameters
        ----------
        electrons
            The energy collected by the sensor in units of electrons.
        wavelength
            The assumed wavelength of the incident photons.
        direction
            The assumed direction of the incident photons.
        n
            The complex index of refraction of the ambient medium.
        normal
            The vector perpendicular to the surface of the sensor.
        """

        qe = self.quantum_efficiency(
            wavelength=wavelength,
            direction=direction,
            n=n,
            normal=normal,
        )

        return electrons / qe

    def photons_absorbed(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar = 1,
    ) -> na.AbstractScalar:
        # `direction` is the cosine of the refracted angle *inside* the
        # substrate (as passed to `signal`), so compute the substrate
        # absorption directly from it and divide out only the quantum yield and
        # charge collection efficiency, matching `signal` at ``absorbance=1``.
        n_substrate = self._chemical.n(wavelength)

        absorption = absorption_effective(
            wavelength=wavelength,
            n_substrate=n_substrate,
            direction_substrate=direction,
        )

        iqy = quantum_yield_ideal(
            wavelength=wavelength,
            temperature=self.temperature,
        )

        cce = charge_collection_efficiency(
            absorption=absorption,
            thickness_implant=self.thickness_implant,
            cce_backsurface=self.cce_backsurface,
        )

        return electrons / (iqy * cce)

    def efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        result = self.absorbance(
            wavelength=rays.wavelength,
            direction=rays.direction,
            n=rays.n,
            normal=normal,
        )

        return result.average


@dataclasses.dataclass(eq=False, repr=False)
class BackIlluminatedSiliconSensorMaterial(
    AbstractBackIlluminatedSiliconSensorMaterial,
):
    """
    A back-illuminated silicon sensor material which uses the method
    described in :cite:t:`Stern1994` to compute the quantum efficiency.
    """

    thickness_oxide: u.Quantity = 0 * u.nm
    """The thickness of the oxide layer on the illuminated side of the sensor."""

    thickness_implant: u.Quantity = 0 * u.nm
    """The thickness of the ion implant layer."""

    thickness_substrate: u.Quantity | na.AbstractScalar = 0 * u.um
    """The thickness of the light-sensitive silicon substrate."""

    roughness_oxide: u.Quantity = 0 * u.nm
    """The RMS roughness of the oxide layer on the illuminated side of the sensor."""

    roughness_substrate: u.Quantity = 0 * u.um
    """The RMS roughness of the silicon substrate."""

    cce_backsurface: u.Quantity = 0
    """The charge-collection efficiency of the back surface of the sensor."""

    depletion: None | AbstractDepletionModel = None
    """A model of this sensor's depletion region."""

    eqe_measured: None | na.FunctionArray = None
    """An optional measurement of the effective quantum efficiency."""

    @property
    def shape(self) -> dict[str, int]:
        return dict()

    @classmethod
    def fit_eqe(
        cls,
        thickness_substrate: u.Quantity | na.AbstractScalar,
        depletion: AbstractDepletionModel,
        eqe_measured: na.FunctionArray,
        temperature: u.Quantity | na.AbstractScalar = 300 * u.K,
    ) -> Self:
        """
        Fit the parameters of this sensor to a given effective quantum efficiency.

        Parameters
        ---------
        temperature
            The temperature of the light-sensitive silicon substrate.
        thickness_substrate
            The thickness of the light-sensitive silicon substrate.
        depletion
            A model of this sensor's depletion region.
        eqe_measured
            The measured quantum efficiency that will be fit by the function
            :func:`optika.sensors.quantum_efficiency_effective`.
        """
        unit_thickness_oxide = u.AA
        unit_thickness_implant = u.AA
        unit_roughness = u.nm
        unit_cce_backsurface = u.dimensionless_unscaled

        roughness_oxide = 0 * unit_roughness

        def eqe_rms_difference(x: tuple[float, float, float, float]):
            (
                thickness_oxide,
                thickness_implant,
                roughness_substrate,
                cce_backsurface,
            ) = x
            qe_fit = quantum_efficiency_effective(
                wavelength=eqe_measured.inputs,
                direction=1,
                thickness_oxide=thickness_oxide << unit_thickness_oxide,
                thickness_implant=thickness_implant << unit_thickness_implant,
                thickness_substrate=thickness_substrate,
                roughness_oxide=0 * unit_roughness,
                roughness_substrate=roughness_substrate << unit_roughness,
                cce_backsurface=cce_backsurface << unit_cce_backsurface,
            )

            return np.sqrt(np.mean(np.square(eqe_measured.outputs - qe_fit))).ndarray

        thickness_oxide_guess = 50 * u.AA
        thickness_implant_guess = 2317 * u.AA
        roughness_substrate_guess = 5 * u.nm
        cce_backsurface_guess = 0.21 * u.dimensionless_unscaled

        fit = scipy.optimize.minimize(
            fun=eqe_rms_difference,
            x0=[
                thickness_oxide_guess.to_value(unit_thickness_oxide),
                thickness_implant_guess.to_value(unit_thickness_implant),
                roughness_substrate_guess.to_value(unit_roughness),
                cce_backsurface_guess.to_value(unit_cce_backsurface),
            ],
            bounds=[
                (0, None),
                (0, None),
                (0, None),
                (0, 1),
            ],
            method="nelder-mead",
        )

        (
            thickness_oxide,
            thickness_implant,
            roughness_substrate,
            cce_backsurface,
        ) = fit.x

        thickness_oxide = thickness_oxide << unit_thickness_oxide
        thickness_implant = thickness_implant << unit_thickness_implant
        roughness_substrate = roughness_substrate << unit_roughness
        cce_backsurface = cce_backsurface << unit_cce_backsurface

        return cls(
            temperature=temperature,
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            thickness_substrate=thickness_substrate,
            roughness_oxide=roughness_oxide,
            roughness_substrate=roughness_substrate,
            cce_backsurface=cce_backsurface,
            depletion=depletion,
            eqe_measured=eqe_measured,
        )
