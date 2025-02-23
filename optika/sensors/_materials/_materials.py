import abc
import functools
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
from ._depletion import AbstractDepletionModel

__all__ = [
    "energy_bandgap",
    "energy_electron_hole",
    "quantum_yield_ideal",
    "absorbance",
    "charge_collection_efficiency",
    "quantum_efficiency_effective",
    "probability_measurement",
    "electrons_measured",
    "signal",
    "AbstractImagingSensorMaterial",
    "IdealImagingSensorMaterial",
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

_thickness_oxide = 50 * u.AA
_thickness_substrate = 7 * u.um
_thickness_implant = 2317 * u.AA
_cce_backsurface = 0.21
_fano_noise = 0.1 * u.electron / u.photon


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
    This is also known in the literature as the pair-production energy.

    The ideal quantum yield is given in :cite:t:`Janesick2001` as:

    .. math::

        \text{QY}(\epsilon) = \begin{cases}
            0, & 0 \leq \epsilon < E_\text{g}\\
            1, &  E_\text{g} \leq \epsilon < E_\text{e-h} \\
            \epsilon / E_\text{e-h}, & E_\text{e-h} \leq \epsilon < \infty,
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
        ax.set_ylabel(f"quantum yield ({qy.unit:latex_inline})");
    """
    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    result = energy / energy_electron_hole
    result = np.where(energy > energy_electron_hole, result, 1)
    result = np.where(energy > energy_bandgap, result, 0)

    return result * u.electron / u.photon


def absorbance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: float | na.AbstractScalar = 1,
    thickness_oxide: u.Quantity | na.AbstractScalar = _thickness_oxide,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    chemical_oxide: str | optika.chemicals.AbstractChemical = "SiO2",
    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    roughness_oxide: u.Quantity | na.AbstractScalar = 0 * u.nm,
    roughness_substrate: u.Quantity | na.AbstractScalar = 0 * u.nm,
) -> optika.vectors.PolarizationVectorArray:
    """
    The fraction of incident energy absorbed by the light-sensitive
    region of the sensor

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The component of the incident light's propagation direction antiparallel
        to the surface normal of the sensor.
        Default is normal incidence.
    n
        The index of refraction in the ambient medium.
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

    Plot the absorbance as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the absorbance vs wavelength
        absorbance = optika.sensors.absorbance(
            wavelength=wavelength,
        )

        # Plot the effective and maximum quantum efficiency
        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(
            wavelength,
            absorbance.average,
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

    return np.real(result)


def charge_collection_efficiency(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    cos_incidence: float | na.AbstractScalar = 1,
) -> na.AbstractScalar:
    r"""
    Compute the average charge collection efficiency using the differential
    charge collection efficiency profile described in :cite:t:`Stern1994`.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive material for the
        wavelength of interest.
    thickness_implant
        The thickness of the implant layer, the layer where recombination can
        occur.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    cos_incidence
        The cosine of the angle of the incident light's propagation direction
        inside the substrate with the surface normal

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

    Equation :eq:`cce` is only valid for normally-incident light.
    We can generalize it to obliquely-incident light by making the substitution

    .. math::
        :label: x-oblique

        x \rightarrow \frac{x}{\cos \theta}

    where :math:`\theta` is the angle between the propagation direction
    inside the silicon substrate and the normal vector.

    Substituting :eq:`x-oblique` into Equation :eq:`cce` and solving yields

    .. math::
        :label: eqe-oblique

        \text{CCE}(\lambda, \theta) =
            \eta_0
            + \left( \frac{1 - \eta_0}{\alpha W \sec \theta} \right) (1 - e^{-\alpha W \sec \theta})
    """
    z0 = absorption * thickness_implant / np.real(cos_incidence)
    exp_z0 = np.exp(-z0)

    term_1 = cce_backsurface
    term_2 = ((1 - cce_backsurface) / z0) * (1 - exp_z0)

    return term_1 + term_2


def quantum_efficiency_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: None | na.AbstractCartesian3dVectorArray = None,
    n: float | na.AbstractScalar = 1,
    thickness_oxide: u.Quantity | na.AbstractScalar = _thickness_oxide,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    thickness_substrate: u.Quantity | na.AbstractScalar = _thickness_substrate,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    chemical_oxide: str | optika.chemicals.AbstractChemical = "SiO2",
    chemical_substrate: str | optika.chemicals.AbstractChemical = "Si",
    roughness_oxide: u.Quantity | na.AbstractScalar = 0 * u.nm,
    roughness_substrate: u.Quantity | na.AbstractScalar = 0 * u.nm,
    normal: None | na.AbstractCartesian3dVectorArray = None,
) -> na.AbstractScalar:
    r"""
    Calculate the effective quantum efficiency of a backilluminated detector.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The propagation direction of the incident light in the ambient medium.
        If :obj:`None` (default), normal incidence (:math:`\hat{z}`) is assumed.
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

        # Define the incidence directions
        angle = na.linspace(0, 30, axis="angle", num=2) * u.deg
        direction = na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        )

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
    if direction is None:
        direction = na.Cartesian3dVectorArray(0, 0, 1)

    if not isinstance(chemical_oxide, optika.chemicals.AbstractChemical):
        chemical_oxide = optika.chemicals.Chemical(chemical_oxide)

    if not isinstance(chemical_substrate, optika.chemicals.AbstractChemical):
        chemical_substrate = optika.chemicals.Chemical(chemical_substrate)

    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    direction = -direction @ normal

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
    wavenumber_substrate = np.imag(n_substrate)
    absorption_substrate = 4 * np.pi * wavenumber_substrate / wavelength

    direction_substrate = optika.materials.snells_law_scalar(
        cos_incidence=direction,
        index_refraction=n,
        index_refraction_new=n_substrate,
    )

    cce = charge_collection_efficiency(
        absorption=absorption_substrate,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        cos_incidence=direction_substrate,
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


def electrons_measured(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    absorption: u.Quantity | na.AbstractScalar,
    iqy: u.Quantity | na.AbstractScalar = 1 * u.electron / u.photon,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    fano_noise: u.Quantity | na.AbstractScalar = _fano_noise,
) -> na.AbstractScalar:
    r"""
    A random sample from the approximate distribution of measured electrons
    given the number of photons absorbed by the light-sensitive layer of the
    sensor.

    This function accounts for both Fano noise and recombination noise due to
    partial-charge collection.

    Parameters
    ----------
    photons_absorbed
        The number of photons absorbed by the light-sensitive layer of the sensor.
    absorption
        The absorption coefficient of the light-sensitive material for the
        wavelength of interest.
    iqy
        The ideal quantum yield of the sensor in electrons per photon.
    thickness_implant
        The thickness of the implant layer.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    fano_noise
        The `Fano factor <https://en.wikipedia.org/wiki/Fano_factor>`_
        (ratio of the variance to the mean) of the Fano noise for this
        sensor material.
        Defaults to 0.1, the standard estimate of Fano noise in silicon
        :cite:p:`Janesick2001`.
        Must be in units of electrons per photon.

    Examples
    --------

    Plot the energy spectrum of 100 6 keV photons emitted from an Fe-55
    radioactive source.

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
        photons_absorbed = na.broadcast_to(
            array=100* u.photon,
            shape=dict(experiment=num_experiments)
        ).astype(int)

        # Define the wavelength at which to sample the distribution
        wavelength = 5.9 * u.keV
        wavelength = wavelength.to(u.AA, equivalencies=u.spectral())

        # Compute the absorption coefficient of silicon at this wavelength
        absorption=optika.chemicals.Chemical("Si").absorption(wavelength)

        # Compute the ideal quantum yield of silicon for each wavelength
        iqy = optika.sensors.quantum_yield_ideal(wavelength)

        # Compute the actual number of electrons measured for each experiment
        electrons = optika.sensors.electrons_measured(
            photons_absorbed=photons_absorbed,
            absorption=absorption,
            iqy=iqy,
        )

        # Define the histogram bins
        step = 10
        bins = na.arange(
            electrons.value.min()-step/2,
            electrons.value.max()+step/2,
            step=step,
            axis="bin",
        ) * electrons.unit

        # Compute a histogram of resulting energy spectrum
        hist = na.histogram(
            electrons,
            bins=bins,
            axis="experiment",
        )

        # Plot the histogram
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            line = na.plt.stairs(
              hist.inputs,
              hist.outputs,
              ax=ax,
            )
    """
    shape = na.shape_broadcasted(
        photons_absorbed,
        absorption,
        absorbance,
        iqy,
        thickness_implant,
        cce_backsurface,
        fano_noise,
    )

    electrons_expected = iqy * photons_absorbed
    d = (2 / 12) * (u.electron / u.photon) ** 2
    f = fano_noise + d * (photons_absorbed - 1 * u.photon) / electrons_expected
    f = np.nan_to_num(f, neginf=fano_noise)

    a = absorption
    W = thickness_implant
    n0 = cce_backsurface
    aW = (a * W).to(u.dimensionless_unscaled).value

    fraction_complete = np.exp(-aW)
    fraction_partial = 1 - fraction_complete
    photons_absorbed_partial = na.random.binomial(
        n=photons_absorbed,
        p=fraction_partial,
        shape_random=shape,
    )
    photons_absorbed_complete = photons_absorbed - photons_absorbed_partial

    mean_p = n0 + (1 - n0) / (aW) + (n0 - 1) / (np.exp(aW) - 1)
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
        vmr=vmr_partial,
        shape_random=shape,
    )
    electrons_complete = _discrete_gamma(
        mean=photons_absorbed_complete * iqy,
        vmr=f * u.photon,
        shape_random=shape,
    )
    result = electrons_complete + electrons_partial

    return result


def signal(
    photons_expected: u.Quantity | na.AbstractScalar,
    absorption: u.Quantity | na.AbstractScalar,
    absorbance: float | na.AbstractScalar = 1,
    iqy: u.Quantity | na.AbstractScalar = 1 * u.electron / u.photon,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    fano_noise: u.Quantity | na.AbstractScalar = _fano_noise,
) -> na.AbstractScalar:
    r"""
    A random sample from the approximate distribution of measured electrons
    given the expected number of photons incident on the front surface of
    the sensor.

    This function adds shot noise to the expected number of photons,
    and then adds Fano noise and recombination noise using
    :func:`electrons_measured`.

    Parameters
    ----------
    photons_expected
        The `expected` number of photons incident on the detector surface.
    absorption
        The absorption coefficient of the light-sensitive material for the
        wavelength of interest.
    absorbance
        The fraction of incident energy absorbed by the light-sensitive layer
        of the detector computed using the average of :func:`absorbance`.
    iqy
        The ideal quantum yield of the sensor in electrons per photon.
    thickness_implant
        The thickness of the implant layer.
        Default is the value given in :cite:t:`Stern1994`.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
        Default is the value given in :cite:t:`Stern1994`.
    fano_noise
        The `Fano factor <https://en.wikipedia.org/wiki/Fano_factor>`_
        (ratio of the variance to the mean) of the Fano noise for this
        sensor material.
        Defaults to 0.1, the standard estimate of Fano noise in silicon
        :cite:p:`Janesick2001`.
        Must be in units of electrons per photon.

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
        photons_expected = na.broadcast_to(
            array=100 * u.photon,
            shape=dict(experiment=num_experiments)
        )

        # Define a grid of wavelengths
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Compute the absorption coefficient of silicon at this wavelength
        absorption=optika.chemicals.Chemical("Si").absorption(wavelength)

        # Compute the fraction of light absorbed by the light-sensitive
        # region of the detector
        absorbance = optika.sensors.absorbance(wavelength).average

        # Compute the ideal quantum yield of silicon for each wavelength
        iqy = optika.sensors.quantum_yield_ideal(wavelength)

        # Compute the actual number of electrons measured for each experiment
        electrons = optika.sensors.signal(
            photons_expected=photons_expected,
            absorption=absorption,
            absorbance=absorbance,
            iqy=iqy,
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

    photons_absorbed_expected = absorbance * photons_expected.to(u.ph)
    photons_absorbed = na.random.poisson(photons_absorbed_expected).astype(int)

    return electrons_measured(
        photons_absorbed=photons_absorbed,
        absorption=absorption,
        iqy=iqy,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        fano_noise=fano_noise,
    )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImagingSensorMaterial(
    optika.materials.AbstractMaterial,
):
    """
    An interface representing the light-sensitive material of an imaging sensor.
    """

    @abc.abstractmethod
    def electrons_measured(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:
        """
        Given a set of incident rays, compute the number of electrons
        measured by the sensor using :func:`electrons_measured`.

        Parameters
        ----------
        rays
            The rays incident on the sensor surface.
            The :attr:`optika.rays.RayVectorArray.intensity` field should
            either be in units of photons or energy.
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
            The number of electrons measured by the sensor.
        wavelength
            An assumed grid of wavelengths for the incident photons.
        direction
            An assumed propagation direction for the incident photons.
        normal
            The vector perpendicular to the surface of the sensor.
        """

    @abc.abstractmethod
    def charge_diffusion(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:
        """
        Given a set of incident rays, compute how much the position of each ray
        is perturbed due to charge diffusion within the sensor.

        Parameters
        ----------
        rays
            The rays incident on the sensor surface.
        normal
            The vector perpendicular to the surface of the sensor.
        """


@dataclasses.dataclass(eq=False, repr=False)
class IdealImagingSensorMaterial(
    optika.materials.Vacuum,
    AbstractImagingSensorMaterial,
):
    """
    An idealized sensor material with a quantum efficiency of unity
    and no charge diffusion.
    """

    def electrons_measured(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:

        intensity = rays.intensity
        if not intensity.unit.is_equivalent(u.photon):
            h = astropy.constants.h
            c = astropy.constants.c
            intensity = intensity / (h * c / rays.wavelength) * u.photon
        electrons = intensity * u.electron / u.photon
        electrons = electrons.to(u.electron)

        result = dataclasses.replace(rays, intensity=electrons)

        return result

    def photons_incident(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        return electrons * u.photon / u.electron

    def charge_diffusion(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:
        return rays


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

    @functools.cached_property
    def _chemical_oxide(self) -> optika.chemicals.Chemical:
        return optika.chemicals.Chemical("SiO2_llnl_cxro_rodriguez")

    @property
    def fano_noise(self) -> u.Quantity:
        """
        The `Fano factor <https://en.wikipedia.org/wiki/Fano_factor>`_
        (ratio of the variance to the mean) of the Fano noise for this
        sensor material.

        This parameter has some variation in the literature,
        this implementation uses the value given by :cite:t:`Rodrigues2021`.
        """
        return 0.119 * u.electron / u.photon

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
    def roughness_oxide(self):
        """The RMS roughness of the oxide layer surface."""

    @property
    @abc.abstractmethod
    def roughness_substrate(self):
        """The RMS roughness of the substrate surface."""

    @property
    @abc.abstractmethod
    def cce_backsurface(self) -> float | na.AbstractScalar:
        """
        The charge collection efficiency on the backsurface of the CCD sensor.
        """

    @property
    @abc.abstractmethod
    def depletion(self) -> AbstractDepletionModel:
        """
        A model of the sensor's depletion region.
        """

    def width_charge_diffusion(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        The standard deviation of the charge diffusion kernel for this sensor.
        Calculated using :func:`optika.sensors.charge_diffusion`.

        Parameters
        ----------
        rays
            The rays incident on the sensor surface.
        normal
            The vector perpendicular to the surface of the sensor.
        """
        return optika.sensors.charge_diffusion(
            self._chemical.absorption(rays.wavelength),
            thickness_substrate=self.thickness_substrate,
            thickness_depletion=self.depletion.thickness,
        )

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

    def absorbance(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.vectors.PolarizationVectorArray:
        """
        Compute the fraction of energy absorbed by the light-sensitive region
        of the sensor.

        Parameters
        ----------
        rays
            The light rays incident on the CCD surface.
        normal
            The vector perpendicular to the surface of the CCD sensor.
        """
        return absorbance(
            wavelength=rays.wavelength,
            direction=-rays.direction @ normal,
            n=rays.index_refraction,
            thickness_oxide=self.thickness_oxide,
            thickness_substrate=self.thickness_substrate,
            chemical_oxide=self._chemical_oxide,
            chemical_substrate=self._chemical,
            roughness_oxide=self.roughness_oxide,
            roughness_substrate=self.roughness_substrate,
        )

    def charge_collection_efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the charge collection efficiency of this CCD sensor material
        using :func:`charge_collection_efficiency`.

        Parameters
        ----------
        rays
            The light rays incident on the CCD surface.
        normal
            The vector perpendicular to the surface of the CCD sensor.
        """
        return charge_collection_efficiency(
            absorption=self._chemical.absorption(rays.wavelength),
            thickness_implant=self.thickness_implant,
            cce_backsurface=self.cce_backsurface,
            cos_incidence=-rays.direction @ normal,
        )

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
        k = rays.attenuation * rays.wavelength / (4 * np.pi)
        n = rays.index_refraction + k * 1j

        return quantum_efficiency_effective(
            wavelength=rays.wavelength,
            direction=rays.direction,
            n=n,
            thickness_oxide=self.thickness_oxide,
            thickness_implant=self.thickness_implant,
            thickness_substrate=self.thickness_substrate,
            cce_backsurface=self.cce_backsurface,
            chemical_oxide=self._chemical_oxide,
            chemical_substrate=self._chemical,
            roughness_oxide=self.roughness_oxide,
            roughness_substrate=self.roughness_substrate,
            normal=normal,
        )

    def quantum_efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the quantum efficiency of this CCD material using
        :meth:`quantum_efficiency_effective` and
        :meth:`quantum_yield_ideal`.

        Parameters
        ----------
        rays
            The light rays incident on the CCD surface
        normal
            The vector perpendicular to the surface of the CCD.
        """
        iqy = self.quantum_yield_ideal(rays.wavelength)
        eqe = self.quantum_efficiency_effective(rays, normal)
        return iqy * eqe

    def probability_measurement(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Compute the probability of measuring an absorbed photon for this sensor
        using :func:`probability_measurement`.

        Parameters
        ----------
        rays
            The light rays incident on the CCD surface
        normal
            The vector perpendicular to the surface of the CCD.
        """
        return probability_measurement(
            iqy=self.quantum_yield_ideal(rays.wavelength),
            cce=self.charge_collection_efficiency(rays, normal),
        )

    def electrons_measured(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:

        intensity = rays.intensity
        wavelength = rays.wavelength

        electrons = electrons_measured(
            photons_absorbed=intensity,
            absorption=self._chemical.absorption(wavelength),
            iqy=self.quantum_yield_ideal(wavelength),
            thickness_implant=self.thickness_implant,
            cce_backsurface=self.cce_backsurface,
            fano_noise=self.fano_noise,
        )

        result = dataclasses.replace(rays, intensity=electrons)

        return result

    def signal(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:

        intensity = rays.intensity
        wavelength = rays.wavelength

        if not intensity.unit.is_equivalent(u.photon):
            h = astropy.constants.h
            c = astropy.constants.c
            intensity = intensity / (h * c / wavelength) * u.photon

        electrons = signal(
            photons_expected=intensity,
            absorption=self._chemical.absorption(wavelength),
            absorbance=self.absorbance(rays, normal).average,
            iqy=self.quantum_yield_ideal(wavelength),
            thickness_implant=self.thickness_implant,
            cce_backsurface=self.cce_backsurface,
            fano_noise=self.fano_noise,
        )

        result = dataclasses.replace(rays, intensity=electrons)

        return result

    def photons_incident(
        self,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
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
        normal
            The vector perpendicular to the surface of the sensor.
        """

        qe = self.quantum_efficiency(
            rays=optika.rays.RayVectorArray(
                wavelength=wavelength,
                direction=direction,
            ),
            normal=normal,
        )

        return electrons / qe

    def charge_diffusion(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> optika.rays.RayVectorArray:

        width = self.width_charge_diffusion(rays, normal)

        position = dataclasses.replace(
            rays.position,
            x=na.random.normal(rays.position.x, width),
            y=na.random.normal(rays.position.y, width),
        )

        rays = dataclasses.replace(
            rays,
            position=position,
        )

        return rays

    def efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1


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
        unit_roughness = u.nm
        unit_cce_backsurface = u.dimensionless_unscaled

        def eqe_rms_difference(x: tuple[float, float, float, float]):
            (
                thickness_oxide,
                thickness_implant,
                roughness_substrate,
                cce_backsurface,
            ) = x
            qe_fit = quantum_efficiency_effective(
                wavelength=qe_measured.inputs,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
                thickness_oxide=thickness_oxide << unit_thickness_oxide,
                thickness_implant=thickness_implant << unit_thickness_implant,
                thickness_substrate=self.thickness_substrate,
                roughness_oxide=self.roughness_oxide,
                roughness_substrate=roughness_substrate << unit_roughness,
                cce_backsurface=cce_backsurface << unit_cce_backsurface,
            )

            return np.sqrt(np.mean(np.square(qe_measured.outputs - qe_fit))).ndarray

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

        return dict(
            thickness_oxide=thickness_oxide,
            thickness_implant=thickness_implant,
            roughness_substrate=roughness_substrate,
            cce_backsurface=cce_backsurface,
        )

    @property
    def thickness_oxide(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_oxide"]

    @property
    def thickness_implant(self) -> u.Quantity:
        return self._quantum_efficiency_fit["thickness_implant"]

    @property
    def roughness_oxide(self):
        return 0 * u.nm

    @property
    def roughness_substrate(self):
        return self._quantum_efficiency_fit["roughness_substrate"]

    @property
    def cce_backsurface(self) -> float:
        return self._quantum_efficiency_fit["cce_backsurface"]
