import pathlib
import math
import random
import numpy as np
import numba
import astropy.units as u
import named_arrays as na
import optika
from .._stern_1994 import (
    _thickness_implant,
    _cce_backsurface,
)

__all__ = [
    "energy_bandgap",
    "energy_pair",
    "energy_pair_inf",
    "quantum_yield_ideal",
    "fano_factor",
    "fano_factor_inf",
    "electrons_measured",
]


def _probability_of_n_pairs_from_file(
    path: pathlib.Path,
) -> na.FunctionArray[na.CartesianNdVectorArray, na.ScalarArray]:

    a = np.loadtxt(path)
    a = na.ScalarArray(a, axes=("wavelength", "num_electron"))
    energy = a[dict(num_electron=0)] << u.eV
    pn = a[dict(num_electron=slice(1, None))]
    n = na.arange(1, pn.shape["num_electron"] + 1, axis="num_electron")

    return na.FunctionArray(
        inputs=na.CartesianNdVectorArray(
            components=dict(
                n=n,
                energy=energy,
            ),
        ),
        outputs=pn,
    )


def _probability_of_n_pairs_ramanathan() -> na.FunctionArray[
    na.CartesianNdVectorArray,
    na.ScalarArray,
]:

    directory = pathlib.Path(__file__).parent
    pn_000K = _probability_of_n_pairs_from_file(directory / "p0K.dat")
    pn_100K = _probability_of_n_pairs_from_file(directory / "p100K.dat")
    pn_300K = _probability_of_n_pairs_from_file(directory / "p300K.dat")

    probability = na.stack(
        arrays=[
            pn_000K.outputs,
            pn_100K.outputs,
            pn_300K.outputs,
        ],
        axis="temperature",
    )

    n = pn_000K.inputs.components["n"]
    energy = pn_000K.inputs.components["energy"]

    temperature = na.ScalarArray(
        ndarray=[0, 100, 300] * u.K,
        axes="temperature",
    )

    return na.FunctionArray(
        inputs=na.CartesianNdVectorArray(
            components=dict(
                energy=energy,
                temperature=temperature,
                n=n,
            )
        ),
        outputs=probability,
    )


def energy_bandgap(
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    r"""
    Bandgap energy in silicon given by :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    temperature
        The temperature of the silicon.

    Examples
    --------

    Reproduce Figure 2 of :cite:t:`Ramanathan2020`, and plot the bandgap
    energy as a function of temperature.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        T = na.linspace(0, 350, axis="temperature", num=101) * u.K
        energy_gap = optika.sensors.energy_bandgap(T)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                T,
                energy_gap
            )
            ax.set_xlabel(f"temperature ({ax.get_xlabel()})")
            ax.set_ylabel(f"bandgap energy ({ax.get_ylabel()})")

    Notes
    -----

    :cite:t:`Ramanathan2020` gives the bandgap energy as

    .. math::

        E_g(T) = E_g(0) - \frac{a T^2}{T + b}

    where :math:`T` is the temperature of the silicon,
    :math:`E_g(0) = 1.192 \, \text{eV}`,
    :math:`a = 4.9 \times 10^{-4} \, \text{eV / K}`,
    and :math:`b = 655 \, \text{K}`.

    """
    T = temperature.to(u.K, equivalencies=u.temperature())
    energy_gap_0 = 1.1692 * u.eV
    a = 4.9e-4 * u.eV / u.K
    b = 655 * u.K

    return energy_gap_0 - a * np.square(T) / (T + b)


def energy_pair(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    """
    Calculate the average pair-production energy in silicon given by
    :cite:t:`Ramanathan2020`.

    Parameters
    ----------
     wavelength
        The vacuum wavelength of the incident photons.
    temperature
        The temperature of the silicon.

    Examples
    --------

    Compute the pair-production energy as a function of incident photon energy.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        energy = na.geomspace(1, 100, axis="energy", num=1001) * u.eV
        energy_pair = optika.sensors.energy_pair(energy)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                energy,
                energy_pair,
            )
            ax.set_xscale("log")
            ax.set_xlabel(f"incident photon energy ({ax.get_xlabel()})")
            ax.set_ylabel(f"pair-production energy ({ax.get_ylabel()})")
    """
    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    temperature = temperature.to(u.K, equivalencies=u.temperature())

    pn = _probability_of_n_pairs_ramanathan()

    _n = pn.inputs.components["n"]
    _energy = pn.inputs.components["energy"]
    _temperature = pn.inputs.components["temperature"]
    _probability = pn.outputs

    _iqy = (_n * _probability).sum("num_electron")

    _energy_pair = _energy / _iqy

    _energy_pair_inf = energy_pair_inf(temperature)

    energy_pair = na.interp(
        x=temperature,
        xp=_temperature,
        fp=_energy_pair,
    )

    energy_pair = na.interp(
        x=energy,
        xp=_energy,
        fp=energy_pair,
        right=_energy_pair_inf.value,
    )

    return energy_pair


def energy_pair_inf(
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    r"""
    The asymptotic electron-hole pair production energy in silicon
    given by :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    temperature
        The temperature of the silicon.

    Notes
    -----

    :cite:t:`Ramanathan2020` gives the mean pair production energy as

    .. math::

        \epsilon_{eh} = 1.7 E_g + 0.084 A + 1.3,

    where :math:`E_g` is the bandgap energy of silicon calculated using
    :func:`energy_bandgap` and
    :math:`A = 5.2 \, \text{eV}^2`.

    """

    A = 5.2 * u.eV**2

    E_g = energy_bandgap(temperature)

    result = 1.7 * E_g + 0.084 * A / u.eV + 1.3 * u.eV

    return result


def quantum_yield_ideal(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    r"""
    Calculate the ideal quantum yield of a silicon detector for a given
    wavelength and temperature using the model given in :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    wavelength
        The vacuum wavelength of the incident photons.
    temperature
        The temperature of the silicon detector.

    Examples
    --------

    Plot the ideal quantum yield vs wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths
        wavelength = na.geomspace(100, 100000, axis="wavelength", num=1001) << u.AA

        # Compute the quantum yield
        iqy = optika.sensors.quantum_yield_ideal(wavelength)

        # Plot the quantum yield vs wavelength
        fig, ax = plt.subplots()
        na.plt.plot(wavelength, iqy, ax=ax);
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"quantum yield ({iqy.unit:latex_inline})");
    """

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    W = energy_pair(
        wavelength=wavelength,
        temperature=temperature,
    )

    iqy = energy / W

    return iqy * u.electron / u.photon


def fano_factor(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    r"""
    Calculate the Fano factor of a silicon detector for a given
    wavelength and temperature using the model given in :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    wavelength
        The vacuum wavelength of the incident photons.
    temperature
        The temperature of the silicon detector.

    Examples
    --------

    Plot the Fano factor vs wavelength

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define an array of wavelengths
        wavelength = na.geomspace(100, 100000, axis="wavelength", num=1001) << u.AA

        # Compute the Fano factor
        f = optika.sensors.fano_factor(wavelength)

        # Plot the Fano factor vs wavelength
        fig, ax = plt.subplots()
        na.plt.plot(wavelength, f, ax=ax);
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"Fano factor ({f.unit:latex_inline})");
    """

    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    temperature = temperature.to(u.K, equivalencies=u.temperature())

    pn = _probability_of_n_pairs_ramanathan()

    _n = pn.inputs.components["n"]
    _energy = pn.inputs.components["energy"]
    _temperature = pn.inputs.components["temperature"]
    _probability = pn.outputs

    _iqy = (_n * _probability).sum("num_electron")

    _v = (np.square(_n) * _probability).sum("num_electron")
    _fano_factor = (_v - np.square(_iqy)) / _iqy

    _fano_factor_inf = fano_factor_inf(_temperature)

    fano_factor = na.interp(
        x=energy,
        xp=_energy,
        fp=_fano_factor,
        right=_fano_factor_inf.value,
    )

    fano_factor = na.interp(
        x=temperature,
        xp=_temperature,
        fp=fano_factor,
    )

    return fano_factor * u.electron / u.photon


def fano_factor_inf(
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:
    r"""
    The asymptotic Fano factor in silicon given by :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    temperature
        The temperature of the silicon.

    Notes
    -----

    :cite:t:`Ramanathan2020` gives the mean Fano factor as

    .. math::

        \epsilon_{eh} = -0.028 E_g + 0.0015 A + 0.14,

    where :math:`E_g` is the bandgap energy of silicon calculated using
    :func:`energy_bandgap` and
    :math:`A = 5.2 \, \text{eV}^2`.
    """

    temperature = temperature.to(u.K, equivalencies=u.temperature())

    A = 5.2 * u.eV**2

    E_g = energy_bandgap(temperature)

    result = -0.028 * E_g + 0.0015 * A / u.eV + 0.14 * u.eV

    result = result * u.electron / u.photon / u.eV

    return result


def probability_of_n_pairs(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
    r"""
    Calculate the PMF of the number of electron-hole pairs generated in a
    silicon detector for a given wavelength and temperature using the model
    given in :cite:t:`Ramanathan2020`.

    Parameters
    ----------
    wavelength
        The vacuum wavelength of the incident photons.
    temperature
        The temperature of the silicon detector.
    """

    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    temperature = temperature.to(u.K, equivalencies=u.temperature())

    pn = _probability_of_n_pairs_ramanathan()

    _n = pn.inputs.components["n"]
    _energy = pn.inputs.components["energy"]
    _temperature = pn.inputs.components["temperature"]
    _probability = pn.outputs

    probability = na.interp(
        x=temperature,
        xp=_temperature,
        fp=_probability,
    )

    probability = na.interp(
        x=energy,
        xp=_energy,
        fp=probability,
    )

    result = na.FunctionArray(
        inputs=_n,
        outputs=probability,
    )

    return result


def electrons_measured(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    absorption: None | u.Quantity | na.AbstractScalar = None,
    thickness_implant: u.Quantity | na.AbstractScalar = _thickness_implant,
    cce_backsurface: u.Quantity | na.AbstractScalar = _cce_backsurface,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
    shape_random: None | dict[str, int] = None,
) -> na.AbstractScalar:
    r"""
    A random sample from the distribution of measured electrons
    given the number of photons absorbed by the light-sensitive layer of the
    sensor.

    This function accounts for both Fano noise and recombination noise due to
    partial-charge collection.

    Parameters
    ----------
    photons_absorbed
        The number of photons absorbed by the light-sensitive layer of the sensor.
    wavelength
        The vacuum wavelength of the absorbed photons.
    absorption
        The absorption coefficient of silicon at the given wavelength.
    thickness_implant
        The thickness of the implant layer, where partial-charge collection occurs.
    cce_backsurface
        The differential charge collection efficiency on the back surface
        of the sensor.
    temperature
        The temperature of the silicon detector.
        Default is room temperature.
    shape_random
        Additional shape used to specify the number of samples to draw.

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
        photons_absorbed = (100 * u.photon).astype(int)

        # Define the wavelength at which to sample the distribution
        wavelength = 5.9 * u.keV
        wavelength = wavelength.to(u.AA, equivalencies=u.spectral())

        # Compute the actual number of electrons measured for each experiment
        electrons = optika.sensors.electrons_measured(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            shape_random=dict(experiment=num_experiments),
        )

        # Define the histogram bins
        step = 10
        bins = na.arange(
            electrons.value.min()-step/2,
            electrons.value.max()+step/2,
            step=step,
            axis="bin",
        ) * u.electron

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
    temperature = temperature.to(u.K, equivalencies=u.temperature())

    if absorption is None:
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

    if shape_random is None:
        shape_random = dict()

    shape = na.broadcast_shapes(
        na.shape(photons_absorbed),
        na.shape(wavelength),
        na.shape(absorption),
        na.shape(thickness_implant),
        na.shape(cce_backsurface),
        na.shape(temperature),
        shape_random,
    )

    photons_absorbed = na.broadcast_to(photons_absorbed, shape)
    absorption = na.broadcast_to(absorption, shape)
    thickness_implant = na.broadcast_to(thickness_implant, shape)
    cce_backsurface = na.broadcast_to(cce_backsurface, shape)

    if not isinstance(cce_backsurface, u.Quantity):
        cce_backsurface = cce_backsurface << u.dimensionless_unscaled

    pmf_pair = probability_of_n_pairs(wavelength, temperature)
    p_n = pmf_pair.outputs
    n = pmf_pair.inputs

    shape_n = na.broadcast_shapes(shape, n.shape)

    p_n = p_n.broadcast_to(shape_n)
    n = n.broadcast_to(shape_n)

    energy_inf = 1 * u.keV
    energy_pair_inf = energy_inf / quantum_yield_ideal(energy_inf, temperature).value
    fano_inf = fano_factor(energy_inf, temperature)

    wavelength = na.broadcast_to(wavelength, shape)
    energy_pair_inf = energy_pair_inf.broadcast_to(shape)
    fano_inf = fano_inf.broadcast_to(shape)

    result = _electrons_measured_quantity(
        photons_absorbed=photons_absorbed.ndarray,
        wavelength=wavelength.ndarray,
        absorption=absorption.ndarray,
        thickness_implant=thickness_implant.ndarray,
        cce_backsurface=cce_backsurface.ndarray,
        p_n=p_n.ndarray,
        n=n.ndarray,
        energy_pair_inf=energy_pair_inf.ndarray,
        fano_inf=fano_inf.ndarray,
    )

    return na.ScalarArray(
        ndarray=result,
        axes=tuple(shape),
    )


def _electrons_measured_quantity(
    photons_absorbed: u.Quantity,
    wavelength: u.Quantity,
    absorption: u.Quantity,
    thickness_implant: u.Quantity,
    cce_backsurface: u.Quantity,
    p_n: np.ndarray,
    n: np.ndarray,
    energy_pair_inf: u.Quantity,
    fano_inf: u.Quantity,
) -> u.Quantity:

    shape = np.broadcast_shapes(
        photons_absorbed.shape,
        absorption.shape,
        thickness_implant.shape,
        cce_backsurface.shape,
    )

    unit_length = u.mm

    photons_absorbed = photons_absorbed.to_value(u.photon)
    energy = wavelength.to(u.eV, equivalencies=u.spectral())
    absorption = absorption.to_value(1 / unit_length)
    thickness_implant = thickness_implant.to_value(unit_length)
    cce_backsurface = cce_backsurface.to_value(u.dimensionless_unscaled)
    energy_pair_inf = energy_pair_inf.to_value(u.eV)
    fano_inf = fano_inf.to_value(u.electron / u.photon)

    result = _electrons_measured_numba(
        photons_absorbed=photons_absorbed.reshape(-1),
        energy=energy.reshape(-1),
        absorption=absorption.reshape(-1),
        thickness_implant=thickness_implant.reshape(-1),
        cce_backsurface=cce_backsurface.reshape(-1),
        p_n=p_n.reshape(-1, p_n.shape[~0]),
        n=n.reshape(-1, n.shape[~0]),
        energy_pair_inf=energy_pair_inf.reshape(-1),
        fano_inf=fano_inf.reshape(-1),
    )

    result = result.reshape(shape)

    result = result << u.electron

    return result


@numba.njit(fastmath=True, parallel=True)
def _electrons_measured_numba(
    photons_absorbed: np.ndarray,
    energy: np.ndarray,
    absorption: np.ndarray,
    thickness_implant: np.ndarray,
    cce_backsurface: np.ndarray,
    p_n: np.ndarray,
    n: np.ndarray,
    energy_pair_inf: np.ndarray,
    fano_inf: np.ndarray,
) -> np.ndarray:  # pragma: nocover

    num_i, num_n = p_n.shape

    result = np.empty(num_i)

    for i in numba.prange(num_i):

        num_photon = int(photons_absorbed[i])
        energy_i = energy[i]
        a = absorption[i]
        W = thickness_implant[i]
        h_0 = cce_backsurface[i]
        cmf_i = np.cumsum(p_n[i])
        n_i = n[i]
        energy_pair_inf_i = energy_pair_inf[i]
        fano_inf_i = fano_inf[i]

        d = 1 / a

        num_electron_i = 0

        mean_inf = energy_i / energy_pair_inf_i
        std_inf = math.sqrt(fano_inf_i * mean_inf)

        low_energy = energy_i <= 50

        for j in range(num_photon):

            if low_energy:

                x_ij = random.uniform(0, 1)

                for k_ij in range(num_n):
                    if cmf_i[k_ij] > x_ij:
                        break

                n_ij = n_i[k_ij]

            else:
                n_ij = random.normalvariate(
                    mu=mean_inf,
                    sigma=std_inf,
                )
                n_ij = round(n_ij)

            y_ij = random.uniform(0, 1)
            z_ij = -d * math.log(1 - y_ij)

            if z_ij < W:
                h_ij = h_0 + (1 - h_0) * z_ij / W
            else:
                h_ij = 1

            m_ij = np.random.binomial(n=n_ij, p=h_ij)

            num_electron_i = num_electron_i + m_ij

        result[i] = num_electron_i

    return result
