import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "probability_of_n_pairs",
    "quantum_yield_ideal",
    "fano_factor",
]


def _probability_of_n_pairs_from_file(
    path: pathlib.Path,
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:

    a = np.loadtxt(path)
    a = na.ScalarArray(a, axes=("wavelength", "num_electron"))
    energy = a[dict(num_electron=0)] << u.eV
    pn = a[dict(num_electron=slice(1, None))]

    return na.FunctionArray(
        inputs=energy,
        outputs=pn,
    )


directory = pathlib.Path(__file__).parent
pn_000K = _probability_of_n_pairs_from_file(directory / "p0K.dat")
pn_100K = _probability_of_n_pairs_from_file(directory / "p100K.dat")
pn_300K = _probability_of_n_pairs_from_file(directory / "p300K.dat")


def _probability_of_n_pairs_temperature_only(
    temperature: u.Quantity | na.ScalarArray,
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:

    pn = na.stack(
        arrays=[pn_000K, pn_100K, pn_300K],
        axis="temperature",
    )

    if np.all(pn.inputs[dict(temperature=0)] == pn.inputs):
        pn.inputs = pn.inputs[dict(temperature=0)]
    else:
        raise ValueError("inputs don't match along temperature axis")

    _temperature = na.ScalarArray(
        ndarray=[0, 100, 300] * u.K,
        axes="temperature",
    )

    pn.outputs = na.interp(
        x=temperature,
        xp=_temperature,
        fp=pn.outputs,
    )

    return pn


def probability_of_n_pairs(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray,
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:

    pn = _probability_of_n_pairs_temperature_only(temperature)

    n = na.arange(1, pn.shape["num_electron"] + 1, axis="num_electron")

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    return na.FunctionArray(
        inputs=n,
        outputs=na.interp(
            x=energy,
            xp=pn.inputs,
            fp=pn.outputs,
        ),
    )


def quantum_yield_ideal(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    pn = _probability_of_n_pairs_temperature_only(temperature)

    n = na.arange(1, pn.shape["num_electron"] + 1, axis="num_electron")

    _energy = pn.inputs

    _iqy = (n * pn.outputs).sum("num_electron")

    _energy_pair = _energy / _iqy

    slice_he = dict(wavelength=slice(-100, None))
    energy_pair_he = _energy_pair[slice_he].mean("wavelength")

    energy_pair = na.interp(
        x=energy,
        xp=_energy,
        fp=_energy_pair,
        right=energy_pair_he.value,
    )
    iqy = energy / energy_pair

    return iqy


def fano_factor(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    pn = _probability_of_n_pairs_temperature_only(temperature)

    n = na.arange(1, pn.shape["num_electron"] + 1, axis="num_electron")

    _energy = pn.inputs

    _iqy = (n * pn.outputs).sum("num_electron")

    v = (np.square(n) * pn.outputs).sum("num_electron")
    _fano_factor = (v - np.square(_iqy)) / _iqy

    fano_he = np.nanmedian(_fano_factor, axis="wavelength")

    fano_factor = na.interp(
        x=energy,
        xp=_energy,
        fp=_fano_factor,
        right=fano_he,
    )

    return fano_factor
