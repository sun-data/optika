import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "quantum_yield_ideal",
    "fano_factor",
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


def _probability_of_n_pairs_ramanathan(
) -> na.FunctionArray[na.CartesianNdVectorArray, na.ScalarArray]:

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


def quantum_yield_ideal(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    pn = _probability_of_n_pairs_ramanathan()

    _n = pn.inputs.components["n"]
    _energy = pn.inputs.components["energy"]
    _temperature = pn.inputs.components["temperature"]
    _probability = pn.outputs

    _iqy = (_n * _probability).sum("num_electron")

    _energy_pair = _energy / _iqy

    slice_he = dict(wavelength=slice(-100, None))
    energy_pair_he = _energy_pair[slice_he].mean("wavelength")

    energy_pair = na.interp(
        x=energy,
        xp=_energy,
        fp=_energy_pair,
        right=energy_pair_he.value,
    )

    energy_pair = na.interp(
        x=temperature,
        xp=_temperature,
        fp=energy_pair,
    )

    iqy = energy / energy_pair

    return iqy * u.electron / u.photon


def fano_factor(
    wavelength: u.Quantity | na.ScalarArray,
    temperature: u.Quantity | na.ScalarArray = 300 * u.K,
) -> na.ScalarArray:

    energy = wavelength.to(u.eV, equivalencies=u.spectral())

    pn = _probability_of_n_pairs_ramanathan()

    _n = pn.inputs.components["n"]
    _energy = pn.inputs.components["energy"]
    _temperature = pn.inputs.components["temperature"]
    _probability = pn.outputs

    _iqy = (_n * _probability).sum("num_electron")

    _v = (np.square(_n) * _probability).sum("num_electron")
    _fano_factor = (_v - np.square(_iqy)) / _iqy

    fano_he = np.nanmedian(_fano_factor, axis="wavelength")

    fano_factor = na.interp(
        x=energy,
        xp=_energy,
        fp=_fano_factor,
        right=fano_he,
    )

    fano_factor = na.interp(
        x=temperature,
        xp=_temperature,
        fp=fano_factor,
    )

    return fano_factor * u.electron / u.photon
