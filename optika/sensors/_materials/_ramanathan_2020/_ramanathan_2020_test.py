import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from . import _ramanathan_2020


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        300 * u.nm,
        5 * u.eV,
        na.geomspace(1, 10000, axis="wavelength", num=7) << u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="temperature",
    argvalues=[
        300 * u.K,
        na.linspace(0, 300, axis="temperature", num=5) << u.K,
    ]
)
def test_quantum_yield_ideal(
    wavelength: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.AbstractScalar,
):
    result = _ramanathan_2020.quantum_yield_ideal(
        wavelength=wavelength,
        temperature=temperature,
    )

    print(f"{result=}")

    assert np.all(result >= 0)
    assert result.shape == na.shape_broadcasted(wavelength, temperature)
