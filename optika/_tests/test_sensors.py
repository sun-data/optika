import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize(
    argnames="wavelength,result_expected",
    argvalues=[
        (1.0 * u.eV, 0),
        (2.0 * u.eV, 1),
        (2 * optika.sensors.energy_electron_hole, 2),
    ],
)
def test_quantum_yield_naive(
    wavelength: u.Quantity | na.AbstractScalar, result_expected: na.AbstractScalar
):
    result = optika.sensors.quantum_yield_naive(wavelength)
    assert np.all(result == result_expected)
