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
def test_quantum_yield_ideal(
    wavelength: u.Quantity | na.AbstractScalar, result_expected: na.AbstractScalar
):
    result = optika.sensors.quantum_yield_ideal(wavelength)
    assert np.all(result == result_expected)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        na.linspace(100, 200, axis="wavelength", num=4) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, 1),
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_oxide",
    argvalues=[
        10 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        1000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        1 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.2,
        1,
    ],
)
def test_quantum_efficiency_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.quantum_efficiency_effective(
        wavelength=wavelength,
        direction=direction,
        thickness_oxide=thickness_oxide,
        thickness_implant=thickness_implant,
        thickness_substrate=thickness_substrate,
        cce_backsurface=cce_backsurface,
    )
    assert np.all(result >= 0)
    assert np.all(result <= 1)
