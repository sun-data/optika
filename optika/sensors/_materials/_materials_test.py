import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.materials._tests.test_materials import AbstractTestAbstractMaterial


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


class AbstractTestAbstractImagingSensorMaterial(
    AbstractTestAbstractMaterial,
):
    pass


class AbstractTestAbstractCCDMaterial(
    AbstractTestAbstractImagingSensorMaterial,
):
    pass


class AbstractTestAbstractBackilluminatedCCDMaterial(
    AbstractTestAbstractCCDMaterial,
):
    def test_thickness_oxide(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
    ):
        result = a.thickness_oxide
        assert result >= 0 * u.mm

    def test_thickness_implant(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
    ):
        result = a.thickness_implant
        assert result >= 0 * u.mm

    def test_thickness_substrate(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
    ):
        result = a.thickness_substrate
        assert result >= 0 * u.mm

    def test_cce_backsurface(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
    ):
        result = a.cce_backsurface
        assert result >= 0

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            500 * u.nm,
        ],
    )
    def test_quantum_yield_ideal(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.quantum_yield_ideal(wavelength)
        assert result >= 0


class AbstractTestAbstractStern1994BackilluminatedCCDMaterial(
    AbstractTestAbstractBackilluminatedCCDMaterial,
):
    def test_quantum_efficiency_measured(
        self,
        a: optika.sensors.AbstractStern1994BackilluminatedCCDMaterial,
    ):
        result = a.quantum_efficiency_measured
        assert isinstance(result, na.AbstractFunctionArray)
        assert np.all(result.outputs >= 0)
        assert np.all(result.outputs <= 1)
        assert np.all(result.inputs >= 0 * u.nm)
