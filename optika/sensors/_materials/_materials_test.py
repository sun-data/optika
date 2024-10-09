import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.materials._tests.test_materials import AbstractTestAbstractMaterial


@pytest.mark.parametrize(
    argnames="wavelength,result_expected",
    argvalues=[
        (1.0 * u.eV, 0 * u.electron / u.photon),
        (2.0 * u.eV, 1 * u.electron / u.photon),
        (2 * optika.sensors.energy_electron_hole, 2 * u.electron / u.photon),
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
        1,
        0.5,
    ],
)
@pytest.mark.parametrize(
    argnames="n",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_oxide",
    argvalues=[
        10 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        10 * u.um,
    ],
)
def test_absorbance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.absorbance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="absorption",
    argvalues=[
        1 / u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        1000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.2,
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="cos_incidence",
    argvalues=[
        1,
    ],
)
def test_charge_collection_efficiency(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
    cos_incidence: float | na.AbstractScalar,
):
    result = optika.sensors.charge_collection_efficiency(
        absorption=absorption,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        cos_incidence=cos_incidence,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 1)


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
        None,
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


@pytest.mark.parametrize(
    argnames="iqy",
    argvalues=[1.61 * u.electron / u.photon],
)
@pytest.mark.parametrize(
    argnames="cce",
    argvalues=[0.9],
)
def test_(
    iqy: u.Quantity | na.AbstractScalar,
    cce: float | na.AbstractScalar,
):
    result = optika.sensors.probability_measurement(
        iqy=iqy,
        cce=cce,
    )
    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="photons",
    argvalues=[100 * u.photon],
)
@pytest.mark.parametrize(
    argnames="absorbance",
    argvalues=[0.75],
)
@pytest.mark.parametrize(
    argnames="iqy",
    argvalues=[1.61 * u.electron / u.photon],
)
@pytest.mark.parametrize(
    argnames="cce",
    argvalues=[0.9],
)
def test_electrons_measured(
    photons: u.Quantity | na.AbstractScalar,
    absorbance: float | na.AbstractScalar,
    iqy: u.Quantity | na.AbstractScalar,
    cce: float | na.AbstractScalar,
):
    result = optika.sensors.electrons_measured(
        photons=photons,
        absorbance=absorbance,
        iqy=iqy,
        cce=cce,
    )
    assert np.all(result >= 0 * u.electron)


class AbstractTestAbstractImagingSensorMaterial(
    AbstractTestAbstractMaterial,
):
    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                intensity=1e-6 * u.erg,
                wavelength=100 * u.AA,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_electrons_measured(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.electrons_measured(rays, normal)
        assert isinstance(result, optika.rays.RayVectorArray)
        assert np.all(result.intensity >= 0 * u.electron)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                intensity=10 * u.electron,
                wavelength=100 * u.AA,
                position=na.Cartesian3dVectorArray() * u.mm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_charge_diffusion(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.charge_diffusion(rays, normal)
        assert isinstance(result, optika.rays.RayVectorArray)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.IdealImagingSensorMaterial(),
    ],
)
class TestIdealImagingSensorMaterial(
    AbstractTestAbstractImagingSensorMaterial,
):
    pass


class AbstractTestAbstractCCDMaterial(
    AbstractTestAbstractImagingSensorMaterial,
):
    def test_fano_noise(self, a: optika.sensors.AbstractCCDMaterial):
        result = a.fano_noise
        assert np.all(result > 0 * u.electron / u.photon)


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

    def test_depletion(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
    ):
        result = a.depletion
        assert isinstance(result, optika.sensors.AbstractDepletionModel)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=100 * u.AA,
                position=na.Cartesian3dVectorArray() * u.mm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_width_charge_diffusion(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.width_charge_diffusion(rays, normal)
        assert np.all(result >= 0 * u.um)

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

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=100 * u.AA,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_absorbance(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.absorbance(rays, normal)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=100 * u.AA,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_charge_collection_efficiency(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.charge_collection_efficiency(rays, normal)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=100 * u.AA,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_quantum_efficiency(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.quantum_efficiency(rays, normal)
        assert result > 0 * u.electron / u.photon

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=100 * u.AA,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_probability_measurement(
        self,
        a: optika.sensors.AbstractBackilluminatedCCDMaterial,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.probability_measurement(rays, normal)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


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
        assert np.all(result.outputs <= 1.1)
        assert np.all(result.inputs >= 0 * u.nm)
