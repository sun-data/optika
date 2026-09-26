import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.materials._tests.test_materials import AbstractTestAbstractMaterial


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
def test_transmittance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.transmittance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
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
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "Beer-Lambert",
        "exact",
    ],
)
def test_absorbance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    method: str,
):
    result = optika.sensors.absorbance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
        method=method,
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
def test_charge_collection_efficiency(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.charge_collection_efficiency(
        absorption=absorption,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
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
        1,
        0.5,
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
    direction: float | na.AbstractScalar,
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
def test_probability_measurement(
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
    argnames="photons_absorbed",
    argvalues=[
        (100 * u.photon).astype(int),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        100 * u.nm,
        na.geomspace(1, 10000, axis="wavelength", num=5) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        2000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.5,
    ],
)
@pytest.mark.parametrize("temperature", [300 * u.K])
def test_electrons_measured_approx(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.ScalarArray,
):
    result = optika.sensors.electrons_measured_approx(
        photons_absorbed=photons_absorbed,
        wavelength=wavelength,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron)

    shape = na.shape_broadcasted(
        photons_absorbed,
        wavelength,
        thickness_implant,
        cce_backsurface,
        temperature,
    )

    assert result.shape == shape


@pytest.mark.parametrize(
    argnames="photons_expected",
    argvalues=[
        100 * u.photon,
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1000 * u.AA,
        na.geomspace(1, 10000, axis="wavelength", num=9) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "monte-carlo",
        "expected",
    ],
)
def test_signal(
    photons_expected: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.AbstractScalar,
    method: str,
):
    result = optika.sensors.signal(
        photons_expected=photons_expected,
        wavelength=wavelength,
        method=method,
    )
    assert np.all(result >= 0 * u.electron)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1000 * u.AA,
        na.geomspace(1, 10000, axis="wavelength", num=9) * u.AA,
    ],
)
def test_vmr_signal(
    wavelength: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.vmr_signal(
        wavelength=wavelength,
    )
    assert np.all(result >= 0 * u.electron)


def test_vmr_signal_diffusion():
    wavelength = 304 * u.AA
    thickness_depletion = 2 * u.um
    axis_xy = ("detector_x", "detector_y")

    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=16, detector_y=16),
    )

    signal = optika.sensors.signal(
        photons_expected=photons_expected,
        wavelength=wavelength,
        thickness_depletion=thickness_depletion,
        axis_xy=axis_xy,
        wrap=True,
        shape_random=dict(experiment=500),
    )

    vmr_measured = signal.vmr(("experiment",) + axis_xy)

    result = optika.sensors.vmr_signal(
        wavelength=wavelength,
        thickness_depletion=thickness_depletion,
    )
    result_no_diffusion = optika.sensors.vmr_signal(
        wavelength=wavelength,
        thickness_depletion=thickness_depletion,
        diffusion=False,
    )

    assert np.all(result > 0 * u.electron)
    assert np.all(result < result_no_diffusion)
    assert np.abs(vmr_measured - result) < 0.1 * result

    result_default = optika.sensors.vmr_signal(wavelength=wavelength)
    result_default_no_diffusion = optika.sensors.vmr_signal(
        wavelength=wavelength,
        diffusion=False,
    )
    assert np.all(result_default == result_default_no_diffusion)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        20 * u.AA,
    ],
)
def test_vmr_signal_depleted(
    wavelength: u.Quantity | na.AbstractScalar,
):
    """
    On a fully depleted sensor the only spread is acquired drifting across the
    depletion region, which the analytic VMR must reproduce, for photons
    absorbed at the back surface and for photons absorbed throughout the
    depletion region.
    """
    axis_xy = ("detector_x", "detector_y")

    kwargs = dict(
        wavelength=wavelength,
        thickness_depletion=14 * u.um,
        thickness_substrate=14 * u.um,
        width_depleted=2 * u.um,
        width_pixel=4 * u.um,
    )

    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=16, detector_y=16),
    )

    signal = optika.sensors.signal(
        photons_expected=photons_expected,
        axis_xy=axis_xy,
        wrap=True,
        shape_random=dict(experiment=200),
        **kwargs,
    )

    vmr_measured = signal.vmr(("experiment",) + axis_xy)

    result = optika.sensors.vmr_signal(**kwargs)
    result_sharp = optika.sensors.vmr_signal(**(kwargs | dict(width_depleted=None)))

    assert np.all(result < result_sharp)
    assert np.abs(vmr_measured - result) < 0.05 * result


def test_vmr_signal_widths_default():
    """
    Janesick's values of the new widths reproduce the default exactly.
    """
    ccd = optika.sensors.materials.e2v_ccd97()

    kwargs = dict(
        wavelength=na.geomspace(1, 10000, axis="wavelength", num=11) * u.AA,
        thickness_depletion=ccd.depletion.thickness,
        thickness_substrate=ccd.thickness_substrate,
        width_pixel=16 * u.um,
    )

    result = optika.sensors.vmr_signal(**kwargs)
    explicit = optika.sensors.vmr_signal(
        **kwargs,
        width_max=ccd.thickness_substrate - ccd.depletion.thickness,
        width_depleted=0 * u.um,
    )

    assert np.all(result == explicit)


@pytest.mark.parametrize(
    argnames="width_max,width_depleted",
    argvalues=[
        (None, None),
        (None, 0.8 * u.um),
        (4 * u.um, 3 * u.um),
    ],
)
def test_vmr_signal_quadrature(
    width_max: None | u.Quantity | na.AbstractScalar,
    width_depleted: None | u.Quantity | na.AbstractScalar,
):
    """
    The charge-diffusion integral should be converged at the default number of
    quadrature nodes, across the full range of optical depths that silicon
    spans between 1 and 10000 angstroms, over the depletion region as well as
    the field-free region.
    """
    ccd = optika.sensors.materials.e2v_ccd97()

    kwargs = dict(
        wavelength=na.geomspace(1, 10000, axis="wavelength", num=101) * u.AA,
        thickness_implant=ccd.thickness_implant,
        thickness_depletion=ccd.depletion.thickness,
        thickness_substrate=ccd.thickness_substrate,
        width_max=width_max,
        width_depleted=width_depleted,
        width_pixel=16 * u.um,
        cce_backsurface=ccd.cce_backsurface,
        temperature=ccd.temperature,
    )

    result = optika.sensors.vmr_signal(**kwargs)

    num = optika.sensors.materials._materials._num_gauss_legendre
    try:
        optika.sensors.materials._materials._num_gauss_legendre = 8 * num
        expected = optika.sensors.vmr_signal(**kwargs)
    finally:
        optika.sensors.materials._materials._num_gauss_legendre = num

    assert np.all(np.abs(result / expected - 1) < 1e-5)


class AbstractTestAbstractSensorMaterial(
    AbstractTestAbstractMaterial,
):
    @pytest.mark.parametrize(
        argnames="photons",
        argvalues=[
            1e-6 * u.erg,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
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
        argnames="noise",
        argvalues=[True, False],
    )
    def test_signal(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        noise: bool,
    ):
        result = a.signal(
            photons=photons,
            wavelength=wavelength,
            direction=direction,
            noise=noise,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result >= 0 * u.electron)

    @pytest.mark.parametrize(
        argnames="electrons",
        argvalues=[
            100 * u.electron,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, 1),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_photons_incident(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.photons_incident(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.photon)

    @pytest.mark.parametrize(
        argnames="photons",
        argvalues=[
            100 * u.photon,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    def test_photons_absorbed(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
    ):
        # `photons_absorbed` inverts the noiseless `signal` (which uses unit
        # absorbance), recovering the number of absorbed photons.
        electrons = a.signal(
            photons=photons,
            wavelength=wavelength,
            direction=direction,
            noise=False,
        )
        result = a.photons_absorbed(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.photon)
        assert np.allclose(
            na.as_named_array(result / photons).ndarray.to_value(
                u.dimensionless_unscaled
            ),
            1,
        )

    @pytest.mark.parametrize(
        argnames="electrons",
        argvalues=[
            1000 * u.electron,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    def test_uncertainty(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
    ):
        result = a.uncertainty(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.electron)
        assert np.all(result >= 0 * u.electron)

        result_diffusion = a.uncertainty(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
            width_pixel=15 * u.um,
        )
        assert np.all(result_diffusion <= result)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    def test_direction_refracted(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        # with the default direction and normal (normal incidence) the
        # refracted cosine is unity
        result = a.direction_refracted(wavelength=wavelength)
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(np.real(result) > 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.IdealSensorMaterial(),
    ],
)
class TestIdealSensorMaterial(
    AbstractTestAbstractSensorMaterial,
):
    pass


class AbstractTestAbstractSiliconSensorMaterial(
    AbstractTestAbstractSensorMaterial,
):

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            500 * u.nm,
        ],
    )
    def test_quantum_yield_ideal(
        self,
        a: optika.sensors.materials.AbstractSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.quantum_yield_ideal(wavelength)
        assert result >= 0

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            3 * u.eV,
            na.geomspace(1, 10000, axis="wavelength", num=5) * u.AA,
        ],
    )
    def test_fano_factor(
        self,
        a: optika.sensors.materials.AbstractSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.fano_factor(wavelength)
        assert np.all(result >= 0 * u.electron / u.photon)


class AbstractTestAbstractBackIlluminatedSiliconSensorMaterial(
    AbstractTestAbstractSiliconSensorMaterial,
):
    def test_thickness_oxide(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_oxide
        assert result >= 0 * u.mm

    def test_thickness_implant(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_implant
        assert result >= 0 * u.mm

    def test_thickness_substrate(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_substrate
        assert result >= 0 * u.mm

    def test_cce_backsurface(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.cce_backsurface
        assert result >= 0

    def test_num_interpolation(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.num_interpolation
        assert (result is None) or (result > 0)

    def test_efficiency_interpolated(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        """
        Interpolating the absorbance over the angle of incidence reproduces
        computing it for every ray, and does so better the more nodes it is
        given.
        """
        angle = na.linspace(0, 20, axis="angle", num=25) * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=na.linspace(100, 1000, axis="wavelength", num=5) * u.AA,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        expected = a.efficiency(rays, normal)
        scale = np.abs(expected).max()

        def error(num: int) -> float:
            result = dataclasses.replace(a, num_interpolation=num).efficiency(
                rays=rays,
                normal=normal,
            )
            assert na.shape(result) == na.shape(expected)
            return np.abs(result - expected).max() / scale

        coarse, fine = error(4), error(32)

        assert coarse < 0.01
        assert fine <= coarse

    def test_efficiency_interpolated_scalar(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        """With a single angle of incidence there is nothing to interpolate
        over, so the absorbance is computed directly."""
        rays = optika.rays.RayVectorArray(
            wavelength=200 * u.AA,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        result = dataclasses.replace(a, num_interpolation=8).efficiency(
            rays=rays,
            normal=normal,
        )

        assert np.all(result == a.efficiency(rays, normal))

    def test_depletion(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.depletion
        assert isinstance(
            result,
            optika.sensors.materials.depletion.AbstractDepletionModel,
        )

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    def test_width_charge_diffusion(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.width_charge_diffusion(wavelength=wavelength)
        assert np.all(result >= 0 * u.um)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_transmittance(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.transmittance(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_absorbance(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.absorbance(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_charge_collection_efficiency(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.charge_collection_efficiency(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_quantum_efficiency(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.quantum_efficiency(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert result > 0 * u.electron / u.photon

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_probability_measurement(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.probability_measurement(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="photons_absorbed",
        argvalues=[
            (100 * u.photon).astype(int),
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_electrons_measured(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        photons_absorbed: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.electrons_measured(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result >= 0 * u.electron)


def _e2v_ccd97_widths() -> (
    optika.sensors.materials.BackIlluminatedSiliconSensorMaterial
):
    """An e2v CCD97 whose charge spreads in the depletion region too."""
    result = optika.sensors.materials.e2v_ccd97()
    return result.replace(
        depletion=result.depletion.replace(
            width_max=5 * u.um,
            width_depleted=0.8 * u.um,
        ),
    )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.tektronix_tk512cb(),
        optika.sensors.materials.e2v_ccd97(),
        optika.sensors.materials.e2v_ccd203(),
        _e2v_ccd97_widths(),
    ],
)
class TestBackIlluminatedSiliconSensorMaterial(
    AbstractTestAbstractBackIlluminatedSiliconSensorMaterial,
):
    def test_eqe_measured(
        self,
        a: optika.sensors.materials.BackIlluminatedSiliconSensorMaterial,
    ):
        result = a.eqe_measured
        assert isinstance(result, na.AbstractFunctionArray)
        assert np.all(result.outputs >= 0)
        assert np.all(result.outputs <= 1.1)
        assert np.all(result.inputs >= 0 * u.nm)
