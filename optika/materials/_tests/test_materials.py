import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import optika.rays._tests.test_ray_vectors

_wavelength = na.linspace(100, 300, axis="wavelength", num=11) * u.AA

_angle = na.linspace(0, 40, axis="angle", num=3) * u.deg


class AbstractTestAbstractMaterial(
    optika._tests.test_mixins.AbstractTestTransformable,
    optika._tests.test_mixins.AbstractTestShaped,
):
    def test_is_mirror(self, a: optika.materials.AbstractMaterial):
        assert isinstance(a.is_mirror, bool)

    @pytest.mark.parametrize("rays", optika.rays._tests.test_ray_vectors.rays)
    class TestRayDependentMethods:
        def test_index_refraction(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
        ):
            result = a.index_refraction(rays)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)

        def test_attenuation(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
        ):
            result = a.attenuation(rays)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert na.unit_normalized(result).is_equivalent(1 / u.mm)

        @pytest.mark.parametrize(
            argnames="normal",
            argvalues=[
                na.Cartesian3dVectorArray(0, 0, -1),
            ],
        )
        def test_efficiency(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
            normal: na.AbstractCartesian3dVectorArray,
        ):
            result = a.efficiency(rays, normal)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
            assert np.all(result >= 0)
            assert np.all(result <= 1)


@pytest.mark.parametrize("a", [optika.materials.Vacuum()])
class TestVacuum(
    AbstractTestAbstractMaterial,
):
    pass


class AbstractTestAbstractMirror(
    AbstractTestAbstractMaterial,
):
    def test_substrate(self, a: optika.materials.AbstractMirror):
        result = a.substrate
        if result is not None:
            assert isinstance(result, optika.materials.Layer)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Mirror(),
        optika.materials.Mirror(
            substrate=optika.materials.Layer(
                chemical="SiO2",
                thickness=10 * u.mm,
            ),
        ),
    ],
)
class TestMirror(
    AbstractTestAbstractMirror,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.MeasuredMirror(
            efficiency_measured=na.FunctionArray(
                inputs=na.SpectralDirectionalVectorArray(
                    wavelength=_wavelength,
                    direction=na.Cartesian3dVectorArray(0, 0, 1),
                ),
                outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2),
            ),
        ),
        optika.materials.MeasuredMirror(
            efficiency_measured=na.FunctionArray(
                inputs=na.SpectralDirectionalVectorArray(
                    wavelength=_wavelength,
                    direction=_angle,
                ),
                outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2)
                * np.cos(_angle),
            ),
            axis_angle="angle",
        ),
    ],
)
class TestMeasuredMirror(
    AbstractTestAbstractMirror,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Glass(),
        optika.materials.Glass.n_bk7(),
        optika.materials.Glass.f2(),
    ],
)
class TestGlass(
    AbstractTestAbstractMaterial,
):
    pass


@pytest.mark.parametrize(
    argnames="glass,n_d",
    argvalues=[
        (optika.materials.Glass.n_bk7(), 1.5168),
        (optika.materials.Glass.f2(), 1.6200),
    ],
)
def test_glass_dispersion(
    glass: optika.materials.Glass,
    n_d: float,
):
    # index of refraction at the helium d Fraunhofer line should match the
    # published value of the glass.
    rays_d = optika.rays.RayVectorArray(wavelength=587.5618 * u.nm)
    n = glass.index_refraction(rays_d)
    assert np.isclose(float(n), n_d, atol=1e-3)

    # the glass must be dispersive: a higher index toward the blue end of the
    # spectrum (normal dispersion).
    rays_F = optika.rays.RayVectorArray(wavelength=486.1327 * u.nm)
    rays_C = optika.rays.RayVectorArray(wavelength=656.2725 * u.nm)
    assert glass.index_refraction(rays_F) > glass.index_refraction(rays_C)

    # a glass transmits rather than reflects.
    assert not glass.is_mirror


_efficiency_measured = na.FunctionArray(
    inputs=na.SpectralDirectionalVectorArray(
        wavelength=_wavelength,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    ),
    outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2),
)

_efficiency_measured_angle = na.FunctionArray(
    inputs=na.SpectralDirectionalVectorArray(
        wavelength=_wavelength,
        direction=_angle,
    ),
    outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2) * np.cos(_angle),
)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Dielectric("MgF2"),
        optika.materials.Dielectric(
            chemical=optika.chemicals.Chemical("SiO2"),
        ),
    ],
)
class TestDielectric(
    AbstractTestAbstractMaterial,
):
    pass


def test_dielectric_mgf2():
    mgf2 = optika.materials.Dielectric("MgF2")

    # the index of refraction at Lyman alpha should match the tabulated
    # ordinary-ray value from Palik.
    rays = optika.rays.RayVectorArray(wavelength=121.6 * u.nm)
    n = mgf2.index_refraction(rays)
    assert np.isclose(n.ndarray, 1.630, atol=1e-2)

    # the material must be dispersive: a higher index toward the blue end of
    # the spectrum (normal dispersion).
    rays_red = optika.rays.RayVectorArray(wavelength=250 * u.nm)
    assert mgf2.index_refraction(rays) > mgf2.index_refraction(rays_red)

    # magnesium fluoride is transparent in its transmission window, but
    # absorbs strongly below the band edge.
    assert mgf2.attenuation(rays_red) == 0 / u.mm
    rays_blue = optika.rays.RayVectorArray(wavelength=100 * u.nm)
    assert mgf2.attenuation(rays_blue) > 0 / u.mm

    # a dielectric transmits rather than reflects.
    assert not mgf2.is_mirror

    # a formula and the equivalent chemical give the same index
    chemical = optika.chemicals.Chemical("MgF2")
    assert optika.materials.Dielectric(chemical).index_refraction(rays) == n


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.MeasuredFilter(
            efficiency_measured=_efficiency_measured,
        ),
        optika.materials.MeasuredFilter(
            efficiency_measured=_efficiency_measured,
            medium=optika.materials.Dielectric("MgF2"),
        ),
        optika.materials.MeasuredFilter(
            efficiency_measured=_efficiency_measured,
            medium=optika.materials.Glass.n_bk7(),
        ),
        optika.materials.MeasuredFilter(
            efficiency_measured=_efficiency_measured,
            medium=optika.materials.Dielectric("MgF2"),
            is_medium_measured=False,
        ),
        optika.materials.MeasuredFilter(
            efficiency_measured=_efficiency_measured_angle,
            axis_angle="angle",
            medium=optika.materials.Dielectric("MgF2"),
        ),
    ],
)
class TestMeasuredFilter(
    AbstractTestAbstractMaterial,
):
    def test_medium(self, a: optika.materials.MeasuredFilter):
        assert isinstance(a.medium, optika.materials.AbstractMaterial)
        assert not a.medium.is_mirror


def test_measured_filter_window():
    """
    Trace a ray obliquely through a coated window modeled as a pair of
    surfaces and check it against the analytic result for a parallel plate.
    """
    wavelength = na.linspace(120, 250, axis="wavelength", num=14) * u.nm
    efficiency_measured = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        ),
        outputs=0.2 * np.exp(-np.square((wavelength - 170 * u.nm) / (50 * u.nm))),
    )
    medium = optika.materials.Dielectric("MgF2")
    material = optika.materials.MeasuredFilter(
        efficiency_measured=efficiency_measured,
        medium=medium,
    )

    thickness = 2 * u.mm
    front = optika.surfaces.Surface(
        material=material,
    )
    back = optika.surfaces.Surface(
        material=optika.materials.Vacuum(),
        transformation=na.transformations.Cartesian3dTranslation(z=thickness),
    )

    angle = 30 * u.deg
    rays = optika.rays.RayVectorArray(
        wavelength=na.linspace(125, 245, axis="wavelength", num=7) * u.nm,
        position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
    )

    rays_front = front.propagate_rays(rays)
    rays_back = back.propagate_rays(rays_front)

    # the transmissivity is applied once, at the front surface, and the
    # tabulated absorption of magnesium fluoride is zero in this band.
    efficiency = material.efficiency(rays, normal=na.Cartesian3dVectorArray(0, 0, -1))
    assert np.allclose(rays_front.intensity, efficiency)
    assert np.allclose(rays_back.intensity, efficiency)
    assert np.all(rays_front.index_refraction == medium.index_refraction(rays))

    # a ray exits a parallel plate travelling in its original direction,
    # displaced by the refraction inside the plate.
    n = medium.index_refraction(rays)
    x = thickness * np.sin(angle) / np.sqrt(np.square(n) - np.square(np.sin(angle)))
    assert np.allclose(rays_back.direction.x, rays.direction.x)
    assert np.allclose(rays_back.direction.z, rays.direction.z)
    assert np.allclose(rays_back.position.x, x)
    assert np.allclose(rays_back.position.z, thickness)
    assert np.all(rays_back.index_refraction == 1)

    # the shorter wavelengths see a higher index and are displaced less
    assert (
        rays_back.position.x[dict(wavelength=0)]
        < rays_back.position.x[dict(wavelength=~0)]
    )


@pytest.mark.parametrize("is_medium_measured", [True, False])
def test_measured_filter_is_medium_measured(is_medium_measured: bool):
    """
    Trace a ray through a window of an absorbing medium and check that the
    absorption of the medium is only applied when the measurement does not
    already include it.
    """
    wavelength = na.linspace(90, 110, axis="wavelength", num=3) * u.nm
    efficiency_measured = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        ),
        outputs=na.ScalarArray(np.array([0.3, 0.5, 0.7]), axes="wavelength"),
    )

    # magnesium fluoride absorbs strongly below its band edge
    medium = optika.materials.Dielectric("MgF2")
    material = optika.materials.MeasuredFilter(
        efficiency_measured=efficiency_measured,
        medium=medium,
        is_medium_measured=is_medium_measured,
    )

    thickness = 10 * u.nm
    front = optika.surfaces.Surface(material=material)
    back = optika.surfaces.Surface(
        material=optika.materials.Vacuum(),
        transformation=na.transformations.Cartesian3dTranslation(z=thickness),
    )

    rays = optika.rays.RayVectorArray(
        wavelength=100 * u.nm,
        position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )
    rays_back = back.propagate_rays(front.propagate_rays(rays))

    normal = na.Cartesian3dVectorArray(0, 0, -1)
    efficiency = material.efficiency(rays, normal)
    attenuation = medium.attenuation(rays)
    assert attenuation > 0 / u.mm

    if is_medium_measured:
        assert material.attenuation(rays) == 0 / u.mm
        assert np.allclose(rays_back.intensity, efficiency)
    else:
        assert material.attenuation(rays) == attenuation
        transmission = np.exp(-(attenuation * thickness).to(u.dimensionless_unscaled))
        assert transmission < 0.9
        assert np.allclose(rays_back.intensity, efficiency * transmission)
