import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_transforms


class AbstractTestAbstractRulings(
    optika._tests.test_transforms.AbstractTestTransformable,
):
    def test_diffraction_order(self, a: optika.rulings.AbstractRulings):
        assert na.get_dtype(a.diffraction_order) == int

    def test_ruling_normal(self, a: optika.rulings.AbstractRulings):
        position = na.Cartesian3dVectorArray() * u.mm
        result = a.ruling_normal(position)
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        assert np.all(result.length == 1)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=500 * u.nm,
                position=na.Cartesian3dVectorArray(
                    x=na.linspace(-5, 5, axis="x", num=3) * u.mm,
                    y=na.linspace(-5, 5, axis="y", num=4) * u.mm,
                    z=0 * u.mm,
                ),
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            )
        ],
    )
    def test_rays_apparent(
        self,
        a: optika.rulings.AbstractRulings,
        rays: optika.rays.RayVectorArray,
    ):
        result = a.rays_apparent(rays, index_refraction=1)
        assert isinstance(result, optika.rays.AbstractRayVectorArray)
        assert np.all(result.wavelength == rays.wavelength)
        assert np.all(result.position == rays.position)
        assert np.any(result.direction.z != rays.direction.z)
        assert np.allclose(result.direction.length, 1)
        assert np.all(result.intensity == rays.intensity)
        assert np.any(result.index_refraction != rays.index_refraction)


class AbstractTestAbstractConstantDensityRulings(
    AbstractTestAbstractRulings,
):
    def test_ruling_density(
        self,
        a: optika.rulings.AbstractConstantDensityRulings,
    ):
        assert a.ruling_density.unit.is_equivalent(1 / u.mm)

    def test_ruling_spacing(
        self,
        a: optika.rulings.AbstractConstantDensityRulings,
    ):
        assert a.ruling_spacing.unit.is_equivalent(u.mm)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.ConstantDensityRulings(
            ruling_density=5000 / u.mm,
            diffraction_order=1,
        ),
        optika.rulings.ConstantDensityRulings(
            ruling_density=na.linspace(1, 5, axis="rulings", num=4) / u.mm,
            diffraction_order=na.ScalarArray(np.array([-1, 0, 1]), axes="m"),
        ),
    ],
)
class TestConstantDensityRulings(
    AbstractTestAbstractConstantDensityRulings,
):
    pass
