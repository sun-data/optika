import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_transforms
import optika.rays._tests.test_ray_vectors


class AbstractTestAbstractMaterial(
    optika._tests.test_transforms.AbstractTestTransformable,
):
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

        def test_transmissivity(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
        ):
            result = a.transmissivity(rays)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)

        @pytest.mark.parametrize(
            argnames="sag",
            argvalues=[
                optika.sags.NoSag(),
            ],
        )
        @pytest.mark.parametrize(
            argnames="rulings",
            argvalues=[
                None,
            ],
        )
        def test_refract_rays(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
            sag: optika.sags.AbstractSag,
            rulings: None | optika.rulings.AbstractRulings,
        ):
            result = a.refract_rays(
                rays=rays,
                sag=sag,
                rulings=rulings,
            )
            assert isinstance(result, optika.rays.AbstractRayVectorArray)
            assert np.all(result.index_refraction == a.index_refraction(rays))
            assert np.all(result.attenuation == a.attenuation(rays))


@pytest.mark.parametrize("a", [optika.materials.Vacuum()])
class TestVacuum(
    AbstractTestAbstractMaterial,
):
    pass


@pytest.mark.parametrize("a", [optika.materials.Mirror()])
class TestMirror(
    AbstractTestAbstractMaterial,
):
    pass
