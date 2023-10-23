import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import optika.rays._tests.test_ray_vectors


class AbstractTestAbstractMaterial(
    optika._tests.test_mixins.AbstractTestTransformable,
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

        def test_transmissivity(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
        ):
            result = a.transmissivity(rays)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)


@pytest.mark.parametrize("a", [optika.materials.Vacuum()])
class TestVacuum(
    AbstractTestAbstractMaterial,
):
    pass


class AbstractTestAbstractMirror(
    AbstractTestAbstractMaterial,
):
    def test_thickness_substrate(self, a: optika.materials.AbstractMirror):
        result = a.thickness_substrate
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Mirror(),
        optika.materials.Mirror(
            thickness_substrate=10 * u.mm,
        ),
    ],
)
class TestMirror(
    AbstractTestAbstractMirror,
):
    pass
