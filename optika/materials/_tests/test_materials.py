import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import optika.rays._tests.test_ray_vectors

_wavelength = na.linspace(100, 300, axis="wavelength", num=11) * u.AA


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
        )
    ],
)
class TestMeasuredMirror(
    AbstractTestAbstractMirror,
):
    pass
