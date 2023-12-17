import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import optika.rays._tests.test_ray_vectors


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        350 * u.nm,
        na.linspace(300 * u.nm, 400 * u.nm, axis="wavelength", num=3),
    ],
)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, 1),
    ],
)
@pytest.mark.parametrize(
    argnames="index_refraction",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="index_refraction_new",
    argvalues=[
        1.5,
        na.linspace(1, 2, axis="index_refraction_new", num=4),
    ],
)
@pytest.mark.parametrize(
    argnames="normal",
    argvalues=[
        None,
    ],
)
@pytest.mark.parametrize(
    argnames="is_mirror",
    argvalues=[
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    argnames="diffraction_order,spacing_rulings,normal_rulings",
    argvalues=[
        (0, None, None),
        (1, 5 * u.um, na.Cartesian3dVectorArray(1, 0, 0)),
    ],
)
def test_snells_law(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    index_refraction: float | na.AbstractScalar,
    index_refraction_new: float | na.AbstractScalar,
    normal: None | na.AbstractCartesian3dVectorArray,
    is_mirror: bool | na.AbstractScalar,
    diffraction_order: int,
    spacing_rulings: None | u.Quantity | na.AbstractScalar,
    normal_rulings: None | na.AbstractCartesian3dVectorArray,
):
    result = optika.materials.snells_law(
        wavelength=wavelength,
        direction=direction,
        index_refraction=index_refraction,
        index_refraction_new=index_refraction_new,
        normal=normal,
        is_mirror=is_mirror,
        diffraction_order=diffraction_order,
        spacing_rulings=spacing_rulings,
        normal_rulings=normal_rulings,
    )
    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    assert isinstance(result, na.AbstractCartesian3dVectorArray)
    assert np.allclose(result.length, 1)
    if is_mirror:
        assert not np.allclose(np.sign(direction @ normal), np.sign(result @ normal))
    else:
        assert np.allclose(np.sign(direction @ normal), np.sign(result @ normal))


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

        @pytest.mark.parametrize(
            argnames="normal",
            argvalues=[
                na.Cartesian3dVectorArray(0, 0, -1),
            ],
        )
        def test_transmissivity(
            self,
            a: optika.materials.AbstractMaterial,
            rays: optika.rays.AbstractRayVectorArray,
            normal: na.AbstractCartesian3dVectorArray,
        ):
            result = a.transmissivity(rays, normal)
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
