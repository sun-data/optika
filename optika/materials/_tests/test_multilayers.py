import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import optika
from . import test_materials


@pytest.mark.parametrize(
    argnames="material_layers,thickness_layers,axis_layers",
    argvalues=[
        (
            na.ScalarArray(np.array(2 * ["Y", "Al"], dtype=object), axes="layer"),
            na.ScalarArray((2 * [10, 30]) * u.AA, axes="layer"),
            "layer",
        )
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength_ambient",
    argvalues=[
        200 * u.AA,
        na.linspace(100, 200, axis="wavelength", num=4) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="direction_ambient",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, 1),
        na.Cartesian3dVectorArray(
            x=np.sin(na.linspace(-1, 1, axis="angle", num=5)),
            y=0,
            z=np.cos(na.linspace(-1, 1, axis="angle", num=5)),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="n_ambient",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="n_substrate",
    argvalues=[
        1.5,
    ],
)
@pytest.mark.parametrize(
    argnames="normal",
    argvalues=[
        None,
    ],
)
@pytest.mark.parametrize(
    argnames="profile_interface",
    argvalues=[
        None,
        optika.materials.profiles.ErfInterfaceProfile(5 * u.AA),
    ],
)
def test_multilayer_efficiency(
    material_layers: na.AbstractScalarArray,
    thickness_layers: na.AbstractScalarArray,
    axis_layers: str,
    wavelength_ambient: u.Quantity | na.AbstractScalar,
    direction_ambient: na.AbstractCartesian3dVectorArray,
    n_ambient: complex | na.AbstractScalar,
    n_substrate: complex | na.AbstractScalar,
    normal: None | na.AbstractCartesian3dVectorArray,
    profile_interface: None | optika.materials.profiles.AbstractInterfaceProfile,
):
    reflected, transmitted = optika.materials.multilayer_efficiency(
        material_layers=material_layers,
        thickness_layers=thickness_layers,
        axis_layers=axis_layers,
        wavelength_ambient=wavelength_ambient,
        direction_ambient=direction_ambient,
        n_ambient=n_ambient,
        n_substrate=n_substrate,
        normal=normal,
        profile_interface=profile_interface,
    )

    assert np.all(reflected >= 0)
    assert np.all(reflected <= 1)
    assert np.all(transmitted >= 0)
    assert np.all(transmitted <= 1)
    assert np.all(reflected + transmitted <= 1)
    assert np.all(np.imag(reflected) == 0)
    assert np.all(np.imag(transmitted) == 0)


class AbstractTestAbstractMultilayerMaterial(
    test_materials.AbstractTestAbstractMaterial,
):
    def test_material_layers(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.material_layers
        assert isinstance(result, (str, na.AbstractScalarArray))
        assert np.issubdtype(result.dtype, object)
        assert a.axis_layers in result.shape

    def test_thickness_layers(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.thickness_layers
        assert isinstance(result, (u.Quantity, na.AbstractScalar))
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert a.axis_layers in result.shape

    def test_axis_layers(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.axis_layers
        assert isinstance(result, str)


class AbstractTestAbstractMultilayerFilm(
    AbstractTestAbstractMultilayerMaterial,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.MultilayerFilm(
            material_layers=na.ScalarArray(
                ndarray=np.array((["Al2O3", "Al", "Al2O3"]), dtype=object),
                axes="layer",
            ),
            thickness_layers=na.ScalarArray(
                ndarray=[5, 100, 5] * u.nm,
                axes="layer",
            ),
            axis_layers="layer",
        ),
    ],
)
class TestMultilayerFilm(
    AbstractTestAbstractMultilayerFilm,
):
    pass
