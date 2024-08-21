import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize(
    argnames="cos_incidence",
    argvalues=[
        1,
        na.linspace(-1, 1, axis="angle", num=7),
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
        1.5 + 0.2j,
        na.linspace(1, 2, axis="index_refraction_new", num=4),
    ],
)
def test_snells_law_scalar(
    cos_incidence: float | na.AbstractScalar,
    index_refraction: float | na.AbstractScalar,
    index_refraction_new: float | na.AbstractScalar,
):
    result = optika.materials.snells_law_scalar(
        cos_incidence=cos_incidence,
        index_refraction=index_refraction,
        index_refraction_new=index_refraction_new,
    )

    n1 = index_refraction
    n2 = index_refraction_new
    result_expected = np.cos(
        np.emath.arcsin(n1 * np.sin(np.emath.arccos(cos_incidence)) / n2)
    )

    assert np.allclose(result, result_expected)


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
        (
            0,
            None,
            None,
        ),
        (
            1,
            5 * u.um,
            na.Cartesian3dVectorArray(1, 0, 0),
        ),
        (
            na.linspace(-2, 2, "m", 5),
            5 * u.um,
            na.Cartesian3dVectorArray(1, 0, 0),
        ),
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
