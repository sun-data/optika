import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

_wavelength = na.linspace(100, 200, axis="wavelength", num=4) * u.AA


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize("direction_left", [1])
@pytest.mark.parametrize("direction_right", [1])
@pytest.mark.parametrize("polarized_s", [False, True])
@pytest.mark.parametrize("n_left", [1])
@pytest.mark.parametrize("n_right", [optika.chemicals.Chemical("Si").n(_wavelength)])
@pytest.mark.parametrize(
    argnames="interface",
    argvalues=[optika.materials.profiles.ErfInterfaceProfile(5 * u.nm)],
)
def test_refraction(
    wavelength: u.Quantity | na.AbstractScalar,
    direction_left: float | na.AbstractScalar,
    direction_right: float | na.AbstractScalar,
    polarized_s: bool | na.AbstractScalar,
    n_left: float | na.AbstractScalar,
    n_right: float | na.AbstractScalar,
    interface: None | optika.materials.profiles.AbstractInterfaceProfile,
):
    result = optika.materials.matrices.refraction(
        wavelength=wavelength,
        direction_left=direction_left,
        direction_right=direction_right,
        polarized_s=polarized_s,
        n_left=n_left,
        n_right=n_right,
        interface=interface,
    )
    assert isinstance(result, na.AbstractCartesian2dMatrixArray)
    assert np.all(result.determinant != 0)


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize("direction", [1])
@pytest.mark.parametrize("thickness", [10 * u.nm])
@pytest.mark.parametrize("n", [optika.chemicals.Chemical("Si").n(_wavelength)])
def test_propagation(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    thickness: u.Quantity | na.AbstractScalar,
    n: u.Quantity | na.AbstractScalar,
):
    result = optika.materials.matrices.propagation(
        wavelength=wavelength,
        direction=direction,
        thickness=thickness,
        n=n,
    )
    assert isinstance(result, na.AbstractCartesian2dMatrixArray)
    assert np.all(result.determinant != 0)


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize("direction", [na.Cartesian3dVectorArray(0, 0, 1)])
@pytest.mark.parametrize("polarized_s", [False, True])
@pytest.mark.parametrize("thickness", [10 * u.nm])
@pytest.mark.parametrize("n", [optika.chemicals.Chemical("Si").n(_wavelength)])
@pytest.mark.parametrize("normal", [na.Cartesian3dVectorArray(0, 0, -1)])
def test_transfer(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    polarized_s: bool | na.AbstractScalar,
    thickness: u.Quantity | na.AbstractScalar,
    n: float | na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
):
    result = optika.materials.matrices.transfer(
        wavelength=wavelength,
        direction=direction,
        polarized_s=polarized_s,
        thickness=thickness,
        n=n,
        normal=normal,
    )
    assert isinstance(result, na.AbstractCartesian2dMatrixArray)
    assert np.all(result.determinant != 0)
