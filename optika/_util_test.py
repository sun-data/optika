import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize(
    argnames="angles",
    argvalues=[
        na.Cartesian2dVectorArray(1, 2) * u.deg,
        na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=1,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=5,
        ),
    ],
)
def test_direction(angles: na.AbstractCartesian2dVectorArray):
    result = optika.direction(angles)

    rotation_x = na.Cartesian3dYRotationMatrixArray(-angles.x)
    rotation_y = na.Cartesian3dXRotationMatrixArray(+angles.y)
    result_expected = rotation_x @ rotation_y @ na.Cartesian3dVectorArray(z=1)
    # result_expected = rotation_y @ rotation_x @ na.Cartesian3dVectorArray(z=1)

    print(f"{result=}")
    print(f"{result_expected=}")

    assert isinstance(result, na.AbstractCartesian3dVectorArray)
    assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        na.Cartesian3dVectorArray(1, 2, 5).normalized,
    ],
)
def test_angles(direction: na.AbstractCartesian3dVectorArray):
    result = optika.angles(direction)

    print(f"{direction=}")
    print(f"{optika.direction(result)=}")

    assert isinstance(result, na.AbstractCartesian2dVectorArray)
    assert np.allclose(direction, optika.direction(result))
