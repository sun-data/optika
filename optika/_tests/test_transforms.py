import pytest
import numpy as np
import astropy.units as u   # type: ignore[import]
import named_arrays as na   # type: ignore[import]
import optika


class AbstractTestAbstractTransform:
    def test_matrix(self, transform: optika.transforms.AbstractTransform):
        assert isinstance(
            transform.matrix,
            na.AbstractCartesian3dMatrixArray,
        )
        assert isinstance(
            transform.matrix.to(u.dimensionless_unscaled),
            na.AbstractCartesian3dMatrixArray,
        )

    def test_vector(self, transform: optika.transforms.AbstractTransform):
        assert isinstance(
            transform.vector,
            na.AbstractCartesian3dVectorArray)
        assert isinstance(
            transform.vector.to(u.mm),
            na.AbstractCartesian3dVectorArray,
        )

    def test__call__(self, transform: optika.transforms.AbstractTransform):
        x = na.Cartesian3dVectorArray(x=1, y=-2, z=3) * u.m
        y = transform(x)
        assert isinstance(y, na.AbstractCartesian3dVectorArray)

    def test_inverse(self, transform: optika.transforms.AbstractTransform):
        x = na.Cartesian3dVectorArray(x=1, y=-2, z=3) * u.m
        y = transform(x)
        z = transform.inverse(y)
        assert np.allclose(x, z)


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.Translation(na.Cartesian3dVectorArray() * u.mm),
        optika.transforms.Translation(na.Cartesian3dVectorArray(1, -2, 3) * u.mm)
    ],
)
class TestTranslation(
    AbstractTestAbstractTransform,
):
    pass


class AbstractTestAbstractRotation(
    AbstractTestAbstractTransform,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.RotationX(0 * u.deg),
        optika.transforms.RotationX(45 * u.deg),
        optika.transforms.RotationX(90 * u.deg),
        optika.transforms.RotationX(223 * u.deg),
    ],
)
class TestRotationX(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.RotationY(0 * u.deg),
        optika.transforms.RotationY(45 * u.deg),
        optika.transforms.RotationY(90 * u.deg),
        optika.transforms.RotationY(223 * u.deg),
    ],
)
class TestRotationY(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.RotationZ(0 * u.deg),
        optika.transforms.RotationZ(45 * u.deg),
        optika.transforms.RotationZ(90 * u.deg),
        optika.transforms.RotationZ(223 * u.deg),
    ],
)
class TestRotationZ(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames='transform',
    argvalues=[
        optika.transforms.TransformList([
            optika.transforms.Translation(na.Cartesian3dVectorArray(x=2) * u.m),
            optika.transforms.RotationZ(90 * u.deg),
            optika.transforms.Translation(na.Cartesian3dVectorArray(x=2) * u.m),
            optika.transforms.RotationY(90 * u.deg),
            optika.transforms.Translation(na.Cartesian3dVectorArray(x=2) * u.m),
        ])
    ],
)
class TestTransformList(
    AbstractTestAbstractTransform,
):
    def test__call__(self, transform: optika.transforms.TransformList):  # type: ignore[override]
        super().test__call__(transform=transform)
        x = na.Cartesian3dVectorArray() * u.m
        b = transform(x)
        c = x
        for t in transform.transforms:
            c = t(c)
        assert b == c
