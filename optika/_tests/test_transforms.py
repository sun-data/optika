import pytest
import numpy as np
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika


class AbstractTestAbstractTransform:
    def test_matrix(self, transform: optika.transforms.AbstractTransform):
        if transform.matrix is not None:
            assert isinstance(
                transform.matrix,
                na.AbstractCartesian3dMatrixArray,
            )
            assert isinstance(
                transform.matrix.to(u.dimensionless_unscaled),
                na.AbstractCartesian3dMatrixArray,
            )

    def test_vector(self, transform: optika.transforms.AbstractTransform):
        if transform.vector is not None:
            assert isinstance(
                transform.vector,
                na.AbstractCartesian3dVectorArray,
            )
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
        argnames="transform_2",
        argvalues=[
            optika.transforms.Translation(na.Cartesian3dVectorArray(5) * u.mm),
            optika.transforms.RotationX(53 * u.deg),
            optika.transforms.TransformList(
                [
                    optika.transforms.Translation(na.Cartesian3dVectorArray(5) * u.mm),
                    optika.transforms.RotationX(53 * u.deg),
                ]
            ),
        ],
    )
    class TestMatmul:
        def test__matmul__(
            self,
            transform: optika.transforms.AbstractTransform,
            transform_2: optika.transforms.AbstractTransform,
        ):
            result = transform @ transform_2
            assert isinstance(result, optika.transforms.TransformList)
            for t in result:
                assert isinstance(t, optika.transforms.AbstractTransform)

            x = na.Cartesian3dVectorUniformRandomSample(
                start=-5 * u.mm,
                stop=5 * u.mm,
                shape_random=dict(x=5, y=6),
            )
            assert np.allclose(result(x), transform(transform_2(x)))

        def test__matmul__reversed(
            self,
            transform: optika.transforms.AbstractTransform,
            transform_2: optika.transforms.AbstractTransform,
        ):
            return self.test__matmul__(
                transform=transform_2,
                transform_2=transform,
            )


displacement_parameterization = [
    na.Cartesian3dVectorArray() * u.mm,
    na.Cartesian3dVectorArray(1, -2, 3) * u.mm,
    na.Cartesian3dVectorArray(
        x=1 * u.mm,
        y=na.linspace(-5, 5, axis="translation", num=4) * u.mm,
        z=0 * u.mm,
    ),
    na.Cartesian3dVectorArray(
        x=0 * u.mm,
        y=na.linspace(-5, 5, axis="translation", num=4) * u.mm,
        z=na.NormalUncertainScalarArray(6 * u.mm, width=0.1 * u.mm),
    ),
]


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.Translation(displacement)
        for displacement in displacement_parameterization
    ],
)
class TestTranslation(
    AbstractTestAbstractTransform,
):
    pass


angle_parameterization = [
    0 * u.deg,
    45 * u.deg,
    na.linspace(0 * u.deg, 360 * u.deg, axis="angle", num=3),
    na.NormalUncertainScalarArray(45 * u.deg, width=5 * u.deg),
]


class AbstractTestAbstractRotation(
    AbstractTestAbstractTransform,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[optika.transforms.RotationX(angle) for angle in angle_parameterization],
)
class TestRotationX(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[optika.transforms.RotationY(angle) for angle in angle_parameterization],
)
class TestRotationY(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[optika.transforms.RotationZ(angle) for angle in angle_parameterization],
)
class TestRotationZ(
    AbstractTestAbstractRotation,
):
    pass


@pytest.mark.parametrize(
    argnames="transform",
    argvalues=[
        optika.transforms.TransformList(
            [
                optika.transforms.Translation(displacement),
                optika.transforms.RotationZ(angle),
                optika.transforms.Translation(-displacement),
                optika.transforms.RotationY(-angle),
            ]
        )
        for displacement in displacement_parameterization
        for angle in angle_parameterization
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
        assert np.allclose(b, c)


transform_parameterization = [
    None,
    optika.transforms.TransformList(
        [
            optika.transforms.Translation(na.Cartesian3dVectorArray(5) * u.mm),
            optika.transforms.RotationZ(53 * u.deg),
            optika.transforms.Translation(na.Cartesian3dVectorArray(5) * u.mm),
        ]
    ),
    optika.transforms.RotationZ(
        na.ScalarLinearSpace(0 * u.deg, 90 * u.deg, axis="transform", num=3)
    ),
    optika.transforms.RotationZ(
        na.NormalUncertainScalarArray(53 * u.deg, width=5 * u.deg)
    ),
]


class AbstractTestTransformable:
    def test_transform(self, a: optika.transforms.Transformable):
        if a.transform is not None:
            assert isinstance(a.transform, optika.transforms.AbstractTransform)
