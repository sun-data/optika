import pytest
import abc
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika
from . import test_transforms


class AbstractTestAbstractSag(
    abc.ABC,
):
    @pytest.mark.parametrize(
        argnames="position",
        argvalues=[
            na.Cartesian3dVectorArray() * u.mm,
            na.Cartesian3dVectorLinearSpace(0, 1, axis="s", num=5) * u.mm,
            na.Cartesian3dVectorArray(
                x=na.ScalarLinearSpace(0, 1, axis="x", num=5) * u.mm,
                y=na.NormalUncertainScalarArray(
                    nominal=na.ScalarLinearSpace(-1, 0, axis="y", num=6) * u.mm,
                    width=0.1 * u.mm,
                ),
                z=0*u.mm,
            ),
        ],
    )
    class TestFunctionsOfPosition:
        def test__call__(
            self,
            sag: optika.sags.AbstractSag,
            position: na.AbstractCartesian3dVectorArray,
        ):
            result = sag(position)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert set(na.shape(position)).issubset(na.shape(result))

        def test_normal(
            self,
            sag: optika.sags.AbstractSag,
            position: na.AbstractCartesian3dVectorArray,
        ):
            result = sag.normal(position)
            assert isinstance(result, na.AbstractCartesian3dVectorArray)
            assert set(na.shape(position)).issubset(na.shape(result))


class AbstractTestAbstractSphericalSag(
    AbstractTestAbstractSag,
):
    def test_curvature(self, sag: optika.sags.SphericalSag):
        assert isinstance(na.as_named_array(sag.curvature), na.AbstractScalar)
        assert na.shape(sag.curvature) == na.shape(sag.radius)


def radius_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        100 * u.mm,
        na.ScalarLinearSpace(100, 1000, axis="radius", num=5) * u.mm,
    ]
    widths = [
        None,
        10 * u.mm,
    ]
    return [
        nominal if width is None else na.NormalUncertainScalarArray(nominal, width)
        for nominal in nominals
        for width in widths
    ]


@pytest.mark.parametrize(
    argnames="sag",
    argvalues=[
        optika.sags.SphericalSag(
            radius=radius,
            transform=transform,
        )
        for radius in radius_parameterization()
        for transform in test_transforms.transform_parameterization
    ],
)
class TestSphericalSag(
    AbstractTestAbstractSphericalSag,
):
    pass


def conic_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        0 * u.dimensionless_unscaled,
        na.ScalarLinearSpace(0, 1, axis="conic", num=4),
    ]
    widths = [
        None,
        0.1 * u.dimensionless_unscaled,
    ]
    return [
        nominal if width is None else na.NormalUncertainScalarArray(nominal, width)
        for nominal in nominals
        for width in widths
    ]


@pytest.mark.parametrize(
    argnames="sag",
    argvalues=[
        optika.sags.ConicSag(
            radius=radius,
            conic=conic,
            transform=transform,
        )
        for radius in radius_parameterization()
        for conic in conic_parameterization()
        for transform in test_transforms.transform_parameterization
    ],
)
class TestConicSag(
    AbstractTestAbstractSphericalSag,
):
    pass


@pytest.mark.parametrize(
    argnames="sag",
    argvalues=[
        optika.sags.ToroidalSag(
            radius=radius,
            radius_of_rotation=2 * radius_of_rotation,
            transform=transform
        )
        for radius in radius_parameterization()
        for radius_of_rotation in radius_parameterization()
        for transform in test_transforms.transform_parameterization
    ],
)
class TestToroidalSag(
    AbstractTestAbstractSphericalSag,
):
    pass
