import pytest
import abc
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika


class AbstractTestAbstractSag(
    abc.ABC,
):
    @pytest.mark.parametrize(
        argnames="position",
        argvalues=[
            na.Cartesian2dVectorArray(0, 0) * u.mm,
            na.Cartesian2dVectorLinearSpace(0, 1, axis="s", num=5) * u.mm,
            na.Cartesian2dVectorArray(
                x=na.ScalarLinearSpace(0, 1, axis="x", num=5) * u.mm,
                y=na.NormalUncertainScalarArray(
                    nominal=na.ScalarLinearSpace(-1, 0, axis="y", num=6) * u.mm,
                    width=0.1 * u.mm,
                ),
            ),
        ],
    )
    class TestFunctionsOfPosition:
        def test__call__(
            self,
            sag: optika.sags.AbstractSag,
            position: na.AbstractCartesian2dVectorArray,
        ):
            result = sag(position)
            assert isinstance(na.as_named_array(result), na.AbstractScalar)
            assert set(na.shape(position)).issubset(na.shape(result))

        def test_normal(
            self,
            sag: optika.sags.AbstractSag,
            position: na.AbstractCartesian2dVectorArray,
        ):
            result = sag.normal(position)
            assert isinstance(result, na.AbstractCartesian3dVectorArray)
            assert set(na.shape(position)).issubset(na.shape(result))


class TestSphericalSag(
    AbstractTestAbstractSag,
):
    def test_curvature(self, sag: optika.sags.SphericalSag):
        assert isinstance(na.as_named_array(sag.curvature), na.AbstractScalar)
        assert na.shape(sag.curvature) == na.shape(sag.radius)


class TestConicSag(
    TestSphericalSag,
):
    pass


class TestToroidalSag(
    TestSphericalSag,
):
    pass


def _radii() -> list[u.Quantity | na.AbstractScalar]:
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
    argvalues=[optika.sags.SphericalSag(radius=radius) for radius in _radii()],
)
class TestSphericalSag(TestSphericalSag):  # type: ignore[no-redef]
    pass


def _conics() -> list[u.Quantity | na.AbstractScalar]:
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
        optika.sags.ConicSag(radius=radius, conic=conic)
        for radius in _radii()
        for conic in _conics()
    ],
)
class TestConicSag(TestConicSag):  # type: ignore[no-redef]
    pass


@pytest.mark.parametrize(
    argnames="sag",
    argvalues=[
        optika.sags.ToroidalSag(
            radius=radius,
            radius_of_rotation=2 * radius_of_rotation,
        )
        for radius in _radii()
        for radius_of_rotation in _radii()
    ],
)
class TestToroidalSag(TestToroidalSag):  # type: ignore[no-redef]
    pass
