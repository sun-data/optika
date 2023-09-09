import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from named_arrays._vectors.cartesian.tests import test_vectors_cartesian
import optika

_num_y = test_vectors_cartesian._num_y


rays = [
    optika.rays.RayVectorArray(
        wavelength=500 * u.nm,
        position=na.Cartesian3dVectorArray(0, 1, 2) * u.mm,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
        attenuation=0 / u.mm,
    ),
    optika.rays.RayVectorArray(
        wavelength=na.linspace(
            start=400 * u.nm,
            stop=500 * u.nm,
            axis="y",
            num=_num_y,
        ),
        position=na.Cartesian3dVectorLinearSpace(
            start=-10 * u.mm,
            stop=10 * u.mm,
            axis="y",
            num=_num_y,
        ).explicit,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(na.linspace(-60, 60, axis="y", num=_num_y) * u.deg),
            y=0,
            z=np.cos(na.linspace(-60, 60, axis="y", num=_num_y) * u.deg),
        ),
        attenuation=na.linspace(0, 1, axis="y", num=_num_y) / u.mm,
        index_refraction=na.linspace(1, 1.5, axis="y", num=_num_y).explicit,
    ),
    optika.rays.RayVectorArray(
        wavelength=na.NormalUncertainScalarArray(600 * u.nm, width=5 * u.nm).explicit,
        position=na.Cartesian3dVectorArray(
            x=na.NormalUncertainScalarArray(3 * u.mm, width=1 * u.mm),
            y=na.NormalUncertainScalarArray(4 * u.mm, width=2 * u.mm),
            z=na.NormalUncertainScalarArray(5 * u.mm, width=3 * u.mm),
        ).explicit,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
        attenuation=na.UniformUncertainScalarArray(0.5, width=0.25) / u.mm,
        index_refraction=na.UniformUncertainScalarArray(1.5, width=0.5).explicit,
    ),
]


def _items() -> list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis="y")),
    ]


class AbstractTestAbstractRayVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray
):
    def test_direction(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(
            na.as_named_array(array.direction),
            na.AbstractCartesian3dVectorArray,
        )
        assert np.all(array.direction.length == 1)

    def test_intensity(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(na.as_named_array(array.intensity), na.AbstractScalar)
        assert np.all(array.intensity >= 0)

    def test_attenuation(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(na.as_named_array(array.attenuation), na.AbstractScalar)
        assert np.all(array.attenuation >= 0)
        assert na.unit_normalized(array.attenuation).is_equivalent(1 / u.m)

    def test_index_refraction(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(na.as_named_array(array.index_refraction), na.AbstractScalar)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_matrix(self, array):
        return super().test_matrix(array=array)

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=_items(),
    )
    def test__getitem__(
        self,
        array: na.AbstractPositionalVectorArray,
        item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize("array_2", rays)
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize("array_2", rays)
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):
        @pytest.mark.parametrize("array_2", rays)
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions
        ):
            pass

        @pytest.mark.parametrize(
            argnames="where",
            argvalues=[
                np._NoValue,
            ],
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="q",
            argvalues=[
                25 * u.percent,
            ],
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestPercentileLikeFunctions,
        ):
            pass

    class TestNamedArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions,
    ):
        @pytest.mark.xfail(raises=TypeError)
        class TestPltPlotLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass


@pytest.mark.parametrize("array", rays)
class TestRayVectorArray(
    AbstractTestAbstractRayVectorArray,
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArray,
):
    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(y=slice(None)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            optika.rays.RayVectorArray(
                wavelength=700 * u.nm,
                position=5 * u.mm,
                direction=na.Cartesian3dVectorArray(0, 1, 0),
            ),
        ],
    )
    def test__setitem__(
        self,
        array: na.ScalarArray,
        item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
        value: float | na.ScalarArray,
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitRayVectorArray(
    AbstractTestAbstractRayVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass
