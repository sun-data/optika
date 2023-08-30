import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from named_arrays._vectors.cartesian.tests import test_vectors_cartesian
import optika

_num_y = test_vectors_cartesian._num_y


def _arrays() -> list[optika.rays.RayVectorArray]:
    return [
        optika.rays.RayVectorArray(
            wavelength=500 * u.nm,
            position=na.Cartesian3dVectorArray(0, 1, 2) * u.mm,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        ),
        optika.rays.RayVectorArray(
            wavelength=na.linspace(400, 500, axis="y", num=_num_y) * u.mm,
            position=na.Cartesian3dVectorLinearSpace(-10, 10, axis="y", num=_num_y) * u.mm,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )
    ]


def _arrays_2() -> list[optika.rays.RayVectorArray]:
    return [
        optika.rays.RayVectorArray(
            wavelength=600 * u.nm,
            position=na.Cartesian3dVectorArray(3, 4, 5) * u.mm,
            direction=na.Cartesian3dVectorArray(1, 0, 0),
        ),
        optika.rays.RayVectorArray(
            wavelength=na.NormalUncertainScalarArray(600 * u.nm, width=5 * u.nm),
            position=na.Cartesian3dVectorArray(
                x=na.NormalUncertainScalarArray(3 * u.mm, width=1 * u.mm),
                y=na.NormalUncertainScalarArray(4 * u.mm, width=2 * u.mm),
                z=na.NormalUncertainScalarArray(5 * u.mm, width=3 * u.mm),
            ),
            direction=na.Cartesian3dVectorArray(1, 0, 0),
        )
    ]


def _items() -> list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractRayVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray
):
    def test_direction(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(na.as_named_array(array.direction), na.AbstractCartesian3dVectorArray)
        assert np.all(array.direction.length == 1)

    def test_intensity(self, array: optika.rays.AbstractRayVectorArray) -> na.AbstractScalar:
        assert isinstance(na.as_named_array(array.intensity), na.AbstractScalar)
        assert np.all(array.intensity >= 0)

    def test_index_refraction(self, array: optika.rays.AbstractRayVectorArray):
        assert isinstance(na.as_named_array(array.index_refraction), na.AbstractScalar)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_matrix(self, array):
        return super().test_matrix(array=array)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_items(),
    )
    def test__getitem__(
            self,
            array: na.AbstractPositionalVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
            ]
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                25 * u.percent,
            ]
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
            .TestPercentileLikeFunctions,
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


@pytest.mark.parametrize("array", _arrays(), scope="module")
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
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitRayVectorArray(
    AbstractTestAbstractRayVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass
