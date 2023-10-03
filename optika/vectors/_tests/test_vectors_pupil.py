import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from named_arrays._vectors.cartesian.tests import test_vectors_cartesian
import optika

_num_y = test_vectors_cartesian._num_y


vectors_pupil = [
    optika.vectors.PupilVectorArray(
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-5 * u.mm,
            stop=5 * u.mm,
            axis="y",
            num=_num_y,
        ).explicit
    )
]


def _items() -> list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis="y")),
    ]


class AbstractTestAbstractPupilVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray
):
    def test_pupil(self, array: optika.vectors.AbstractPupilVectorArray):
        assert isinstance(array.pupil, na.AbstractCartesian2dVectorArray)

    @pytest.mark.xfail(raises=NotImplementedError)
    def test_matrix(self, array):
        return super().test_matrix(array=array)

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=_items(),
    )
    def test__getitem__(
        self,
        array: optika.vectors.AbstractPupilVectorArray,
        item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize("array_2", vectors_pupil)
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize("array_2", vectors_pupil)
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
    ):
        @pytest.mark.parametrize("array_2", vectors_pupil)
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
        @pytest.mark.skip
        class TestPltPlotLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass

        @pytest.mark.skip
        class TestJacobian(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestJacobian,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeRoot(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass


@pytest.mark.parametrize("array", vectors_pupil)
class TestPupilVectorArray(
    AbstractTestAbstractPupilVectorArray,
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
            optika.vectors.PupilVectorArray(
                pupil=0 * u.mm,
            ),
        ],
    )
    def test__setitem__(
        self,
        array: optika.vectors.AbstractPupilVectorArray,
        item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
        value: optika.vectors.AbstractPupilVectorArray,
    ):
        super().test__setitem__(array=array, item=item, value=value)
