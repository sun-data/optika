import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


class AbstractTestAbstractRulingSpacing(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
):
    @pytest.mark.parametrize(
        argnames="position",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
            na.Cartesian3dVectorArray(
                x=na.linspace(-1, 1, axis="position_x", num=5) * u.mm,
                y=na.linspace(-1, 1, axis="position_y", num=6) * u.mm,
                z=0 * u.mm,
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test__call__(
        self,
        a: optika.rulings.AbstractRulingSpacing,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.Cartesian3dVectorArray,
    ):
        result = a(position, normal)
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        assert np.all(result.length > (0 * u.mm))


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.ConstantRulingSpacing(
            constant=1 * u.um,
            normal=na.Cartesian3dVectorArray(1, 0, 0),
        ),
    ],
)
class TestConstantSpacing(
    AbstractTestAbstractRulingSpacing,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.Polynomial1dRulingSpacing(
            coefficients={
                0: 1 * u.um,
                1: 2,
                2: 3 / u.um,
            },
            normal=na.Cartesian3dVectorArray(1, 0, 0),
            transformation=na.transformations.Cartesian3dRotationZ(30 * u.deg),
        ),
    ],
)
class TestPolynomial1dRulingSpacing(
    AbstractTestAbstractRulingSpacing,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.HolographicRulingSpacing(
            x1=na.Cartesian3dVectorArray(0, 1, 2) * u.mm,
            x2=na.Cartesian3dVectorArray(1, 2, 3) * u.mm,
            wavelength=500 * u.nm,
        ),
    ],
)
class TestHolographicRulingSpacing(
    AbstractTestAbstractRulingSpacing,
):
    pass
