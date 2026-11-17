import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


class AbstractTestAbstractRulingSpacing(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
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
        argnames="position",
        argvalues=[
            na.Cartesian3dVectorArray(
                x=na.linspace(-1, 1, axis="position_x", num=5) * u.um,
                y=na.linspace(-1, 1, axis="position_y", num=6) * u.um,
                z=0 * u.um,
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_count(
        self,
        a: optika.rulings.AbstractRulingSpacing,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.Cartesian3dVectorArray,
    ) -> None:
        """
        The count is zero at the origin, and any closed form agrees with the
        integral of the ruling density along the chord from the origin.
        """
        result = a.count(position, normal)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)

        origin = na.Cartesian3dVectorArray() * u.mm
        assert np.allclose(a.count(origin, normal), 0)

        result_integral = optika.rulings.AbstractRulingSpacing.count(
            a,
            position,
            normal,
        )
        assert np.allclose(result, result_integral)


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


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.ConstantRulingSpacing(
            constant=1 * u.um,
            normal=na.Cartesian3dVectorArray(1, 0, 0),
        ),
        optika.rulings.Polynomial1dRulingSpacing(
            coefficients={
                0: 1 * u.um,
                1: 1e-4,
                2: 1e-8 / u.um,
            },
            normal=na.Cartesian3dVectorArray(1, 0, 0),
        ),
        optika.rulings.HolographicRulingSpacing(
            x1=na.Cartesian3dVectorArray(0, 1, -20) * u.mm,
            x2=na.Cartesian3dVectorArray(1, 2, -30) * u.mm,
            wavelength=500 * u.nm,
        ),
    ],
)
def test_count_derivative(a: optika.rulings.AbstractRulingSpacing) -> None:
    """
    The derivative of the count along a flat surface is the ruling density.

    This only holds where the density is the gradient of some function, so
    it is not part of the abstract tests, whose polynomial spacing rotates
    the argument of the polynomial without rotating the rulings.
    """
    normal = na.Cartesian3dVectorArray(0, 0, -1)
    position = na.Cartesian3dVectorArray(
        x=na.linspace(-1, 1, axis="position_x", num=5) * u.mm,
        y=na.linspace(-1, 1, axis="position_y", num=6) * u.mm,
        z=0 * u.mm,
    )

    d = a(position, normal)
    density = d / np.square(d.length)

    step = 1 * u.nm
    for unit_vector in [
        na.Cartesian3dVectorArray(1, 0, 0),
        na.Cartesian3dVectorArray(0, 1, 0),
    ]:
        count_plus = a.count(position + step * unit_vector, normal)
        count_minus = a.count(position - step * unit_vector, normal)
        derivative = (count_plus - count_minus) / (2 * step)
        assert np.allclose(derivative, density @ unit_vector)
