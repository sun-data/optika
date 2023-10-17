import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import test_mixins


_position = [
    na.Cartesian3dVectorArray() * u.mm,
    na.Cartesian3dVectorArray(
        x=na.linspace(-5, 5, axis="x", num=3) * u.mm,
        y=na.linspace(-5, 5, axis="y", num=4) * u.mm,
        z=0 * u.mm,
    ),
]


class AbstractTestAbstractRulings(
    test_mixins.AbstractTestTransformable,
):
    @pytest.mark.parametrize("position", _position)
    def test_spacing(
        self,
        a: optika.rulings.AbstractRulings,
        position: na.AbstractCartesian3dVectorArray,
    ):
        result = a.spacing(position)
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result > 0)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_diffraction_order(self, a: optika.rulings.AbstractRulings):
        assert na.get_dtype(a.diffraction_order) == int

    @pytest.mark.parametrize("position", _position)
    def test_normal(
        self,
        a: optika.rulings.AbstractRulings,
        position: na.AbstractCartesian3dVectorArray,
    ):
        result = a.normal(position)
        assert isinstance(result, na.AbstractCartesian3dVectorArray)
        assert np.all(result.length == 1)


class AbstractTestAbstractPolynomialDensityRulings(
    AbstractTestAbstractRulings,
):
    @pytest.mark.parametrize("position", _position)
    def test_frequency(
        self,
        a: optika.rulings.AbstractPolynomialDensityRulings,
        position: na.AbstractCartesian3dVectorArray,
    ):
        result = a.frequency(position)
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result > 0)
        assert na.unit_normalized(result).is_equivalent(1 / u.mm)

    def test_ruling_density(
        self,
        a: optika.rulings.AbstractConstantDensityRulings,
    ):
        assert a.ruling_density.unit.is_equivalent(1 / u.mm)


class AbstractTestAbstractConstantDensityRulings(
    AbstractTestAbstractPolynomialDensityRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.ConstantDensityRulings(
            ruling_density=5000 / u.mm,
            diffraction_order=1,
        ),
        optika.rulings.ConstantDensityRulings(
            ruling_density=na.linspace(1, 5, axis="rulings", num=4) / u.mm,
            diffraction_order=na.ScalarArray(np.array([-1, 0, 1]), axes="m"),
        ),
    ],
)
class TestConstantDensityRulings(
    AbstractTestAbstractConstantDensityRulings,
):
    pass
