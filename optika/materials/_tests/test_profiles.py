import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


class AbstractTestAbstractInterfaceProfile(
    abc.ABC,
):

    def test_width(self, a: optika.materials.profiles.AbstractInterfaceProfile):
        result = a.width
        assert na.unit_normalized(result).is_equivalent(u.mm)

    @pytest.mark.parametrize(
        argnames="z",
        argvalues=[
            0 * u.nm,
            na.linspace(-1, 1, axis="z", num=11) * u.nm,
            na.NormalUncertainScalarArray(0 * u.nm, width=1 * u.nm, num_distribution=11)
        ],
    )
    def test__call__(
        self,
        a: optika.materials.profiles.AbstractInterfaceProfile,
        z: u.Quantity | na.AbstractScalar
    ):
        result = a(z)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.profiles.ErfInterfaceProfile(width=5 * u.AA)
    ]
)
class TestErfInterfaceProfile(
    AbstractTestAbstractInterfaceProfile,
):
    pass
