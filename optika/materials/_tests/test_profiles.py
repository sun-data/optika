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
            na.NormalUncertainScalarArray(0, width=1, num_distribution=11) * u.nm,
        ],
    )
    def test__call__(
        self,
        a: optika.materials.profiles.AbstractInterfaceProfile,
        z: u.Quantity | na.AbstractScalar,
    ):
        result = a(z)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            304 * u.AA,
            na.linspace(300, 400, axis="wavelength", num=5) * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, 1),
            na.Cartesian3dYRotationMatrixArray(
                na.linspace(-90, 90, axis="angle", num=5) * u.deg,
            )
            @ na.Cartesian3dVectorArray(0, 0, 1),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, 1),
        ],
    )
    def test_reflectivity(
        self,
        a: optika.materials.profiles.AbstractInterfaceProfile,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.reflectivity(wavelength, direction, normal)
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.profiles.ErfInterfaceProfile(width=5 * u.AA),
    ],
)
class TestErfInterfaceProfile(
    AbstractTestAbstractInterfaceProfile,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.profiles.ExponentialInterfaceProfile(
            width=5 * u.AA,
        ),
        optika.materials.profiles.ExponentialInterfaceProfile(
            width=na.linspace(5, 10, axis="width", num=6) * u.AA,
        ),
        optika.materials.profiles.ExponentialInterfaceProfile(
            width=na.NormalUncertainScalarArray(5, 1, num_distribution=11) * u.AA,
        ),
    ],
)
class TestExponentialInterfaceProfile(
    AbstractTestAbstractInterfaceProfile,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.profiles.LinearInterfaceProfile(
            width=5 * u.AA,
        ),
        optika.materials.profiles.LinearInterfaceProfile(
            width=na.linspace(5, 10, axis="width", num=6) * u.AA,
        ),
        optika.materials.profiles.LinearInterfaceProfile(
            width=na.NormalUncertainScalarArray(5, 1, num_distribution=11) * u.AA,
        ),
    ],
)
class TestLinearInterfaceProfile(
    AbstractTestAbstractInterfaceProfile,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.profiles.SinusoidalInterfaceProfile(
            width=5 * u.AA,
        ),
        optika.materials.profiles.SinusoidalInterfaceProfile(
            width=na.linspace(5, 10, axis="width", num=6) * u.AA,
        ),
        optika.materials.profiles.SinusoidalInterfaceProfile(
            width=na.NormalUncertainScalarArray(5, 1, num_distribution=11) * u.AA,
        ),
    ],
)
class TestSinusoidalInterfaceProfile(
    AbstractTestAbstractInterfaceProfile,
):
    pass
