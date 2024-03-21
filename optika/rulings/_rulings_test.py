import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins
from optika.rays._tests import test_ray_vectors

_wavelength = na.linspace(100, 300, axis="wavelength", num=11) * u.AA


class AbstractTestAbstractRulings(
    test_mixins.AbstractTestPrintable,
):
    def test_diffraction_order(self, a: optika.rulings.AbstractRulings):
        assert na.get_dtype(a.diffraction_order) == int

    def test_spacing(self, a: optika.rulings.AbstractRulings):
        result = a.spacing
        types = (u.Quantity, na.AbstractScalar, optika.rulings.AbstractRulingSpacing)
        assert isinstance(result, types)

    def test_spacing_(self, a: optika.rulings.AbstractRulings):
        result = a.spacing_
        assert isinstance(result, optika.rulings.AbstractRulingSpacing)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=test_ray_vectors.rays
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ]
    )
    def test_efficiency(
        self,
        a: optika.rulings.AbstractRulings,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.efficiency(
            rays=rays,
            normal=normal,
        )

        assert np.all(result > 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.Rulings(
            spacing=1 * u.um,
            diffraction_order=1,
        ),
        optika.rulings.Rulings(
            spacing=1 * u.um,
            diffraction_order=na.ScalarArray(np.array([-1, 0, 1]), axes="m"),
        ),
    ],
)
class TestRulings(
    AbstractTestAbstractRulings
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.MeasuredRulings(
            spacing=1 * u.um,
            diffraction_order=1,
            efficiency_measured=na.FunctionArray(
                inputs=na.SpectralDirectionalVectorArray(
                    wavelength=_wavelength,
                    direction=na.Cartesian3dVectorArray(0, 0, 1),
                ),
                outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2),
            ),
        )
    ],
)
class TestMeasuredRulings(
    AbstractTestAbstractRulings
):
    pass
