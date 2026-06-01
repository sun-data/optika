import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


def _wavelength() -> na.AbstractScalar:
    return na.linspace(100, 1000, axis="wavelength", num=10) * u.AA


def _area() -> na.AbstractScalar:
    return na.linspace(1, 5, axis="wavelength", num=10) * u.cm**2


class AbstractTestAbstractEffectiveAreaModel(
    test_mixins.AbstractTestPrintable,
):
    def test__call__(self, a: optika.radiometry.AbstractEffectiveAreaModel):
        wavelength = na.linspace(200, 900, axis="wavelength", num=5) * u.AA
        result = a(wavelength)
        assert isinstance(result, na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.cm**2)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=_wavelength(),
            area=_area(),
            axis_wavelength="wavelength",
        ),
    ],
)
class TestInterpolatedEffectiveAreaModel(
    AbstractTestAbstractEffectiveAreaModel,
):
    def test_wavelength(self, a: optika.radiometry.InterpolatedEffectiveAreaModel):
        assert isinstance(a.wavelength, na.AbstractScalar)

    def test_area(self, a: optika.radiometry.InterpolatedEffectiveAreaModel):
        assert isinstance(a.area, na.AbstractScalar)

    def test_axis_wavelength(
        self,
        a: optika.radiometry.InterpolatedEffectiveAreaModel,
    ):
        assert isinstance(a.axis_wavelength, str)

    def test_interpolates_calibration(
        self,
        a: optika.radiometry.InterpolatedEffectiveAreaModel,
    ):
        # evaluating at the calibration wavelengths returns the calibration areas
        result = a(a.wavelength)
        assert np.all(result == a.area)
