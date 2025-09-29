import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


class AbstractTestAbstractDepletionModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestShaped,
):
    def test_thickness(
        self,
        a: optika.sensors.materials.depletion.AbstractDepletionModel,
    ):
        result = a.thickness
        assert np.all(result > 0 * u.um)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.depletion.e2v_ccd64_thick(),
        optika.sensors.materials.depletion.e2v_ccd64_thin(),
    ]
)
class TestJanesickDepletionModel(
    AbstractTestAbstractDepletionModel,
):

    def test_chemical_substrate(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        result = a.chemical_substrate
        assert isinstance(result, optika.chemicals.AbstractChemical)

    def test_thickness_substrate(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        result = a.thickness_substrate
        assert np.all(result > 0 * u.um)

    def test_width_pixel(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        result = a.width_pixel
        assert np.all(result > 0 * u.um)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
            na.geomspace(1, 10000, "w", num=11) * u.AA,
        ],
    )
    def test_mean_charge_capture(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.mean_charge_capture(wavelength)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_mcc_measured(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        result = a.mcc_measured
        assert isinstance(result, na.AbstractFunctionArray)
        assert np.all(result.inputs > 0 * u.AA)
        assert np.all(result.outputs > 0)
