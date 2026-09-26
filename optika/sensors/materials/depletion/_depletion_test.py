import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


class AbstractTestAbstractDepletionModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test_thickness(
        self,
        a: optika.sensors.materials.depletion.AbstractDepletionModel,
    ):
        result = a.thickness
        assert np.all(result > 0 * u.um)

    def test_width_max(
        self,
        a: optika.sensors.materials.depletion.AbstractDepletionModel,
    ):
        result = a.width_max
        if result is not None:
            assert np.all(result >= 0 * u.um)

    def test_width_depleted(
        self,
        a: optika.sensors.materials.depletion.AbstractDepletionModel,
    ):
        result = a.width_depleted
        if result is not None:
            assert np.all(result >= 0 * u.um)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.depletion.e2v_ccd64_thick(),
        optika.sensors.materials.depletion.e2v_ccd64_thin(),
        optika.sensors.materials.depletion.e2v_ccd64_thick().replace(
            width_max=5 * u.um,
            width_depleted=0.8 * u.um,
        ),
    ],
)
class TestJanesickDepletionModel(
    AbstractTestAbstractDepletionModel,
):
    def test_fit_mcc_width_depleted(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        """
        A spread in the depletion region widens the charge cloud, so the fit
        makes the field-free region thinner to compensate.
        """
        kwargs = dict(
            thickness_substrate=a.thickness_substrate,
            chemical_substrate=a.chemical_substrate,
            width_pixel=a.width_pixel,
            mcc_measured=a.mcc_measured,
        )
        cls = optika.sensors.materials.depletion.JanesickDepletionModel
        result = cls.fit_mcc(**kwargs, width_depleted=1.5 * u.um)
        result_sharp = cls.fit_mcc(**kwargs)
        assert result.width_depleted == 1.5 * u.um
        assert np.all(result.thickness > result_sharp.thickness)

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

    def test_shape_excludes_mcc_measured(
        self,
        a: optika.sensors.materials.depletion.JanesickDepletionModel,
    ):
        """
        The measurement this model was fitted against does not shape it.

        It is sampled along an axis of its own, which need not agree with the
        axes of any other measurement carried alongside it, and which says
        nothing about the shape of the model.
        """
        assert set(na.shape(a.mcc_measured)).isdisjoint(a.shape)
