import warnings
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins

_thickness_substrate = 14 * u.um


class AbstractTestAbstractDiffusionModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test_thickness_depletion(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        result = a.thickness_depletion
        assert np.all(result >= 0 * u.um)

    @pytest.mark.parametrize(
        argnames="depth",
        argvalues=[
            0 * u.um,
            na.linspace(0, 14, axis="depth", num=15) * u.um,
        ],
    )
    def test_width(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        depth: u.Quantity | na.AbstractScalar,
    ):
        s = _thickness_substrate
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = a.width(depth, s)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0 * u.um)

        # the charge created at the back surface spreads the most,
        # and the charge created at the gates not at all, if it has a
        # depletion region to be created in
        back = a.width(0 * u.um, s)
        assert np.all(result <= back * (1 + 1e-12))
        if a.thickness_depletion > 0 * u.um:
            assert np.allclose(a.width(s, s), 0 * u.um)

    @pytest.mark.parametrize(
        argnames="absorption",
        argvalues=[
            0.01 / u.um,
            0.3 / u.um,
            10 / u.um,
        ],
    )
    def test_width_average(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        absorption: u.Quantity | na.AbstractScalar,
    ):
        """
        The average width is the square root of the variance of the width
        averaged over the depth at which the photons are absorbed.
        """
        s = _thickness_substrate
        result = a.width_average(absorption, s)

        axis = "depth"
        num = 100000
        depth = (na.arange(0, num, axis=axis) + 0.5) * s / num
        weight = np.exp(-absorption * depth)
        variance = (np.square(a.width(depth, s)) * weight).sum(axis) / weight.sum(axis)

        assert np.allclose(result, np.sqrt(variance), rtol=1e-5, atol=1e-9 * u.um)

    @pytest.mark.parametrize(
        argnames="width_pixel",
        argvalues=[
            15 * u.um,
            na.Cartesian2dVectorArray(10, 20) * u.um,
            0 * u.um,
        ],
    )
    def test_probability_same_pixel(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    ):
        depth = na.linspace(0, 14, axis="depth", num=15) * u.um
        result = a.probability_same_pixel(depth, _thickness_substrate, width_pixel)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        if not isinstance(width_pixel, na.AbstractCartesian2dVectorArray):
            if width_pixel == 0 * u.um:
                assert np.all(result == 1)

    def test_mean_charge_capture(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        absorption = na.geomspace(1e-3, 1e3, axis="absorption", num=7) / u.um
        result = a.mean_charge_capture(absorption, _thickness_substrate, 15 * u.um)
        assert np.all(result > 0)
        assert np.all(result <= 1)

    def test_kernel(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        result = a.kernel(1 / u.um, _thickness_substrate, 15 * u.um, "x", "y")
        assert isinstance(result, na.FunctionArray)
        assert np.allclose(result.outputs.sum(("x", "y")), 1)

    def test_parameters_monte_carlo(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        result = a._parameters_monte_carlo(_thickness_substrate)
        assert isinstance(result, dict)
        for value in result.values():
            assert np.all(value >= 0 * u.um)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
            width_max=4 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
            width_depleted=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
            width_max=4 * u.um,
            width_depleted=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=14 * u.um,
            width_depleted=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=0 * u.um,
            width_max=4 * u.um,
            width_depleted=0.8 * u.um,
        ),
    ],
)
class TestJanesickDiffusionModel(
    AbstractTestAbstractDiffusionModel,
):
    def test_width_limits(
        self,
        a: optika.sensors.diffusion.JanesickDiffusionModel,
    ):
        """
        The width is the width at the back surface plus the whole spread of the
        depletion region there, and the spread of the depletion region alone
        at its edge.
        """
        s = _thickness_substrate
        f = s - a.thickness_depletion
        width_max = f if a.width_max is None else a.width_max
        width_depleted = 0 * u.um if a.width_depleted is None else a.width_depleted
        if f > 0 * u.um:
            back = np.sqrt(np.square(width_max) + np.square(width_depleted))
            assert np.allclose(a.width(0 * u.um, s), back)
        assert np.allclose(a.width(f, s), width_depleted)

    def test_width_average_janesick(
        self,
        a: optika.sensors.diffusion.JanesickDiffusionModel,
    ):
        """The average width is the closed form of Janesick (2001), generalized."""
        absorption = na.geomspace(1e-4, 1e3, axis="absorption", num=8) / u.um
        result = a.width_average(absorption, _thickness_substrate)
        expected = optika.sensors.charge_diffusion(
            absorption=absorption,
            thickness_substrate=_thickness_substrate,
            thickness_depletion=a.thickness_depletion,
            width_max=a.width_max,
            width_depleted=a.width_depleted,
        )
        assert np.all(result == expected)


@pytest.mark.parametrize(
    argnames="thickness_depletion",
    argvalues=[
        None,
        5 * u.um,
    ],
)
def test_model_or_janesick(
    thickness_depletion: None | u.Quantity,
):
    """With no model, a function uses Janesick's with the given depletion region."""
    result = optika.sensors.diffusion._models._model_or_janesick(
        model_diffusion=None,
        thickness_depletion=thickness_depletion,
        thickness_substrate=_thickness_substrate,
    )
    assert isinstance(result, optika.sensors.diffusion.JanesickDiffusionModel)
    expected = (
        _thickness_substrate if thickness_depletion is None else thickness_depletion
    )
    assert result.thickness_depletion == expected

    model = optika.sensors.diffusion.JanesickDiffusionModel(
        thickness_depletion=3 * u.um
    )
    if thickness_depletion is None:
        result = optika.sensors.diffusion._models._model_or_janesick(
            model_diffusion=model,
            thickness_depletion=None,
            thickness_substrate=_thickness_substrate,
        )
        assert result is model
    else:
        with pytest.raises(ValueError, match="not both"):
            optika.sensors.diffusion._models._model_or_janesick(
                model_diffusion=model,
                thickness_depletion=thickness_depletion,
                thickness_substrate=_thickness_substrate,
            )
