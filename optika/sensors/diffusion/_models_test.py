import warnings
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins
from . import _gaussian

_thickness_substrate = 14 * u.um

_width_pixel = [
    15 * u.um,
    na.Cartesian2dVectorArray(10, 20) * u.um,
]


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

    def test_cdf(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        """
        The profile rises from zero to one, is symmetric about where the
        charge was created, and has the variance of :meth:`width`.
        """
        s = _thickness_substrate
        depth = na.linspace(0, 14, axis="depth", num=8) * u.um
        edges = na.linspace(-80, 80, axis="edge", num=16001) * u.um
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = a.cdf(edges, depth, s)

        assert np.all(np.diff(result, axis="edge") >= 0)
        assert np.allclose(result[dict(edge=0)], 0, atol=1e-12)
        assert np.allclose(result[dict(edge=-1)], 1, atol=1e-12)

        above = a.cdf(5 * u.um, depth, s)
        below = a.cdf(-5 * u.um, depth, s)
        assert np.allclose(above + below, 1)

        fraction = np.diff(result, axis="edge")
        lower = edges[dict(edge=slice(None, -1))]
        upper = edges[dict(edge=slice(1, None))]
        center = (lower + upper) / 2
        variance = (fraction * np.square(center)).sum("edge")
        assert np.allclose(np.sqrt(variance), a.width(depth, s), atol=0.02 * u.um)

    @pytest.mark.parametrize("width_pixel", _width_pixel + [0 * u.um])
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

    @pytest.mark.parametrize("width_pixel", _width_pixel)
    @pytest.mark.parametrize("num", [3, 5])
    def test_kernel(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
        num: int,
    ):
        absorption = na.geomspace(1e-3, 1e3, axis="absorption", num=4) / u.um
        s = _thickness_substrate
        result = a.kernel(absorption, s, width_pixel, "x", "y", num=num)
        assert isinstance(result, na.FunctionArray)
        assert result.outputs.shape == dict(absorption=4, x=num, y=num)
        assert np.all(result.outputs >= 0)
        assert np.allclose(result.outputs.sum(("x", "y")), 1)

        # the kernel is centered on the pixel the photon was absorbed in
        center = result.outputs[dict(x=num // 2, y=num // 2)]
        assert np.all(center == result.outputs.max(("x", "y")))

        with pytest.raises(ValueError, match="odd"):
            a.kernel(absorption, s, width_pixel, "x", "y", num=num + 1)

    @pytest.mark.parametrize("width_pixel", _width_pixel)
    def test_mean_charge_capture(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    ):
        s = _thickness_substrate
        absorption = na.geomspace(1e-3, 1e3, axis="absorption", num=7) / u.um
        result = a.mean_charge_capture(absorption, s, width_pixel)
        assert np.all(result > 0)
        assert np.all(result <= 1)

        # the mean charge capture is the center of a kernel wide enough to
        # hold all of the charge
        kernel = a.kernel(absorption, s, width_pixel, "x", "y", num=15).outputs
        center = kernel[dict(x=7, y=7)]
        assert np.allclose(result, center, rtol=1e-9)

    @pytest.mark.parametrize("width_pixel", _width_pixel + [4 * u.um])
    def test_average_depth_converged(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    ):
        """
        The averages over depth are converged at the default number of
        quadrature nodes, across the full range of optical depths that silicon
        spans between 1 and 10000 angstroms.
        """
        s = _thickness_substrate
        wavelength = na.geomspace(1, 10000, axis="wavelength", num=101) * u.AA
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)

        def averages():
            mcc = a.mean_charge_capture(absorption, s, width_pixel)
            kernel = a.kernel(absorption, s, width_pixel, "x", "y", num=5)
            return mcc, kernel.outputs

        mcc, kernel = averages()

        quadrature = optika.sensors.diffusion._quadrature
        num = quadrature._num_gauss_legendre
        try:
            quadrature._num_gauss_legendre = 8 * num
            mcc_expected, kernel_expected = averages()
        finally:
            quadrature._num_gauss_legendre = num

        assert np.allclose(mcc, mcc_expected, rtol=1e-5)
        assert np.allclose(kernel, kernel_expected, atol=1e-6)

    @pytest.mark.parametrize(
        argnames="absorption",
        argvalues=[
            1e-4 / u.um,
            0.3 / u.um,
            1e4 / u.um,
            na.geomspace(1e-3, 1e3, axis="absorption", num=7) / u.um,
        ],
    )
    def test_average_depth(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
        absorption: u.Quantity | na.AbstractScalar,
    ):
        """
        The quadrature over depth reproduces averages known in closed form:
        a constant, the variance of the charge cloud, and a function with a
        kink, wherever the kink falls.
        """
        s = _thickness_substrate

        result = a._average_depth(lambda depth: 0 * depth / u.um + 1, absorption, s)
        assert np.allclose(result, 1)

        result = a._average_depth(
            integrand=lambda depth: np.square(a.width(depth, s)),
            absorption=absorption,
            thickness_substrate=s,
        )
        expected = np.square(a.width_average(absorption, s))
        assert np.allclose(result, expected, rtol=1e-6, atol=1e-12 * u.um**2)

        alpha = absorption
        absorbed = -np.expm1(-alpha * s)
        for depth_break in [0.01 * u.um, 1 * u.um, 12 * u.um, s, 20 * u.um]:
            result = a._average_depth(
                integrand=lambda depth: np.minimum(depth / depth_break, 1),
                absorption=absorption,
                thickness_substrate=s,
                depth_break=depth_break,
            )
            b = np.minimum(depth_break, s)
            ramp = -np.expm1(-alpha * b) - alpha * b * np.exp(-alpha * b)
            ramp = ramp / (alpha * depth_break)
            expected = ramp + np.exp(-alpha * b) - np.exp(-alpha * s)
            expected = expected / absorbed
            assert np.allclose(result, expected, rtol=1e-6)

    def test_fit_mean_charge_capture(
        self,
        a: optika.sensors.diffusion.AbstractDiffusionModel,
    ):
        """
        The fit reproduces a mean charge capture computed by the model with
        a depletion region inside the light-sensitive region.
        """
        s = _thickness_substrate
        width_pixel = 16 * u.um
        wavelength = na.geomspace(10, 1e4, axis="wavelength", num=31) * u.AA
        absorption = optika.chemicals.Chemical("Si").absorption(wavelength)
        target = a.replace(thickness_depletion=0.3 * s)
        mcc_measured = na.FunctionArray(
            inputs=wavelength,
            outputs=target.mean_charge_capture(absorption, s, width_pixel),
        )
        start = a.replace(thickness_depletion=s / 2)
        result = start.fit_mean_charge_capture(mcc_measured, s, width_pixel)
        assert isinstance(result, type(a))
        mcc = result.mean_charge_capture(absorption, s, width_pixel)
        assert np.allclose(mcc, mcc_measured.outputs, atol=1e-4)

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
            width_backsurface=4 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
            width_depletion=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=8.7 * u.um,
            width_backsurface=4 * u.um,
            width_depletion=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=14 * u.um,
            width_depletion=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=0 * u.um,
            width_backsurface=4 * u.um,
            width_depletion=0.8 * u.um,
        ),
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=20 * u.um,
            width_depletion=0.8 * u.um,
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
        width_backsurface = f if a.width_backsurface is None else a.width_backsurface
        width_depletion = 0 * u.um if a.width_depletion is None else a.width_depletion
        if f > 0 * u.um:
            back = np.sqrt(np.square(width_backsurface) + np.square(width_depletion))
            assert np.allclose(a.width(0 * u.um, s), back)
        assert np.allclose(a.width(f, s), width_depletion)

    def test_width_average_janesick(
        self,
        a: optika.sensors.diffusion.JanesickDiffusionModel,
    ):
        """With Janesick's widths the average reduces to his closed form."""
        s = _thickness_substrate
        absorption = na.geomspace(1e-4, 1e3, axis="absorption", num=8) / u.um
        janesick = a.replace(width_backsurface=None, width_depletion=None)
        result = janesick.width_average(absorption, s)
        f = np.maximum(s - a.thickness_depletion, 0 * s)
        k = absorption
        expected = np.sqrt(
            f * (k * f + np.exp(-k * f) - 1) / (k * (1 - np.exp(-k * s)))
        )
        assert np.allclose(result, expected, rtol=1e-6, atol=1e-12 * u.um)

    @pytest.mark.parametrize("width_pixel", _width_pixel)
    def test_mean_charge_capture_depth(
        self,
        a: optika.sensors.diffusion.JanesickDiffusionModel,
        width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    ):
        """
        The mean charge capture averages that of the Gaussian charge cloud of
        each depth over the depth at which the photons are absorbed,
        rather than taking that of a single Gaussian with the average variance.
        """
        s = _thickness_substrate
        absorption = na.geomspace(1e-3, 1e3, axis="absorption", num=7) / u.um
        result = a.mean_charge_capture(absorption, s, width_pixel)

        if not isinstance(width_pixel, na.AbstractCartesian2dVectorArray):
            width_pixel = na.Cartesian2dVectorArray(width_pixel, width_pixel)

        axis = "depth"
        num = 100000
        depth = (na.arange(0, num, axis=axis) + 0.5) * s / num
        weight = np.exp(-absorption * depth)
        width = a.width(depth, s)
        capture = _gaussian._capture(_gaussian._ratio(width, width_pixel.x))
        capture = capture * _gaussian._capture(_gaussian._ratio(width, width_pixel.y))
        expected = (capture * weight).sum(axis) / weight.sum(axis)

        assert np.allclose(result, expected, rtol=1e-5)
