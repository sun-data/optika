import pytest
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests.test_surfaces import AbstractTestAbstractSurface


class AbstractTestAbstractImagingSensor(
    AbstractTestAbstractSurface,
):
    def test_num_pixel(self, a: optika.sensors.AbstractImagingSensor):
        result = a.num_pixel
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.issubdtype(na.get_dtype(result.x), int)
        assert np.issubdtype(na.get_dtype(result.y), int)

    def test_timedelta_exposure(self, a: optika.sensors.AbstractImagingSensor):
        result = a.timedelta_exposure
        assert result >= 0 * u.s

    def test_read_noise(self, a: optika.sensors.AbstractImagingSensor):
        result = a.read_noise
        assert result >= 0 * u.electron

    def test_clip_rays(self, a: optika.sensors.AbstractImagingSensor):
        assert isinstance(a.clip_rays, bool)
        # the light-sensitive area is the same either way, and only whether
        # it vignettes follows the flag
        assert a.aperture.active == a.clip_rays

    def test_pixels(self, a: optika.sensors.AbstractImagingSensor):
        # the corners of the light-sensitive area map to pixel 0 and
        # `num_pixel`
        pixels_lower = a.pixels(a.aperture.bound_lower.xy)
        pixels_upper = a.pixels(a.aperture.bound_upper.xy)
        assert na.unit(pixels_lower).is_equivalent(u.pix)
        assert np.allclose(pixels_lower, 0 * u.pix)
        assert np.allclose(pixels_upper, a.num_pixel * u.pix)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                intensity=100 * u.photon / u.s,
                wavelength=500 * u.nm,
                position=na.Cartesian3dVectorArray() * u.mm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            optika.rays.RayVectorArray(
                intensity=na.random.poisson(100, shape_random=dict(t=11)) * u.erg / u.s,
                wavelength=500 * u.nm,
                position=na.Cartesian3dVectorArray(
                    x=na.random.uniform(-1, 1, shape_random=dict(t=11)) * u.mm,
                    y=na.random.uniform(-1, 1, shape_random=dict(t=11)) * u.mm,
                    z=0 * u.mm,
                ),
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            na.linspace(500, 600, axis="wavelength", num=11) * u.nm,
        ],
    )
    def test_measure(
        self,
        a: optika.sensors.AbstractImagingSensor,
        rays: optika.rays.RayVectorArray,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.measure(rays, wavelength)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert isinstance(result.outputs, na.AbstractScalar)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert a.axis_pixel.x in result.outputs.shape
        assert a.axis_pixel.y in result.outputs.shape

        # Also measure rays whose wavelength varies along more than one axis,
        # for example a scene composed of several disjoint spectral lines, each
        # sampled by its own set of wavelength bins.
        line = na.ScalarArray([500, 600] * u.nm, axes="line")
        wavelength_lines = line + na.linspace(-1, 1, axis="wavelength", num=3) * u.nm
        rays_lines = optika.rays.RayVectorArray(
            intensity=100 * u.photon / u.s,
            wavelength=line + na.linspace(-0.5, 0.5, axis="wavelength", num=2) * u.nm,
            position=na.Cartesian3dVectorArray(
                x=na.random.uniform(-1, 1, shape_random=dict(wavelength=2, t=11)),
                y=na.random.uniform(-1, 1, shape_random=dict(wavelength=2, t=11)),
                z=0,
            )
            * u.mm,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )
        result_lines = a.measure(
            rays_lines,
            wavelength_lines,
            axis=("wavelength", "t"),
            axis_wavelength="wavelength",
        )
        assert isinstance(result_lines, na.FunctionArray)
        assert result_lines.outputs.unit.is_equivalent(u.electron)
        assert "line" in result_lines.outputs.shape
        assert a.axis_pixel.x in result_lines.outputs.shape
        assert a.axis_pixel.y in result_lines.outputs.shape

    @staticmethod
    def _image(
        a: optika.sensors.AbstractImagingSensor,
        outputs: float | na.AbstractScalar,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """An image on 12 by 10 pixels of the sensor, in three wavelength bins."""
        return na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=na.linspace(500, 600, axis="wavelength", num=4) * u.nm,
                position=na.Cartesian2dVectorArray(
                    x=na.arange(0, 12, axis=a.axis_pixel.x) * u.pix,
                    y=na.arange(0, 10, axis=a.axis_pixel.y) * u.pix,
                ),
            ),
            outputs=outputs,
        )

    @staticmethod
    def _shape(
        a: optika.sensors.AbstractImagingSensor,
        wavelength: bool = True,
    ) -> dict[str, int]:
        """The shape of the outputs of :meth:`_image`, with or without wavelength."""
        result = {a.axis_pixel.x: 12, a.axis_pixel.y: 10}
        if wavelength:
            result = dict(wavelength=3) | result
        return result

    def test_kernel(self, a: optika.sensors.AbstractImagingSensor):
        """
        The noiseless exposure of photons absorbed in a single pixel is the
        kernel, centered on that pixel.
        """
        a = dataclasses.replace(a, timedelta_exposure=10 * u.s)
        result = a.kernel(550 * u.nm)

        assert isinstance(result, na.FunctionArray)
        assert result.outputs.unit.is_equivalent(u.electron / u.photon)
        num_x = result.outputs.shape[a.axis_pixel.x]
        num_y = result.outputs.shape[a.axis_pixel.y]

        rate = np.zeros((12, 10)) * u.photon / u.s
        rate[6, 5] = 100 * u.photon / u.s
        image = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray([549, 551] * u.nm, axes="wavelength"),
                position=self._image(a, 0).inputs.position,
            ),
            outputs=na.ScalarArray(rate, axes=(a.axis_pixel.x, a.axis_pixel.y)),
        )
        electrons = a.expose(image, noise=False, integrate=False).outputs
        electrons = electrons[dict(wavelength=0)]

        window = {
            a.axis_pixel.x: slice(6 - num_x // 2, 6 + num_x // 2 + 1),
            a.axis_pixel.y: slice(5 - num_y // 2, 5 + num_y // 2 + 1),
        }
        expected = 100 * u.photon / u.s * a.timedelta_exposure * result.outputs
        assert np.allclose(electrons[window], expected)
        assert np.isclose(electrons.sum(), expected.sum())
        assert electrons[{a.axis_pixel.x: 6, a.axis_pixel.y: 5}] == electrons.max()

    def test_kernel_variance(self, a: optika.sensors.AbstractImagingSensor):
        wavelength = 550 * u.nm
        kernel = a.kernel(wavelength)
        result = a.kernel_variance(wavelength)
        assert isinstance(result, na.FunctionArray)
        assert result.outputs.unit.is_equivalent(u.electron**2 / u.photon)

        # the ratio of the variance to the mean is that of the noise model
        electrons = 1000 * u.electron
        uncertainty = a.material.uncertainty(
            electrons=electrons,
            wavelength=wavelength,
            width_pixel=a.width_pixel,
        )
        vmr = np.square(uncertainty) / electrons
        assert np.allclose(result.outputs, vmr * kernel.outputs)

    @pytest.mark.parametrize("integrate", [False, True])
    def test_expose_transposed(
        self,
        a: optika.sensors.AbstractImagingSensor,
        integrate: bool,
    ):
        """The transpose is the adjoint of the noiseless exposure."""
        a = dataclasses.replace(a, timedelta_exposure=10 * u.s)

        rate = na.random.uniform(0, 100, shape_random=self._shape(a))
        image = self._image(a, rate * u.photon / u.s)
        electrons = na.random.uniform(
            low=0,
            high=1000,
            shape_random=self._shape(a, wavelength=not integrate),
        )
        image_electrons = self._image(a, electrons * u.electron)

        forward = a.expose(image, noise=False, integrate=integrate)
        result = a.expose_transposed(image_electrons)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert "wavelength" in na.shape(result.outputs)
        assert np.isclose(
            (forward.outputs * image_electrons.outputs).sum(),
            (image.outputs * result.outputs).sum(),
            rtol=1e-12,
        )

    def test_backproject(self, a: optika.sensors.AbstractImagingSensor):
        # use a nonzero exposure time so the default `timedelta` is invertible
        a = dataclasses.replace(a, timedelta_exposure=10 * u.s)

        rate = na.random.uniform(0, 100, shape_random=self._shape(a))
        rate = rate * u.photon / u.s
        image = self._image(a, rate)

        electrons = a.expose(image, noise=False, integrate=False)
        result = a.backproject(electrons, integrate=False)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert result.outputs.unit.is_equivalent(u.photon / u.s)

        # without diffusion, the backprojection is the exact inverse of
        # `expose`, and with it, it maps a uniform rate back onto itself
        # in the middle of a sensor wide enough that no charge reaching the
        # middle pixel leaves it
        kernel = a.kernel(550 * u.nm)
        if kernel.outputs.size == 1:
            assert np.allclose(result.outputs, rate)
        else:
            num = 4 * kernel.outputs.shape[a.axis_pixel.x]
            shape = {a.axis_pixel.x: num, a.axis_pixel.y: num}
            uniform = na.FunctionArray(
                inputs=na.SpectralPositionalVectorArray(
                    wavelength=self._image(a, 0).inputs.wavelength,
                    position=na.Cartesian2dVectorArray(
                        x=na.arange(0, num, axis=a.axis_pixel.x) * u.pix,
                        y=na.arange(0, num, axis=a.axis_pixel.y) * u.pix,
                    ),
                ),
                outputs=na.broadcast_to(50 * u.photon / u.s, shape),
            )
            electrons = a.expose(uniform, noise=False, integrate=False)
            result = a.backproject(electrons, integrate=False)
            middle = {a.axis_pixel.x: num // 2, a.axis_pixel.y: num // 2}
            assert np.allclose(result.outputs[middle], 50 * u.photon / u.s, rtol=1e-5)

        # an integrated readout is spread back over the wavelength bins
        integrated = self._image(
            a,
            outputs=na.random.uniform(
                low=0,
                high=1000,
                shape_random=self._shape(a, wavelength=False),
            )
            * u.electron,
        )
        result = a.backproject(integrated, integrate=True)
        assert result.outputs.unit.is_equivalent(u.photon / u.s)
        assert "wavelength" in na.shape(result.outputs)
        assert np.all(np.isfinite(result.outputs.to_value(u.photon / u.s)))

    def test_uncertainty(self, a: optika.sensors.AbstractImagingSensor):
        electrons = na.random.uniform(0, 1000, shape_random=self._shape(a))
        image = self._image(a, electrons * u.electron)

        # integrated over wavelength (the default), with read noise once
        result = a._uncertainty(image)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert "wavelength" not in na.shape(result.outputs)
        # the total noise is at least the read noise (added in quadrature)
        assert np.all(result.outputs >= a.read_noise)

        # per wavelength (no integration, no read noise)
        result = a._uncertainty(image, integrate=False)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert "wavelength" in na.shape(result.outputs)
        assert np.all(result.outputs >= 0 * u.electron)

    def test_expose_uncertainty(self, a: optika.sensors.AbstractImagingSensor):
        """
        The width of the noise of an exposure is that of the noise model at the
        expected electrons, which the kernel spreads over the pixels.
        """
        a = dataclasses.replace(a, timedelta_exposure=10 * u.s)
        rate = na.random.uniform(0, 100, shape_random=self._shape(a))
        image = self._image(a, rate * u.photon / u.s)

        result = a.expose(image, noise=False, uncertainty=True)
        expected = a.expose(image, noise=False, integrate=False)
        width = a._uncertainty(expected)

        assert isinstance(result.outputs, na.NormalUncertainScalarArray)
        assert np.allclose(result.outputs.width, width.outputs)


def _sensor(clip_rays: bool = True) -> optika.sensors.ImagingSensor:
    """A small sensor, 2048 by 1024 pixels of 15 microns."""
    return optika.sensors.ImagingSensor(
        name="test sensor",
        width_pixel=15 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        num_pixel=na.Cartesian2dVectorArray(2048, 1024),
        read_noise=4 * u.electron,
        clip_rays=clip_rays,
        transformation=na.transformations.Cartesian3dTranslation(x=1 * u.mm),
    )


def _sensor_ccd97() -> optika.sensors.ImagingSensor:
    """A sensor of e2v CCD97 silicon, over whose 16-micron pixels charge diffuses."""
    return dataclasses.replace(
        _sensor(),
        width_pixel=16 * u.um,
        material=optika.sensors.materials.e2v_ccd97(),
    )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        _sensor(),
        _sensor(clip_rays=False),
        _sensor_ccd97(),
    ],
)
class TestImagingSensor(
    AbstractTestAbstractImagingSensor,
):
    pass


def test_clip_rays_is_keyword_only():
    """
    `clip_rays` came after the other fields, so it takes no place among the
    positional arguments of an imaging sensor.
    """
    positional = [
        field.name
        for field in dataclasses.fields(optika.sensors.ImagingSensor)
        if not field.kw_only
    ]
    assert "clip_rays" not in positional


def test_clip_rays_off_passes_the_rays_which_miss_the_sensor():
    """
    A sensor which does not clip lets the rays which miss its pixels through
    unvignetted, lands every ray where it would have, and keeps the same
    pixels.
    """
    rays = optika.rays.RayVectorArray(
        wavelength=500 * u.nm,
        position=na.Cartesian3dVectorArray(
            # the sensor is 30.72 mm wide, so the outer two miss it
            x=na.linspace(-20, 20, axis="x", num=5) * u.mm,
            y=0 * u.mm,
            z=-10 * u.mm,
        ),
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )

    clipped = _sensor().propagate_rays(rays)
    unclipped = _sensor(clip_rays=False).propagate_rays(rays)

    assert not np.all(clipped.unvignetted)
    assert np.all(unclipped.unvignetted)
    assert np.allclose(unclipped.position, clipped.position)

    # the aperture keeps its shape, which the pixels are measured from
    assert np.allclose(
        _sensor(clip_rays=False).aperture.wire(),
        _sensor().aperture.wire(),
    )
    position = na.Cartesian2dVectorArray(3, -2) * u.mm
    assert np.allclose(
        _sensor(clip_rays=False).pixels(position),
        _sensor().pixels(position),
    )
