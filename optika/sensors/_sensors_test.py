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

    def test_photons_absorbed(self, a: optika.sensors.AbstractImagingSensor):
        # use a nonzero exposure time so the default `timedelta` is invertible
        a = dataclasses.replace(a, timedelta_exposure=10 * u.s)

        # a photon rate incident on a few pixels, as a function of the
        # wavelength bin edges
        wavelength = na.linspace(500, 600, axis="wavelength", num=4) * u.nm
        position = na.Cartesian2dVectorArray(
            x=na.arange(0, 5, axis=a.axis_pixel.x) * u.pix,
            y=na.arange(0, 5, axis=a.axis_pixel.y) * u.pix,
        )
        rate = (
            na.random.uniform(
                low=0,
                high=100,
                shape_random={"wavelength": 3, a.axis_pixel.x: 5, a.axis_pixel.y: 5},
            )
            * u.photon
            / u.s
        )
        image = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=wavelength,
                position=position,
            ),
            outputs=rate,
        )

        # per wavelength, `photons_absorbed` is the exact inverse of `expose`
        electrons = a.expose(image, noise=False, integrate=False)
        result = a.photons_absorbed(electrons, integrate=False)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert result.outputs.unit.is_equivalent(u.photon / u.s)
        assert np.allclose(
            result.outputs.to_value(u.photon / u.s),
            rate.to_value(u.photon / u.s),
        )

        # an integrated readout is spread back over the wavelength bins
        integrated = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=wavelength,
                position=position,
            ),
            outputs=na.random.uniform(
                low=0,
                high=1000,
                shape_random={a.axis_pixel.x: 5, a.axis_pixel.y: 5},
            )
            * u.electron,
        )
        result = a.photons_absorbed(integrated, integrate=True)
        assert result.outputs.unit.is_equivalent(u.photon / u.s)
        assert np.all(np.isfinite(result.outputs.to_value(u.photon / u.s)))

    def test_uncertainty(self, a: optika.sensors.AbstractImagingSensor):
        # electrons measured in a few pixels, as a function of the wavelength
        # bin edges
        wavelength = na.linspace(500, 600, axis="wavelength", num=4) * u.nm
        position = na.Cartesian2dVectorArray(
            x=na.arange(0, 5, axis=a.axis_pixel.x) * u.pix,
            y=na.arange(0, 5, axis=a.axis_pixel.y) * u.pix,
        )
        electrons = (
            na.random.uniform(
                low=0,
                high=1000,
                shape_random={"wavelength": 3, a.axis_pixel.x: 5, a.axis_pixel.y: 5},
            )
            * u.electron
        )
        image = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=wavelength,
                position=position,
            ),
            outputs=electrons,
        )

        # integrated over wavelength (the default), with read noise once
        result = a.uncertainty(image)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert "wavelength" not in na.shape(result.outputs)
        # the total noise is at least the read noise (added in quadrature)
        assert np.all(result.outputs >= a.read_noise)

        # per wavelength (no integration, no read noise)
        result = a.uncertainty(image, integrate=False)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert "wavelength" in na.shape(result.outputs)
        assert np.all(result.outputs >= 0 * u.electron)


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


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        _sensor(),
        _sensor(clip_rays=False),
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
