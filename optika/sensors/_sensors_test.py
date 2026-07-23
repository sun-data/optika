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

        # `photons_absorbed` is the deterministic inverse of `expose`
        electrons = a.expose(image, noise=False)
        result = a.photons_absorbed(electrons)

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert result.outputs.unit.is_equivalent(u.photon / u.s)
        assert np.allclose(
            result.outputs.to_value(u.photon / u.s),
            rate.to_value(u.photon / u.s),
        )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.ImagingSensor(
            name="test sensor",
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(2048, 1024),
            transformation=na.transformations.Cartesian3dTranslation(x=1 * u.mm),
        ),
    ],
)
class TestImagingSensor(
    AbstractTestAbstractImagingSensor,
):
    pass
