import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._systems_test import AbstractTestAbstractSystem


def _distortion() -> optika.distortion.SimpleDistortionModel:
    return optika.distortion.SimpleDistortionModel(
        plate_scale=0.75 * u.arcsec / u.pix,
        dispersion=3.75 * u.nm / u.pix,
        angle=0 * u.deg,
        reference=na.SpectralPositionalVectorArray(
            wavelength=550 * u.nm,
            position=na.Cartesian2dVectorArray(16, 16) * u.pix,
        ),
    )


def _area_effective() -> optika.radiometry.InterpolatedEffectiveAreaModel:
    return optika.radiometry.InterpolatedEffectiveAreaModel(
        wavelength=na.linspace(400, 700, axis="wavelength", num=10) * u.nm,
        area=na.linspace(1, 2, axis="wavelength", num=10) * u.cm**2,
        axis_wavelength="wavelength",
    )


def _vignetting() -> optika.radiometry.PolynomialVignettingModel:
    scene = na.SpectralPositionalVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-10 * u.arcsec,
            stop=+10 * u.arcsec,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
        ),
    )
    return optika.radiometry.PolynomialVignettingModel(
        coordinates_scene=scene,
        illumination=1 - 0.001 * (scene.position.length / u.arcsec) ** 2,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=2,
    )


def _sensor() -> optika.sensors.ImagingSensor:
    return optika.sensors.ImagingSensor(
        width_pixel=15 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        timedelta_exposure=1 * u.s,
        num_pixel=na.Cartesian2dVectorArray(32, 32),
    )


def _scene(radiance: u.Quantity | na.AbstractScalar) -> na.FunctionArray:
    return na.FunctionArray(
        inputs=na.SpectralPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=4) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=11,
            ),
        ),
        outputs=na.random.uniform(
            low=0 * radiance,
            high=radiance,
            shape_random=dict(field_x=10, field_y=10),
        ),
    )


def _scene_doppler(radiance: u.Quantity | na.AbstractScalar) -> na.FunctionArray:
    return na.FunctionArray(
        inputs=na.DopplerPositionalVectorArray(
            wavelength=na.linspace(500, 600, axis="wavelength", num=4) * u.nm,
            wavelength_rest=550 * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                start=-10 * u.arcsec,
                stop=+10 * u.arcsec,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=11,
            ),
        ),
        outputs=na.random.uniform(
            low=0 * radiance,
            high=radiance,
            shape_random=dict(field_x=10, field_y=10),
        ),
    )


class AbstractTestAbstractLinearSystem(
    AbstractTestAbstractSystem,
):
    def test_distortion(self, a: optika.systems.AbstractLinearSystem):
        assert isinstance(a.distortion, optika.distortion.AbstractDistortionModel)

    def test_area_effective(self, a: optika.systems.AbstractLinearSystem):
        assert isinstance(
            a.area_effective,
            optika.radiometry.AbstractEffectiveAreaModel,
        )

    def test_sensor(self, a: optika.systems.AbstractLinearSystem):
        assert isinstance(a.sensor, optika.sensors.AbstractImagingSensor)

    def test_coordinates_sensor(self, a: optika.systems.AbstractLinearSystem):
        result = a.coordinates_sensor
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert na.unit_normalized(result).is_equivalent(u.pix)

    @pytest.mark.parametrize(
        argnames="radiance",
        argvalues=[
            1e-18 * u.W / u.cm**2 / u.arcsec**2 / u.nm,
            1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm,
        ],
    )
    @pytest.mark.parametrize("noise", [False, True])
    def test_image(
        self,
        a: optika.systems.AbstractLinearSystem,
        noise: bool,
        radiance: u.Quantity,
    ):
        scene = _scene(radiance)
        result = a.image(scene, noise=noise)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert na.unit(result.outputs).is_equivalent(u.electron)
        assert np.all(np.isfinite(result.outputs.value))

        # the scene is binned onto the sensor pixel grid
        axis_pixel = a.sensor.axis_pixel
        assert axis_pixel.x in na.shape(result.outputs)
        assert axis_pixel.y in na.shape(result.outputs)

        if not noise:
            assert np.all(result.outputs >= 0 * u.electron)
            assert result.outputs.sum() > 0 * u.electron

    def test_image_doppler(self, a: optika.systems.AbstractLinearSystem):
        # a scene defined on a Doppler grid (as produced by the ctis
        # `IdealInstrument`) is accepted and imaged onto the sensor.
        scene = _scene_doppler(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result = a.image(scene, noise=False)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert na.unit(result.outputs).is_equivalent(u.electron)
        assert np.all(np.isfinite(result.outputs.value))
        assert result.outputs.sum() > 0 * u.electron

    @pytest.mark.parametrize(
        argnames="radiance",
        argvalues=[
            1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm,
        ],
    )
    def test_backproject(
        self,
        a: optika.systems.AbstractLinearSystem,
        radiance: u.Quantity,
    ):
        scene = _scene(radiance)
        image = a.image(scene, noise=False)
        result = a.backproject(image, scene.inputs)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)

        # the detector response is inverted, so the backprojection recovers a
        # spectral radiance with the same units as the original scene.
        assert na.unit(result.outputs).is_equivalent(na.unit(radiance))
        assert np.all(np.isfinite(result.outputs.value))
        assert result.outputs.sum() > 0 * na.unit(radiance)

    def test_backproject_doppler(self, a: optika.systems.AbstractLinearSystem):
        # a Doppler object-plane grid is accepted, and the backprojected
        # radiance is returned on that same Doppler grid.
        scene = _scene_doppler(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        image = a.image(scene, noise=False)
        result = a.backproject(image, scene.inputs)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.DopplerPositionalVectorArray)
        assert na.unit(result.outputs).is_equivalent(
            u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm
        )
        assert np.all(np.isfinite(result.outputs.value))

    def test_backproject_unit(self, a: optika.systems.AbstractLinearSystem):
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        image = a.image(scene, noise=False)

        # the backprojection can be expressed in photon or energy units
        unit_photon = u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm
        unit_energy = u.W / u.cm**2 / u.arcsec**2 / u.nm

        result_photon = a.backproject(image, scene.inputs, unit=unit_photon)
        result_energy = a.backproject(image, scene.inputs, unit=unit_energy)

        assert na.unit_normalized(result_photon.outputs).is_equivalent(unit_photon)
        assert na.unit_normalized(result_energy.outputs).is_equivalent(unit_energy)
        assert np.all(np.isfinite(result_energy.outputs.value))

        # a unit compatible with neither is rejected
        with pytest.raises(u.UnitConversionError):
            a.backproject(image, scene.inputs, unit=u.s)

    def test_from_weights_infer_axes(self, a: optika.systems.AbstractLinearSystem):
        # `image_from_weights`/`backproject_from_weights` infer `axis_wavelength`
        # and `axis_field` from the scene when they are not given explicitly.
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        axis_wavelength = "wavelength"
        axis_field = ("field_x", "field_y")

        weights = a.weights(scene.inputs, axis_wavelength, axis_field)

        image = a.image_from_weights(weights, scene, noise=False, integrate=False)
        assert isinstance(image, na.FunctionArray)
        assert na.unit(image.outputs).is_equivalent(u.electron)
        assert np.all(np.isfinite(image.outputs.value))

        weights_transposed = a.weights_transposed(
            weights, scene.inputs, axis_wavelength, axis_field
        )
        result = a.backproject_from_weights(
            weights_transposed, image, scene.inputs, integrate=False
        )
        assert isinstance(result, na.FunctionArray)
        assert na.unit(result.outputs).is_equivalent(
            u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm
        )
        assert np.all(np.isfinite(result.outputs.value))

    def test_image_uncertainty(self, a: optika.systems.AbstractLinearSystem):
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result = a.image(scene, noise=False, uncertainty=True)

        # the measurement noise is attached as a normal uncertain array
        assert isinstance(result.outputs, na.NormalUncertainScalarArray)
        assert na.unit(result.outputs.nominal).is_equivalent(u.electron)
        assert na.unit(result.outputs.width).is_equivalent(u.electron)
        assert np.all(result.outputs.width >= 0 * u.electron)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.systems.LinearSystem(
            area_effective=_area_effective(),
            distortion=_distortion(),
            sensor=_sensor(),
        ),
        optika.systems.LinearSystem(
            area_effective=_area_effective(),
            distortion=_distortion(),
            sensor=_sensor(),
            vignetting=_vignetting(),
            field_stop=optika.apertures.RectangularAperture(half_width=15 * u.arcsec),
        ),
    ],
)
class TestLinearSystem(
    AbstractTestAbstractLinearSystem,
):
    pass
