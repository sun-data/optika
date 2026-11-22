import dataclasses

import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.sensors._sensors_test import _double_readout, _assert_doubled
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
            num=6,
        ),
    )
    centers = scene.cell_centers(("field_x", "field_y"))
    return optika.radiometry.PolynomialVignettingModel(
        coordinates_scene=scene,
        illumination=1 - 0.001 * (centers.position.length / u.arcsec) ** 2,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=2,
    )


def _field_stop() -> optika.radiometry.PolynomialFieldStopModel:
    """
    A square field of view 30 arcseconds across which drifts along :math:`x`
    with wavelength, as the field of view of a system whose field stop sits
    behind its grating does.
    """
    wavelength = na.linspace(500, 600, axis="wavelength", num=3) * u.nm
    angle = na.linspace(45, 405, axis="vertex", num=5) * u.deg
    shift = (wavelength - 550 * u.nm) * (0.02 * u.arcsec / u.nm)
    radius = 15 * np.sqrt(2) * u.arcsec
    return optika.radiometry.PolynomialFieldStopModel(
        wavelength=wavelength,
        vertices=na.Cartesian2dVectorArray(
            x=radius * np.cos(angle) + shift,
            y=radius * np.sin(angle) + 0 * shift,
        ),
        axis_wavelength="wavelength",
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


def _scene_spectral(radiance: u.Quantity | na.AbstractScalar) -> na.FunctionArray:
    """A scene whose radiance varies with wavelength as well as position."""
    scene = _scene(radiance)
    return scene.replace(
        outputs=na.random.uniform(
            low=0 * radiance,
            high=radiance,
            shape_random=dict(wavelength=3, field_x=10, field_y=10),
        ),
    )


def _scene_empty(radiance: u.Quantity | na.AbstractScalar) -> na.FunctionArray:
    """A scene with a single wavelength vertex, and so no wavelength bins."""
    scene = _scene(radiance)
    return scene.replace(
        inputs=scene.inputs.replace(
            wavelength=na.linspace(500, 500, axis="wavelength", num=1) * u.nm,
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

    def test_field_stop_(self, a: optika.systems.AbstractLinearSystem):
        result = a.field_stop_
        if a.field_stop is None:
            assert result is None
        else:
            assert isinstance(result, optika.radiometry.AbstractFieldStopModel)

    def test_footprint(self, a: optika.systems.AbstractLinearSystem):
        wavelength = 550 * u.nm
        if a.field_stop is None:
            with pytest.raises(ValueError):
                a.footprint(wavelength)
        else:
            # one outline at a single wavelength, one per wavelength at many
            result = a.footprint(wavelength)
            assert isinstance(result, na.AbstractCartesian2dVectorArray)
            assert "wire" in na.shape(result)
            assert "wavelength" not in na.shape(result)
            wavelength = na.linspace(500, 600, axis="wavelength", num=5) * u.nm
            result = a.footprint(wavelength)
            assert na.shape(result)["wavelength"] == 5

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

    def test_image_area_unit_invariance(
        self,
        a: optika.systems.AbstractLinearSystem,
    ):
        # the result must not depend on the unit the effective area model
        # happens to be expressed in, since `weights` strips the unit and
        # `image_from_weights` restores it as `weights_unit`
        area = a.area_effective
        if not hasattr(area, "area"):  # pragma: nocover
            pytest.skip("requires an area model with an `area` attribute")
        b = dataclasses.replace(
            a,
            area_effective=dataclasses.replace(area, area=area.area.to(u.mm**2)),
        )
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result_a = a.image(scene, noise=False)
        result_b = b.image(scene, noise=False)
        assert np.allclose(result_a.outputs, result_b.outputs)

    def test_image_device_host(self, a: optika.systems.AbstractLinearSystem):
        # `device=None` is the host path and must reproduce the default exactly
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result = a.image(scene, noise=False)
        result_host = a.image(scene, noise=False, device=None)
        assert np.all(result.outputs == result_host.outputs)

    def test_image_uncertainty(self, a: optika.systems.AbstractLinearSystem):
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result = a.image(scene, noise=False, uncertainty=True)

        # the measurement noise is attached as a normal uncertain array
        assert isinstance(result.outputs, na.NormalUncertainScalarArray)
        assert na.unit(result.outputs.nominal).is_equivalent(u.electron)
        assert na.unit(result.outputs.width).is_equivalent(u.electron)
        assert np.all(result.outputs.width >= 0 * u.electron)

    @pytest.mark.parametrize(
        argnames="scene",
        argvalues=[
            _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
            _scene_spectral(1e-18 * u.W / u.cm**2 / u.arcsec**2 / u.nm),
            _scene_doppler(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
            _scene_empty(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
        ],
    )
    @pytest.mark.parametrize("chunk_size", [None, 1, 2, 10])
    @pytest.mark.parametrize("integrate", [True, False])
    @pytest.mark.parametrize("uncertainty", [False, True])
    def test_image_chunk_size(
        self,
        a: optika.systems.AbstractLinearSystem,
        scene: na.FunctionArray,
        chunk_size: None | int,
        integrate: bool,
        uncertainty: bool,
    ) -> None:
        # imaging the scene a few wavelength bins at a time and summing the
        # chunks gives the same image as exposing the sensor to all of them at
        # once with `image_from_weights`; the read noise is not zero, so that
        # the width shows whether it is added once
        sensor = dataclasses.replace(a.sensor, read_noise=10 * u.electron)
        b = dataclasses.replace(a, sensor=sensor)
        kwargs = dict(noise=False, integrate=integrate, uncertainty=uncertainty)
        result = b.image(scene, chunk_size=chunk_size, **kwargs)

        axis_wavelength = "wavelength"
        axis_field = ("field_x", "field_y")
        weights = b.weights(scene.inputs, axis_wavelength, axis_field)
        expected = b.image_from_weights(weights, scene, **kwargs)

        assert isinstance(result, na.FunctionArray)
        assert np.all(result.inputs == expected.inputs)

        if uncertainty:
            # two normal uncertain arrays built separately draw different
            # samples, so the distributions are compared through the nominal
            # value and width which define them
            assert isinstance(result.outputs, na.NormalUncertainScalarArray)
            assert np.allclose(result.outputs.nominal, expected.outputs.nominal)
            assert np.allclose(result.outputs.width, expected.outputs.width)
        else:
            assert np.allclose(result.outputs, expected.outputs)

    @pytest.mark.parametrize(
        argnames="scene",
        argvalues=[
            _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
            _scene_spectral(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
            _scene_doppler(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm),
        ],
    )
    @pytest.mark.parametrize("chunk_size", [None, 1, 2, 10])
    @pytest.mark.parametrize("integrate", [True, False])
    @pytest.mark.parametrize(
        argnames="unit",
        argvalues=[None, u.W / u.cm**2 / u.arcsec**2 / u.nm],
    )
    def test_backproject_chunk_size(
        self,
        a: optika.systems.AbstractLinearSystem,
        scene: na.FunctionArray,
        chunk_size: None | int,
        integrate: bool,
        unit: None | u.UnitBase,
    ) -> None:
        # backprojecting onto a few wavelength bins at a time and joining the
        # chunks gives the same radiance as `backproject_from_weights` with
        # the transposed weights of every bin
        image = a.image(scene, noise=False, integrate=integrate)
        result = a.backproject(
            image,
            scene.inputs,
            integrate=integrate,
            unit=unit,
            chunk_size=chunk_size,
        )

        axis_wavelength = "wavelength"
        axis_field = ("field_x", "field_y")
        weights = a.weights(scene.inputs, axis_wavelength, axis_field)
        weights_transposed = a.weights_transposed(
            weights, scene.inputs, axis_wavelength, axis_field
        )
        expected = a.backproject_from_weights(
            weights_transposed,
            image,
            scene.inputs,
            integrate=integrate,
            unit=unit,
        )

        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, type(expected.inputs))
        assert np.all(result.inputs == expected.inputs)
        assert np.allclose(result.outputs, expected.outputs)

    @pytest.mark.parametrize("chunk_size", [None, 1])
    def test_image_chunk_size_read_noise(
        self,
        a: optika.systems.AbstractLinearSystem,
        chunk_size: None | int,
    ) -> None:
        # the read noise is added once, to the sum of the chunks, so a dark
        # scene imaged one wavelength bin at a time has the read noise of one
        # readout, not of one readout per bin
        read_noise = 10 * u.electron
        sensor = dataclasses.replace(a.sensor, read_noise=read_noise)
        b = dataclasses.replace(a, sensor=sensor)
        scene = _scene(0 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        result = b.image(scene, chunk_size=chunk_size)

        # 32 by 32 pixels measure the width to about 2 percent, and a readout
        # for each of the 3 bins would give sqrt(3) times the read noise
        std = np.std(result.outputs.ndarray)
        assert np.isclose(std, read_noise, rtol=0.15)

    @pytest.mark.parametrize("chunk_size", [None, 1])
    @pytest.mark.parametrize("uncertainty", [False, True])
    def test_image_readout(
        self,
        a: optika.systems.AbstractLinearSystem,
        monkeypatch: pytest.MonkeyPatch,
        chunk_size: None | int,
        uncertainty: bool,
    ) -> None:
        # the chunks are read out once by the `readout` of the sensor, so a
        # sensor which overrides it changes the image
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        kwargs = dict(noise=False, uncertainty=uncertainty, chunk_size=chunk_size)
        expected = a.image(scene, **kwargs)
        _double_readout(monkeypatch, a.sensor)
        result = a.image(scene, **kwargs)
        _assert_doubled(result.outputs, expected.outputs)

    def test_weights_reference_wavelength(
        self,
        a: optika.systems.AbstractLinearSystem,
    ) -> None:
        # a single wavelength bin centered on the reference wavelength of the
        # distortion, where its dispersion term is zero at every point, keeps
        # its wavelength axis in the weights and their transpose, and can be
        # imaged and backprojected
        distortion = a.distortion
        if not isinstance(
            distortion, optika.distortion.SimpleDistortionModel
        ):  # pragma: nocover
            pytest.skip("requires a distortion model with a reference wavelength")

        axis_wavelength = "wavelength"
        axis_field = ("field_x", "field_y")
        offset = na.linspace(-10, 10, axis=axis_wavelength, num=2) * u.nm
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        scene = scene.replace(
            inputs=scene.inputs.replace(
                wavelength=distortion.reference.wavelength + offset,
            ),
        )

        weights = a.weights(scene.inputs, axis_wavelength, axis_field)
        _, shape_input, shape_output = weights
        assert shape_input[axis_wavelength] == 1
        assert shape_output[axis_wavelength] == 1

        _, shape_input, shape_output = a.weights_transposed(
            weights, scene.inputs, axis_wavelength, axis_field
        )
        assert shape_input[axis_wavelength] == 1
        assert shape_output[axis_wavelength] == 1

        image = a.image(scene, noise=False)
        assert np.all(np.isfinite(image.outputs.value))
        assert image.outputs.sum() > 0 * u.electron

        result = a.backproject(image, scene.inputs)
        assert np.all(np.isfinite(result.outputs.value))

    @pytest.mark.parametrize("chunk_size", [0, -1, 2.0, True])
    def test_chunk_size_invalid(
        self,
        a: optika.systems.AbstractLinearSystem,
        chunk_size: object,
    ) -> None:
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        with pytest.raises(ValueError, match="chunk_size"):
            a.image(scene, chunk_size=chunk_size)
        image = a.image(scene, noise=False)
        with pytest.raises(ValueError, match="chunk_size"):
            a.backproject(image, scene.inputs, chunk_size=chunk_size)

    @pytest.mark.parametrize("chunk_size", [None, 1])
    def test_cells_invalid(
        self,
        a: optika.systems.AbstractLinearSystem,
        chunk_size: None | int,
    ) -> None:
        # a radiance with a value for each wavelength vertex, or an image with
        # more wavelength bins than the grid it is backprojected onto, is
        # rejected rather than cut to fit the chunks
        unit = u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm
        radiance = na.linspace(1e3, 2e3, axis="wavelength", num=4) * unit
        with pytest.raises(ValueError, match="cells"):
            a.image(_scene(radiance), chunk_size=chunk_size)

        scene = _scene(1e3 * unit)
        image = a.image(scene, noise=False, integrate=False)
        coordinates = scene.inputs[dict(wavelength=slice(0, 3))]
        with pytest.raises(ValueError, match="cells"):
            a.backproject(image, coordinates, integrate=False, chunk_size=chunk_size)

    @pytest.mark.parametrize("chunk_size", [None, 1])
    def test_image_axis_field_missing(
        self,
        a: optika.systems.AbstractLinearSystem,
        chunk_size: None | int,
    ) -> None:
        # an axis of the field which the grid does not vary along has no
        # cells to image
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        with pytest.raises(ValueError, match="no axis 'field_z'"):
            a.image(scene, axis_field=("field_x", "field_z"), chunk_size=chunk_size)

    def test_backproject_weights(
        self,
        a: optika.systems.AbstractLinearSystem,
    ) -> None:
        # precomputed forward weights give the same backprojection as weights
        # built on the fly
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        image = a.image(scene, noise=False)
        weights = a.weights(scene.inputs, "wavelength", ("field_x", "field_y"))
        result = a.backproject(image, scene.inputs, weights=weights)
        expected = a.backproject(image, scene.inputs)
        assert np.all(result.outputs == expected.outputs)

    def test_backproject_chunk_size_weights(
        self,
        a: optika.systems.AbstractLinearSystem,
    ) -> None:
        # precomputed weights cover every wavelength bin, so they cannot be
        # backprojected a chunk at a time
        scene = _scene(1e3 * u.photon / u.s / u.cm**2 / u.arcsec**2 / u.nm)
        image = a.image(scene, noise=False)
        weights = a.weights(scene.inputs, "wavelength", ("field_x", "field_y"))
        with pytest.raises(ValueError, match="chunk_size"):
            a.backproject(image, scene.inputs, weights=weights, chunk_size=1)


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
        optika.systems.LinearSystem(
            area_effective=_area_effective(),
            distortion=_distortion(),
            sensor=_sensor(),
            vignetting=_vignetting(),
            field_stop=_field_stop(),
        ),
    ],
)
class TestLinearSystem(
    AbstractTestAbstractLinearSystem,
):
    pass
