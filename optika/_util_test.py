import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests._measured import interp_each


@pytest.mark.parametrize(
    argnames="angles",
    argvalues=[
        na.Cartesian2dVectorArray(1, 2) * u.deg,
        na.Cartesian2dVectorLinearSpace(
            start=0,
            stop=1,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=5,
        ),
    ],
)
def test_direction(angles: na.AbstractCartesian2dVectorArray):
    result = optika.direction(angles)

    rotation_x = na.Cartesian3dYRotationMatrixArray(-angles.x)
    rotation_y = na.Cartesian3dXRotationMatrixArray(+angles.y)
    result_expected = rotation_x @ rotation_y @ na.Cartesian3dVectorArray(z=1)
    # result_expected = rotation_y @ rotation_x @ na.Cartesian3dVectorArray(z=1)

    print(f"{result=}")
    print(f"{result_expected=}")

    assert isinstance(result, na.AbstractCartesian3dVectorArray)
    assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        na.Cartesian3dVectorArray(1, 2, 5).normalized,
    ],
)
def test_angles(direction: na.AbstractCartesian3dVectorArray):
    result = optika.angles(direction)

    print(f"{direction=}")
    print(f"{optika.direction(result)=}")

    assert isinstance(result, na.AbstractCartesian2dVectorArray)
    assert np.allclose(direction, optika.direction(result))


_wavelength_measured = na.linspace(100, 200, axis="wavelength", num=11) * u.nm
_angle_measured = na.ScalarArray([10, 14, 18] * u.deg, axes="angle")
_angle_rays = na.ScalarArray([0, 10, 12, 14, 17, 18, 30] * u.deg, axes="ray")


def _rays(angle: na.AbstractScalar) -> optika.rays.RayVectorArray:
    return optika.rays.RayVectorArray(
        wavelength=150 * u.nm,
        direction=na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=np.cos(angle),
        ),
    )


def _linear(wavelength, angle):
    """An efficiency linear in both, which linear interpolation reproduces."""
    return 0.1 + 0.001 * wavelength.to(u.nm).value + 0.01 * angle.to(u.deg).value


def test_interp_efficiency_measured_single_angle():
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_measured,
            direction=4 * u.deg,
        ),
        outputs=_linear(_wavelength_measured, 0 * u.deg),
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )
    # a single measurement applies at every angle of incidence
    assert np.allclose(result, _linear(150 * u.nm, 0 * u.deg))


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        _wavelength_measured,
        # each angle sampled at its own wavelengths
        _wavelength_measured + _angle_measured.value * u.nm,
    ],
)
@pytest.mark.parametrize(
    argnames="angle_measured",
    argvalues=[
        _angle_measured,
        # different angles for each of two channels
        _angle_measured + na.ScalarArray([0, 2] * u.deg, axes="channel"),
    ],
)
@pytest.mark.parametrize(
    argnames="normal",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, -1),
        na.Cartesian3dVectorArray(0, 0, 2),
    ],
)
def test_interp_efficiency_measured(
    wavelength: na.AbstractScalar,
    angle_measured: na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
):
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=angle_measured,
        ),
        outputs=_linear(wavelength, angle_measured),
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=normal,
        axis_angle="angle",
    )

    # interpolated linearly inside the measured angles, and held at the
    # nearest measurement outside them
    angle = np.minimum(
        np.maximum(_angle_rays, angle_measured[{"angle": 0}]),
        angle_measured[{"angle": ~0}],
    )
    result_expected = _linear(150 * u.nm, angle)

    assert result.shape == result_expected.shape
    assert np.allclose(result, result_expected)


def test_shape_efficiency_measured() -> None:
    """
    The wavelength and the angles are interpolated over, and leave the shape.

    Without an axis of the angles, a measurement at several angles is refused
    by the shape too, see test_efficiency_measured_more_than_one_direction.
    """
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_measured,
            direction=_angle_measured,
        ),
        outputs=_linear(_wavelength_measured, _angle_measured)
        + na.linspace(0, 0.1, axis="channel", num=2),
    )
    result = optika._util._shape_efficiency_measured(
        measurement=measurement,
        axis_angle="angle",
    )
    assert result == dict(channel=2)


@pytest.mark.parametrize(
    argnames="wavelength,direction,axis_angle,error,match",
    argvalues=[
        # wavelengths which vary along two axes need the one to interpolate
        # along named
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            4 * u.deg,
            None,
            ValueError,
            "specify the axis along which the wavelength varies",
        ),
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            _angle_measured,
            "angle",
            ValueError,
            "specify the axis along which the wavelength varies",
        ),
        # more than one angle needs the axis they vary along
        (
            _wavelength_measured,
            _angle_measured,
            None,
            ValueError,
            "specify the axis along which the angle of incidence varies",
        ),
        # more than one angle must be given as angles of incidence
        (
            _wavelength_measured,
            na.Cartesian3dVectorArray(0, 0, na.linspace(1, 2, axis="angle", num=3)),
            "angle",
            TypeError,
            "scalar angles of incidence",
        ),
        (
            _wavelength_measured,
            _angle_measured,
            "other",
            ValueError,
            "have no axis 'other'",
        ),
        (
            _wavelength_measured,
            na.ScalarArray([10, 14, 18] * u.m, axes="angle"),
            "angle",
            ValueError,
            "must be angles",
        ),
        (
            _wavelength_measured,
            na.ScalarArray([18, 14, 10] * u.deg, axes="angle"),
            "angle",
            ValueError,
            "must increase along 'angle'",
        ),
    ],
)
def test_efficiency_measured_error(
    wavelength: na.AbstractScalar,
    direction: na.AbstractArray | u.Quantity,
    axis_angle: None | str,
    error: type[Exception],
    match: str,
) -> None:
    """The shape and the interpolation refuse the same measurements alike."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=direction,
        ),
        outputs=0.5 + 0 * wavelength.value,
    )
    with pytest.raises(error, match=match):
        optika._util._shape_efficiency_measured(
            measurement=measurement,
            axis_angle=axis_angle,
        )
    with pytest.raises(error, match=match):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            axis_angle=axis_angle,
        )


_wavelength_channel = _wavelength_measured + na.ScalarArray(
    [0, 3] * u.nm, axes="channel"
)
"""
Wavelength samples which differ for each of two channels.

These are in the units and range of :func:`_linear`, which interpolation
reproduces exactly, where the class tests share theirs from
:mod:`optika._tests._measured`.
"""

_offset_channel = na.ScalarArray(np.array([0, 0.1]), axes="channel")
"""An efficiency added to each channel, so that the channels can be told apart."""


@pytest.mark.parametrize(
    argnames="direction,axis_angle",
    argvalues=[
        (4 * u.deg, None),
        (_angle_measured, "angle"),
    ],
)
def test_interp_efficiency_measured_axis_wavelength(
    direction: na.AbstractScalar | u.Quantity,
    axis_angle: None | str,
) -> None:
    """
    Wavelengths which vary along another axis besides the one interpolated
    over are interpolated separately for each index along it.
    """
    angle = 0 * u.deg if axis_angle is None else direction
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_channel,
            direction=direction,
        ),
        outputs=_linear(_wavelength_channel, angle) + _offset_channel,
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        axis_angle=axis_angle,
        axis_wavelength="wavelength",
    )

    if axis_angle is None:
        angle_expected = 0 * u.deg
    else:
        angle_expected = np.minimum(
            np.maximum(_angle_rays, direction[{"angle": 0}]),
            direction[{"angle": ~0}],
        )
    result_expected = _linear(150 * u.nm, angle_expected) + _offset_channel

    assert result.shape == result_expected.shape
    assert np.allclose(result, result_expected)


def test_interp_efficiency_measured_axis_wavelength_broadcasts_rays() -> None:
    """Rays with an axis of the measurement are matched to it, not crossed with it."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_channel,
            direction=4 * u.deg,
        ),
        outputs=_linear(_wavelength_channel, 0 * u.deg) + _offset_channel,
    )
    wavelength = na.ScalarArray([120, 180] * u.nm, axes="channel")
    rays = optika.rays.RayVectorArray(
        wavelength=wavelength,
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=rays,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        axis_wavelength="wavelength",
    )
    result_expected = _linear(wavelength, 0 * u.deg) + _offset_channel
    assert result.shape == dict(channel=2)
    assert np.allclose(result, result_expected)


def test_shape_efficiency_measured_axis_wavelength() -> None:
    """The axes the wavelengths vary along, besides the interpolated one, stay."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_channel,
            direction=_angle_measured,
        ),
        outputs=_linear(_wavelength_channel, _angle_measured),
    )
    result = optika._util._shape_efficiency_measured(
        measurement=measurement,
        axis_angle="angle",
        axis_wavelength="wavelength",
    )
    assert result == dict(channel=2)


def test_interp_efficiency_measured_stacked() -> None:
    """
    Measurements stacked along an axis, each at the same single angle, are
    interpolated each on its own wavelengths.
    """
    measurements = [
        na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=_wavelength_measured + offset * u.nm,
                direction=na.Cartesian3dVectorArray(0, 0, 1),
            ),
            outputs=_linear(_wavelength_measured, 0 * u.deg) + 0.1 * offset,
        )
        for offset in (0, 3)
    ]
    measurement = na.stack(measurements, axis="channel")
    assert "channel" in measurement.inputs.direction.shape
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        axis_wavelength="wavelength",
    )
    expected = interp_each(
        x=150 * u.nm,
        xp=measurement.inputs.wavelength,
        fp=measurement.outputs,
        axis="wavelength",
        axis_each="channel",
    )
    assert result.shape == dict(channel=2)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        _wavelength_measured,
        # each angle with its own samples, so that the wavelengths vary
        # along the axis of the angles too
        _wavelength_measured + _angle_measured.value * u.nm,
    ],
)
@pytest.mark.parametrize(
    argnames="direction,outputs",
    argvalues=[
        # a measurement at several angles whose axis was not named
        (_angle_measured, _linear(_wavelength_measured, _angle_measured)),
        # separate measurements, each at an angle of its own
        (
            na.ScalarArray([4, 6] * u.deg, axes="channel"),
            _linear(_wavelength_measured, 0 * u.deg) + _offset_channel,
        ),
        # an angle for each wavelength sample
        (
            na.linspace(5, 15, axis="wavelength", num=11) * u.deg,
            _linear(_wavelength_measured, 0 * u.deg),
        ),
    ],
)
def test_efficiency_measured_more_than_one_direction(
    wavelength: na.AbstractScalar,
    direction: na.AbstractScalar,
    outputs: na.AbstractScalar,
) -> None:
    """
    Without an axis of the angle of incidence, a direction which differs
    anywhere is refused, by the shape as well as by the interpolation.
    """
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=direction,
        ),
        outputs=outputs,
    )
    match = "measured at more than one direction"
    with pytest.raises(ValueError, match=match):
        optika._util._shape_efficiency_measured(measurement)
    with pytest.raises(ValueError, match=match):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )


@pytest.mark.parametrize(
    argnames="wavelength,direction,outputs,axis_angle",
    argvalues=[
        # only the wavelengths vary along the channel, as a measurement whose
        # samples moved between channels while its values did not
        (
            _wavelength_channel,
            4 * u.deg,
            _linear(_wavelength_measured, 0 * u.deg),
            None,
        ),
        # only the angles of incidence vary along the channel, which reaches
        # the efficiency through the weights of the angles
        (
            _wavelength_measured,
            _angle_measured + na.ScalarArray([0, 2] * u.deg, axes="channel"),
            _linear(_wavelength_measured, 0 * u.deg),
            "angle",
        ),
    ],
)
def test_shape_efficiency_measured_agrees_with_result(
    wavelength: na.AbstractScalar,
    direction: na.AbstractScalar | u.Quantity,
    outputs: na.AbstractScalar,
    axis_angle: None | str,
) -> None:
    """The shape of a measurement is the shape its efficiency comes out in."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=direction,
        ),
        outputs=outputs,
    )
    kwargs = dict(axis_angle=axis_angle, axis_wavelength="wavelength")
    result = optika._util._shape_efficiency_measured(measurement, **kwargs)
    rays = _rays(_angle_rays)
    efficiency = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=rays,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        **kwargs,
    )
    axes_rays = na.shape(rays)
    shape = {ax: n for ax, n in efficiency.shape.items() if ax not in axes_rays}
    assert result == dict(channel=2)
    assert shape == result


@pytest.mark.parametrize(
    argnames="wavelength,direction,axis_angle,axis_wavelength,match",
    argvalues=[
        # an axis the wavelengths do not have is refused by the shape too,
        # rather than leaving the real one in it
        (
            _wavelength_measured,
            4 * u.deg,
            None,
            "other",
            "have no axis 'other'",
        ),
        (
            _wavelength_measured,
            _angle_measured,
            "angle",
            "other",
            "have no axis 'other'",
        ),
        (
            _wavelength_measured + 0 * _angle_measured.value * u.nm,
            _angle_measured,
            "angle",
            "angle",
            "cannot both vary along 'angle'",
        ),
        # a misspelled axis of the angles is named, rather than blamed on the
        # wavelengths, which vary along it
        (
            _wavelength_measured + 0 * _angle_measured.value * u.nm,
            _angle_measured,
            "angel",
            None,
            "have no axis 'angel'",
        ),
        # wavelengths with nothing to interpolate along
        (
            150 * u.nm,
            4 * u.deg,
            None,
            None,
            "nothing to interpolate along",
        ),
        # angles which vary along the axis the wavelength is interpolated
        # along
        (
            _wavelength_measured,
            _angle_measured + na.linspace(0, 1, axis="wavelength", num=11) * u.deg,
            "angle",
            None,
            "cannot vary along the axis the wavelength is interpolated along",
        ),
    ],
)
def test_efficiency_measured_error_axes(
    wavelength: na.AbstractScalar | u.Quantity,
    direction: na.AbstractScalar | u.Quantity,
    axis_angle: None | str,
    axis_wavelength: None | str,
    match: str,
) -> None:
    """The shape and the interpolation refuse the same measurements alike."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=direction,
        ),
        outputs=0.5 + 0 * na.as_named_array(wavelength).value,
    )
    kwargs = dict(axis_angle=axis_angle, axis_wavelength=axis_wavelength)
    with pytest.raises(ValueError, match=match):
        optika._util._shape_efficiency_measured(measurement, **kwargs)
    with pytest.raises(ValueError, match=match):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            **kwargs,
        )


def test_interp_efficiency_measured_uncertain_wavelength() -> None:
    """
    Uncertain wavelengths vary along a single axis, which is interpolated
    along without being named, for each sample of the wavelengths.
    """
    wavelength = na.NormalUncertainScalarArray(
        nominal=_wavelength_measured,
        width=1 * u.nm,
        num_distribution=3,
        seed=1,
    )
    outputs = _linear(_wavelength_measured, 0 * u.deg)
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=4 * u.deg,
        ),
        outputs=outputs,
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )
    x = 150 * u.nm
    expected = na.UncertainScalarArray(
        nominal=na.interp(x, wavelength.nominal, outputs, axis="wavelength"),
        distribution=interp_each(
            x=x,
            xp=wavelength.distribution,
            fp=outputs,
            axis="wavelength",
            axis_each=wavelength.axis_distribution,
        ),
    )
    assert isinstance(result, na.AbstractUncertainScalarArray)
    assert np.all(np.abs(result - expected) < 1e-12)
    # the samples of the wavelengths move the interpolated efficiency
    assert not np.allclose(result.distribution, result.nominal)


def test_interp_efficiency_measured_broadcast_angles() -> None:
    """
    Angles which are repeated along the interpolated axis, as in a table of
    wavelength against angle, are as good as angles without it.
    """
    outputs = _linear(_wavelength_measured, _angle_measured)
    shape = na.broadcast_shapes(_wavelength_measured.shape, _angle_measured.shape)
    kwargs = dict(
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        axis_angle="angle",
    )
    measurements = [
        na.FunctionArray(
            inputs=na.SpectralDirectionalVectorArray(
                wavelength=_wavelength_measured,
                direction=direction,
            ),
            outputs=outputs,
        )
        for direction in (_angle_measured, na.broadcast_to(_angle_measured, shape))
    ]
    shapes = [
        optika._util._shape_efficiency_measured(m, axis_angle="angle")
        for m in measurements
    ]
    results = [
        optika._util._interp_efficiency_measured(measurement=m, **kwargs)
        for m in measurements
    ]
    assert shapes[0] == shapes[1] == {}
    assert results[0].shape == results[1].shape
    assert np.all(results[0] == results[1])
