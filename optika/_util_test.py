import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


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


@pytest.mark.parametrize(
    argnames="axis_angle,shape_expected",
    argvalues=[
        (None, dict(angle=3, channel=2)),
        ("angle", dict(channel=2)),
    ],
)
def test_shape_efficiency_measured(
    axis_angle: None | str,
    shape_expected: dict[str, int],
):
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
        axis_angle=axis_angle,
    )
    assert result == shape_expected


@pytest.mark.parametrize(
    argnames="wavelength,direction,axis_angle,error",
    argvalues=[
        # a single angle needs one-dimensional wavelengths
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            4 * u.deg,
            None,
            ValueError,
        ),
        # more than one angle needs the axis they vary along
        (
            _wavelength_measured,
            _angle_measured,
            None,
            ValueError,
        ),
        # more than one angle must be given as angles of incidence
        (
            _wavelength_measured,
            na.Cartesian3dVectorArray(0, 0, na.linspace(1, 2, axis="angle", num=3)),
            "angle",
            TypeError,
        ),
        (
            _wavelength_measured,
            _angle_measured,
            "other",
            ValueError,
        ),
        (
            _wavelength_measured,
            na.ScalarArray([10, 14, 18] * u.m, axes="angle"),
            "angle",
            ValueError,
        ),
        (
            _wavelength_measured,
            na.ScalarArray([18, 14, 10] * u.deg, axes="angle"),
            "angle",
            ValueError,
        ),
        # the wavelengths may only vary along the angle axis
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            _angle_measured,
            "angle",
            ValueError,
        ),
    ],
)
def test_interp_efficiency_measured_error(
    wavelength: na.AbstractScalar,
    direction,
    axis_angle: None | str,
    error: type[Exception],
):
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=direction,
        ),
        outputs=0.5 + 0 * wavelength.value,
    )
    with pytest.raises(error):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            axis_angle=axis_angle,
        )


_wavelength_channel = _wavelength_measured + na.ScalarArray(
    [0, 3] * u.nm, axes="channel"
)
"""Wavelength samples which differ for each of two channels."""

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


@pytest.mark.parametrize(
    argnames="axis_angle,shape_expected",
    argvalues=[
        (None, dict(angle=3, channel=2)),
        ("angle", dict(channel=2)),
    ],
)
def test_shape_efficiency_measured_axis_wavelength(
    axis_angle: None | str,
    shape_expected: dict[str, int],
) -> None:
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
        axis_angle=axis_angle,
        axis_wavelength="wavelength",
    )
    assert result == shape_expected


@pytest.mark.parametrize(
    argnames="direction,axis_angle",
    argvalues=[
        (4 * u.deg, None),
        (_angle_measured, "angle"),
    ],
)
def test_interp_efficiency_measured_axis_wavelength_missing(
    direction: na.AbstractScalar | u.Quantity,
    axis_angle: None | str,
) -> None:
    """An axis the wavelengths do not have is refused."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_channel,
            direction=direction,
        ),
        outputs=0.5 + 0 * _wavelength_channel.value,
    )
    with pytest.raises(ValueError, match="have no axis 'other'"):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            axis_angle=axis_angle,
            axis_wavelength="other",
        )


def _interp_numpy(
    wavelength: na.AbstractScalar,
    efficiency: na.AbstractScalar,
    x: u.Quantity,
) -> np.ndarray:
    """
    Interpolate each channel of a measurement on its own, with numpy, as a
    reference which does not go through named-arrays.
    """
    shape = na.broadcast_shapes(wavelength.shape, efficiency.shape)
    wavelength = wavelength.broadcast_to(shape)
    efficiency = efficiency.broadcast_to(shape)
    return np.array(
        [
            np.interp(
                x.to_value(u.nm),
                wavelength[dict(channel=i)].ndarray.to_value(u.nm),
                efficiency[dict(channel=i)].ndarray,
            )
            for i in range(shape["channel"])
        ]
    )


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
    expected = _interp_numpy(
        wavelength=measurement.inputs.wavelength,
        efficiency=measurement.outputs,
        x=150 * u.nm,
    )
    assert result.shape == dict(channel=2)
    assert np.allclose(result.ndarray, expected)


def test_interp_efficiency_measured_angle_per_measurement() -> None:
    """Each measurement may have been made at a single angle of its own."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_channel,
            direction=na.ScalarArray([4, 6] * u.deg, axes="channel"),
        ),
        outputs=_linear(_wavelength_channel, 0 * u.deg) + _offset_channel,
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        axis_wavelength="wavelength",
    )
    assert np.allclose(result, _linear(150 * u.nm, 0 * u.deg) + _offset_channel)


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
    efficiency = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        **kwargs,
    )
    assert result == dict(channel=2)
    assert all(efficiency.shape[ax] == n for ax, n in result.items())


@pytest.mark.parametrize(
    argnames="axis_angle,axis_wavelength,match",
    argvalues=[
        # the shape refuses an axis the wavelengths do not have, rather than
        # leaving the real one in it
        (None, "other", "have no axis 'other'"),
        ("angle", "angle", "cannot both vary along 'angle'"),
    ],
)
def test_shape_efficiency_measured_error(
    axis_angle: None | str,
    axis_wavelength: str,
    match: str,
) -> None:
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_measured + 0 * _angle_measured.value * u.nm,
            direction=_angle_measured,
        ),
        outputs=_linear(_wavelength_measured, _angle_measured),
    )
    with pytest.raises(ValueError, match=match):
        optika._util._shape_efficiency_measured(
            measurement=measurement,
            axis_angle=axis_angle,
            axis_wavelength=axis_wavelength,
        )


def test_interp_efficiency_measured_same_axes() -> None:
    """The wavelength and the angle of incidence cannot share an axis."""
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_measured + 0 * _angle_measured.value * u.nm,
            direction=_angle_measured,
        ),
        outputs=_linear(_wavelength_measured, _angle_measured),
    )
    with pytest.raises(ValueError, match="cannot both vary along 'angle'"):
        optika._util._interp_efficiency_measured(
            measurement=measurement,
            rays=_rays(_angle_rays),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            axis_angle="angle",
            axis_wavelength="angle",
        )


def test_interp_efficiency_measured_uncertain_wavelength() -> None:
    """
    Uncertain wavelengths vary along a single axis, which is interpolated
    along without being named.
    """
    wavelength = na.NormalUncertainScalarArray(
        nominal=_wavelength_measured,
        width=0.1 * u.nm,
        num_distribution=3,
        seed=1,
    )
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=4 * u.deg,
        ),
        outputs=_linear(_wavelength_measured, 0 * u.deg),
    )
    kwargs = dict(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
    )
    result = optika._util._interp_efficiency_measured(**kwargs)
    expected = optika._util._interp_efficiency_measured(
        **kwargs,
        axis_wavelength="wavelength",
    )
    assert isinstance(result, na.AbstractUncertainScalarArray)
    assert np.all(result == expected)
