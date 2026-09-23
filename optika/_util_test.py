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
