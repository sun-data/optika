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
    argnames="normal",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, -1),
        na.Cartesian3dVectorArray(0, 0, 2),
    ],
)
def test_interp_efficiency_measured(
    wavelength: na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
):
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=wavelength,
            direction=_angle_measured,
        ),
        outputs=_linear(wavelength, _angle_measured),
    )
    result = optika._util._interp_efficiency_measured(
        measurement=measurement,
        rays=_rays(_angle_rays),
        normal=normal,
    )

    # interpolated linearly inside the measured angles, and held at the
    # nearest measurement outside them
    angle = np.clip(
        _angle_rays,
        _angle_measured[{"angle": 0}],
        _angle_measured[{"angle": ~0}],
    )
    result_expected = _linear(150 * u.nm, angle)

    assert result.axes == ("ray",)
    assert np.allclose(result, result_expected)


def test_shape_efficiency_measured():
    measurement = na.FunctionArray(
        inputs=na.SpectralDirectionalVectorArray(
            wavelength=_wavelength_measured,
            direction=_angle_measured,
        ),
        outputs=_linear(_wavelength_measured, _angle_measured)
        + na.linspace(0, 0.1, axis="channel", num=2),
    )
    result = optika._util._shape_efficiency_measured(measurement)
    assert result == dict(channel=2)


@pytest.mark.parametrize(
    argnames="wavelength,direction,error",
    argvalues=[
        # a single angle needs one-dimensional wavelengths
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            4 * u.deg,
            ValueError,
        ),
        # more than one direction must be given as angles of incidence
        (
            _wavelength_measured,
            na.Cartesian3dVectorArray(0, 0, na.linspace(1, 2, axis="angle", num=3)),
            TypeError,
        ),
        (
            _wavelength_measured,
            na.ScalarArray([[10, 14], [16, 18]] * u.deg, axes=("angle", "other")),
            ValueError,
        ),
        (
            _wavelength_measured,
            na.ScalarArray([10, 14, 18] * u.m, axes="angle"),
            ValueError,
        ),
        (
            _wavelength_measured,
            na.ScalarArray([18, 14, 10] * u.deg, axes="angle"),
            ValueError,
        ),
        # the wavelengths may only vary along the angle axis
        (
            _wavelength_measured + na.linspace(0, 1, axis="other", num=2) * u.nm,
            _angle_measured,
            ValueError,
        ),
    ],
)
def test_interp_efficiency_measured_error(
    wavelength: na.AbstractScalar,
    direction,
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
        )
