import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


def _scene() -> optika.vectors.SceneVectorArray:
    return optika.vectors.SceneVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
        field=na.Cartesian2dVectorLinearSpace(
            start=-1 * u.deg,
            stop=+1 * u.deg,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
        ),
    )


class AbstractTestAbstractDistortionModel(
    test_mixins.AbstractTestPrintable,
):
    def test_distort(self, a: optika.distortion.AbstractDistortionModel):
        coordinates = _scene()
        result = a.distort(coordinates)
        assert isinstance(result, na.SpectralPositionalVectorArray)
        assert isinstance(result.position, na.AbstractCartesian2dVectorArray)
        # the wavelength is carried through unchanged
        assert np.all(result.wavelength == coordinates.wavelength)

    def test_undistort(self, a: optika.distortion.AbstractDistortionModel):
        coordinates = a.distort(_scene())
        result = a.undistort(coordinates)
        assert isinstance(result, optika.vectors.SceneVectorArray)
        assert np.all(result.wavelength == coordinates.wavelength)

    def test_roundtrip(self, a: optika.distortion.AbstractDistortionModel):
        scene = _scene()
        result = a.undistort(a.distort(scene))
        error = (result.field - scene.field).length
        assert np.all(error < 1e-9 * u.deg)


class AbstractTestAbstractInterpolatedDistortionModel(
    AbstractTestAbstractDistortionModel,
):
    def test_coordinates_scene(
        self,
        a: optika.distortion.AbstractInterpolatedDistortionModel,
    ):
        assert isinstance(a.coordinates_scene, optika.vectors.SceneVectorArray)

    def test_coordinates_sensor(
        self,
        a: optika.distortion.AbstractInterpolatedDistortionModel,
    ):
        assert isinstance(a.coordinates_sensor, na.AbstractCartesian2dVectorArray)

    def test_axis_wavelength(
        self,
        a: optika.distortion.AbstractInterpolatedDistortionModel,
    ):
        assert isinstance(a.axis_wavelength, str)

    def test_axis_field(
        self,
        a: optika.distortion.AbstractInterpolatedDistortionModel,
    ):
        assert isinstance(a.axis_field, tuple)
        assert all(isinstance(ax, str) for ax in a.axis_field)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.distortion.PolynomialDistortionModel(
            coordinates_scene=_scene(),
            coordinates_sensor=na.Cartesian2dVectorArray(
                x=_scene().field.x * (10 * u.mm / u.deg),
                y=_scene().field.y * (10 * u.mm / u.deg),
            ),
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=degree,
        )
        for degree in [1, 2]
    ],
)
class TestPolynomialDistortionModel(
    AbstractTestAbstractInterpolatedDistortionModel,
):
    def test_fit(self, a: optika.distortion.PolynomialDistortionModel):
        assert isinstance(a.fit, na.PolynomialFitFunctionArray)
        assert a.fit.degree == a.degree

    def test_fit_inverse(self, a: optika.distortion.PolynomialDistortionModel):
        assert isinstance(a.fit_inverse, na.PolynomialFitFunctionArray)
        assert a.fit_inverse.degree == a.degree
