import pytest
import numpy as np
import astropy.units as u
import matplotlib.figure
import matplotlib.pyplot as plt
import named_arrays as na
import optika
from .._tests import test_mixins


def _scene() -> na.SpectralPositionalVectorArray:
    return na.SpectralPositionalVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
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
        assert isinstance(result, na.SpectralPositionalVectorArray)
        assert np.all(result.wavelength == coordinates.wavelength)

    def test_roundtrip(self, a: optika.distortion.AbstractDistortionModel):
        scene = _scene()
        result = a.undistort(a.distort(scene))
        error = (result.position - scene.position).length
        assert np.all(error < 1e-9 * u.deg)


class AbstractTestAbstractInterpolatedDistortionModel(
    AbstractTestAbstractDistortionModel,
):
    def test_coordinates_scene(
        self,
        a: optika.distortion.AbstractInterpolatedDistortionModel,
    ):
        assert isinstance(a.coordinates_scene, na.AbstractSpectralPositionalVectorArray)

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
                x=_scene().position.x * (10 * u.mm / u.deg),
                y=_scene().position.y * (10 * u.mm / u.deg),
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

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(),
            dict(figsize=(8, 4), cmap="viridis", vmin=0 * u.um, vmax=5 * u.um),
        ],
    )
    def test_plot_residual(
        self,
        a: optika.distortion.PolynomialDistortionModel,
        kwargs: dict,
    ):
        fig, ax = a.plot_residual(**kwargs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, na.ScalarArray)
        assert a.axis_wavelength in na.shape(ax)
        plt.close(fig)
