import pytest
import numpy as np
import astropy.units as u
import matplotlib.figure
import matplotlib.pyplot as plt
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


def _transmission() -> na.AbstractScalar:
    return 1 - 0.1 * (_scene().field.length / u.deg) ** 2


class AbstractTestAbstractVignettingModel(
    test_mixins.AbstractTestPrintable,
):
    def test__call__(self, a: optika.vignetting.AbstractVignettingModel):
        scene = _scene()
        result = a(scene)
        assert isinstance(result, na.AbstractScalar)
        for ax in ("field_x", "field_y"):
            assert ax in na.shape(result)

    def test_inverse(self, a: optika.vignetting.AbstractVignettingModel):
        scene = _scene()
        result = a.inverse(scene)
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result == 1 / a(scene))


class AbstractTestAbstractInterpolatedVignettingModel(
    AbstractTestAbstractVignettingModel,
):
    def test_coordinates_scene(
        self,
        a: optika.vignetting.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.coordinates_scene, optika.vectors.SceneVectorArray)

    def test_transmission(
        self,
        a: optika.vignetting.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.transmission, na.AbstractScalar)

    def test_axis_wavelength(
        self,
        a: optika.vignetting.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.axis_wavelength, str)

    def test_axis_field(
        self,
        a: optika.vignetting.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.axis_field, tuple)
        assert all(isinstance(ax, str) for ax in a.axis_field)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.vignetting.PolynomialVignettingModel(
            coordinates_scene=_scene(),
            transmission=_transmission(),
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=degree,
        )
        for degree in [1, 2]
    ],
)
class TestPolynomialVignettingModel(
    AbstractTestAbstractInterpolatedVignettingModel,
):
    def test_fit(self, a: optika.vignetting.PolynomialVignettingModel):
        assert isinstance(a.fit, na.PolynomialFitFunctionArray)
        assert a.fit.degree == a.degree

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(),
            dict(figsize=(8, 4), cmap="viridis", vmin=0, vmax=0.01),
        ],
    )
    def test_plot_residual(
        self,
        a: optika.vignetting.PolynomialVignettingModel,
        kwargs: dict,
    ):
        fig, ax = a.plot_residual(**kwargs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, na.ScalarArray)
        assert a.axis_wavelength in na.shape(ax)
        plt.close(fig)
