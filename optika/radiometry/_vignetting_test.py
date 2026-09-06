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


def _illumination() -> na.AbstractScalar:
    return 1 - 0.1 * (_scene().position.length / u.deg) ** 2


class AbstractTestAbstractVignettingModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test__call__(self, a: optika.radiometry.AbstractVignettingModel):
        scene = _scene()
        result = a(scene)
        assert isinstance(result, na.AbstractScalar)
        for ax in ("field_x", "field_y"):
            assert ax in na.shape(result)

    def test_inverse(self, a: optika.radiometry.AbstractVignettingModel):
        scene = _scene()
        result = a.inverse(scene)
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result == 1 / a(scene))


class AbstractTestAbstractInterpolatedVignettingModel(
    AbstractTestAbstractVignettingModel,
):
    def test_coordinates_scene(
        self,
        a: optika.radiometry.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.coordinates_scene, na.AbstractSpectralPositionalVectorArray)

    def test_illumination(
        self,
        a: optika.radiometry.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.illumination, na.AbstractScalar)

    def test_axis_wavelength(
        self,
        a: optika.radiometry.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.axis_wavelength, str)

    def test_axis_field(
        self,
        a: optika.radiometry.AbstractInterpolatedVignettingModel,
    ):
        assert isinstance(a.axis_field, tuple)
        assert all(isinstance(ax, str) for ax in a.axis_field)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.radiometry.PolynomialVignettingModel(
            coordinates_scene=_scene(),
            illumination=_illumination(),
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
    def test_fit(self, a: optika.radiometry.PolynomialVignettingModel):
        assert isinstance(a.fit, na.PolynomialFitFunctionArray)
        assert a.fit.coefficient_names is not None

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(),
            dict(figsize=(8, 4), cmap="viridis", vmin=0, vmax=0.01),
        ],
    )
    def test_plot(
        self,
        a: optika.radiometry.PolynomialVignettingModel,
        kwargs: dict,
    ):
        fig, ax = a.plot(**kwargs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, na.ScalarArray)
        assert a.axis_wavelength in na.shape(ax)
        plt.close(fig)

    @pytest.mark.parametrize(
        argnames="kwargs",
        argvalues=[
            dict(),
            dict(figsize=(8, 4), cmap="viridis", vmin=0, vmax=0.01),
        ],
    )
    def test_plot_residual(
        self,
        a: optika.radiometry.PolynomialVignettingModel,
        kwargs: dict,
    ):
        fig, ax = a.plot_residual(**kwargs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, na.ScalarArray)
        assert a.axis_wavelength in na.shape(ax)
        plt.close(fig)

    @pytest.mark.parametrize(
        argnames="method",
        argvalues=["plot", "plot_residual"],
    )
    def test_plot_ax(
        self,
        a: optika.radiometry.PolynomialVignettingModel,
        method: str,
    ):
        """Both plotters draw into axes given to them, instead of their own."""
        axis = a.axis_wavelength
        num = na.shape(a.coordinates_scene)[axis]

        fig, ax = na.plt.subplots(
            axis_rows="row",
            nrows=2,
            axis_cols=axis,
            ncols=num,
            squeeze=False,
        )

        row = ax[{"row": 0}]

        fig_result, ax_result = getattr(a, method)(ax=row)

        assert fig_result is fig
        assert np.all(ax_result == row)

        # the row it was given has been drawn on, and the other has not
        assert all(b.collections for b in row.ndarray)
        assert not any(b.collections for b in ax[{"row": 1}].ndarray)

        plt.close(fig)

    @pytest.mark.parametrize(
        argnames="method",
        argvalues=["plot", "plot_residual"],
    )
    def test_plot_ax_invalid(
        self,
        a: optika.radiometry.PolynomialVignettingModel,
        method: str,
    ):
        """Axes which are not distributed along the wavelength axis are refused."""
        fig, ax = na.plt.subplots(axis_cols="wrong", ncols=2, squeeze=False)

        with pytest.raises(ValueError, match="must be distributed along"):
            getattr(a, method)(ax=ax)

        plt.close(fig)


def test_polynomial_vignetting_model_channel():
    """
    Calibration points that vary along an axis orthogonal to the scene axes
    (e.g. the channel axis of a multi-channel instrument) must be fit with an
    independent polynomial per channel, not one polynomial averaged over all
    the channels.
    """
    scene = _scene()

    coefficient = na.ScalarArray([0.1, 0.2, 0.05] / u.deg**2, axes="channel")
    illumination = 1 - coefficient * scene.position.length**2

    a = optika.radiometry.PolynomialVignettingModel(
        coordinates_scene=scene,
        illumination=illumination,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=2,
    )

    result = a(scene)
    assert "channel" in result.shape
    assert np.all(np.abs(result - illumination) < 1e-9)


def test_plot_residual_where():
    """
    The residual is undefined at the calibration points the fit was not
    constrained by, so those are left out rather than drawn, and the default
    color scale is set without them.
    """
    scene = _scene()
    illumination = _illumination()

    # the corners of the field are excluded from the fit
    where = scene.position.length < 1.2 * u.deg

    a = optika.radiometry.PolynomialVignettingModel(
        coordinates_scene=scene,
        illumination=illumination,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=1,
        where=where,
    )

    residual = abs(a.illumination - a.fit.predictions)
    residual_inside = np.nanmax(residual[where].ndarray)

    # the excluded corners hold the largest residuals, so had they been kept
    # they would have set the upper limit of the color scale
    assert residual_inside < np.nanmax(residual.ndarray)

    fig, ax = a.plot_residual()

    norm = ax.ndarray.reshape(-1)[0].collections[0].norm
    assert norm.vmax == pytest.approx(residual_inside)

    plt.close(fig)
