import warnings
import pytest
import numpy as np
import astropy.units as u
import matplotlib.figure
import matplotlib.pyplot as plt
import named_arrays as na
import optika
from .._tests import test_mixins


def _scene() -> na.SpectralPositionalVectorArray:
    """The corners of the cells the calibration points were measured in."""
    return na.SpectralPositionalVectorArray(
        wavelength=na.linspace(500, 600, axis="wavelength", num=3) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-1 * u.deg,
            stop=+1 * u.deg,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=6,
        ),
    )


def _centers() -> na.SpectralPositionalVectorArray:
    """The centers of those cells, where a measurement is made by default."""
    return _scene().cell_centers(("field_x", "field_y"))


def _sample() -> na.SpectralPositionalVectorArray:
    """A point drawn at random inside each cell, rather than at its center."""
    scene = _scene()
    position = scene.position.broadcast_to(na.shape(scene.position))
    return na.SpectralPositionalVectorArray(
        wavelength=scene.wavelength,
        position=position.cell_centers(
            axis=("field_x", "field_y"),
            random=True,
            seed=0,
        ),
    )


def _sensor(
    coordinates: None | na.SpectralPositionalVectorArray = None,
) -> na.Cartesian2dVectorArray:
    """A plate scale, measured wherever the calibration points are."""
    if coordinates is None:
        coordinates = _centers()
    return coordinates.position * (10 * u.mm / u.deg)


class AbstractTestAbstractDistortionModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
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


class AbstractTestAbstractLinearDistortionModel(
    AbstractTestAbstractDistortionModel,
):
    def test_matrix(self, a: optika.distortion.AbstractLinearDistortionModel):
        assert isinstance(a.matrix, na.AbstractSpectralPositionalMatrixArray)

    def test_center(self, a: optika.distortion.AbstractLinearDistortionModel):
        assert isinstance(a.center, na.AbstractSpectralPositionalVectorArray)

    def test_intercept(self, a: optika.distortion.AbstractLinearDistortionModel):
        assert isinstance(a.intercept, na.AbstractSpectralPositionalVectorArray)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.distortion.SimpleDistortionModel(
            plate_scale=1 * u.arcsec / u.pix,
            dispersion=0.1 * u.nm / u.pix,
            angle=angle,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(0, 0) * u.pix,
            ),
        )
        for angle in [0 * u.deg, 15 * u.deg]
    ],
)
class TestSimpleDistortionModel(
    AbstractTestAbstractLinearDistortionModel,
):
    pass


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
            coordinates_sample=sample,
            coordinates_sensor=_sensor(sample),
            axis_wavelength="wavelength",
            axis_field=("field_x", "field_y"),
            degree=degree,
        )
        for degree in [1, 2]
        for sample in [None, _sample()]
    ],
)
class TestPolynomialDistortionModel(
    AbstractTestAbstractInterpolatedDistortionModel,
):
    def test_fit(self, a: optika.distortion.PolynomialDistortionModel):
        assert isinstance(a.fit, na.PolynomialFitFunctionArray)
        assert a.fit.coefficient_names is not None

    def test_fit_inverse(self, a: optika.distortion.PolynomialDistortionModel):
        assert isinstance(a.fit_inverse, na.PolynomialFitFunctionArray)
        assert a.fit_inverse.coefficient_names is not None

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

    def test_plot_residual_ax(
        self,
        a: optika.distortion.PolynomialDistortionModel,
    ):
        """The plotter draws into axes given to it, instead of its own."""
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

        fig_result, ax_result = a.plot_residual(ax=row)

        assert fig_result is fig
        assert np.all(ax_result == row)

        # the row it was given has been drawn on, and the other has not
        assert all(b.collections for b in row.ndarray)
        assert not any(b.collections for b in ax[{"row": 1}].ndarray)

        plt.close(fig)

    def test_plot_residual_unit(
        self,
        a: optika.distortion.PolynomialDistortionModel,
    ):
        """The field position is drawn in the unit asked for, labels and all."""
        fig_deg, ax_deg = a.plot_residual()
        fig_arcsec, ax_arcsec = a.plot_residual(unit=u.arcsec)

        axs_deg = ax_deg.ndarray.reshape(-1)[0]
        axs_arcsec = ax_arcsec.ndarray.reshape(-1)[0]

        scale = u.deg.to(u.arcsec)
        for get in ("get_xlim", "get_ylim"):
            lim_deg = getattr(axs_deg, get)()
            lim_arcsec = getattr(axs_arcsec, get)()
            assert lim_arcsec == pytest.approx(tuple(scale * x for x in lim_deg))

        assert format(u.arcsec, "latex_inline") in axs_arcsec.get_xlabel()
        assert format(u.deg, "latex_inline") in axs_deg.get_xlabel()

        plt.close(fig_deg)
        plt.close(fig_arcsec)

    def test_plot_residual_ax_invalid(
        self,
        a: optika.distortion.PolynomialDistortionModel,
    ):
        """Axes which are not distributed along the wavelength axis are refused."""
        fig, ax = na.plt.subplots(axis_cols="wrong", ncols=2, squeeze=False)

        with pytest.raises(ValueError, match="must be distributed along"):
            a.plot_residual(ax=ax)

        plt.close(fig)


def test_polynomial_distortion_model_channel():
    """
    Calibration points that vary along an axis orthogonal to the scene axes
    (e.g. the channel axis of a multi-channel instrument) must be fit with an
    independent polynomial per channel, not one polynomial averaged over all
    the channels.
    """
    scene = _scene()
    centers = _centers()

    scale = na.ScalarArray([10, 12, 8] * u.mm / u.deg, axes="channel")
    angle = na.ScalarArray([0, 10, -15] * u.deg, axes="channel")

    cos, sin = np.cos(angle), np.sin(angle)
    sensor = na.Cartesian2dVectorArray(
        x=scale * (cos * centers.position.x - sin * centers.position.y),
        y=scale * (sin * centers.position.x + cos * centers.position.y),
    )

    a = optika.distortion.PolynomialDistortionModel(
        coordinates_scene=scene,
        coordinates_sensor=sensor,
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=1,
    )

    distorted = a.distort(centers).position
    assert "channel" in distorted.shape
    assert np.all((distorted - sensor).length < 1e-9 * u.mm)

    undistorted = a.undistort(
        na.SpectralPositionalVectorArray(
            wavelength=centers.wavelength,
            position=sensor,
        )
    ).position
    assert "channel" in undistorted.shape
    assert np.all((undistorted - centers.position).length < 1e-9 * u.deg)


def test_plot_residual_draws_the_cells_and_not_the_samples():
    """
    A model measured at points drawn inside its cells still has its residual
    plotted on the cells.

    Such points are not monotonic, and matplotlib cannot work out where one
    cell ends and the next begins from points alone: it says as much, and
    draws a mesh with warped cells and a ragged outline. The mesh is
    `coordinates_scene`, which is the corners, so this holds however the
    measurements inside them are placed.
    """
    scene = _scene()
    sample = _sample()

    a = optika.distortion.PolynomialDistortionModel(
        coordinates_scene=scene,
        coordinates_sample=sample,
        coordinates_sensor=_sensor(sample),
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig, ax = a.plot_residual()

    # the mesh is rectilinear: its corners take one value per column of the
    # grid, rather than one per measurement
    coordinates = ax.ndarray.reshape(-1)[0].collections[0].get_coordinates()
    assert np.unique(coordinates[..., 0]).size == na.shape(scene.position)["field_x"]
    assert np.unique(coordinates[..., 1]).size == na.shape(scene.position)["field_y"]

    plt.close(fig)


def test_coordinates_sample_defaults_to_the_cell_centers():
    """A model which does not say where it was measured was measured at the
    centers of its cells."""
    a = optika.distortion.PolynomialDistortionModel(
        coordinates_scene=_scene(),
        coordinates_sensor=_sensor(),
        axis_wavelength="wavelength",
        axis_field=("field_x", "field_y"),
        degree=1,
    )

    assert np.all(a.coordinates_sample_.position == _centers().position)
