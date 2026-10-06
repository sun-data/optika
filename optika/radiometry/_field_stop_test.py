import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


def _wavelength() -> na.AbstractScalar:
    return na.linspace(500, 600, axis="wavelength", num=3) * u.nm


def _scene() -> na.SpectralPositionalVectorArray:
    return na.SpectralPositionalVectorArray(
        wavelength=na.linspace(480, 620, axis="wavelength", num=5) * u.nm,
        position=na.Cartesian2dVectorLinearSpace(
            start=-1 * u.deg,
            stop=+1 * u.deg,
            axis=na.Cartesian2dVectorArray("field_x", "field_y"),
            num=5,
        ),
    )


_drift = 0.002 * u.deg / u.nm
"""
How fast the drifting field of view below moves across the scene: slowly
enough that the center of the scene stays inside it at every wavelength.
"""


def _vertices(drift: u.Quantity) -> na.Cartesian2dVectorArray:
    """
    The corners of a square field of view half a degree across, drifting
    along :math:`x` with wavelength at the given rate.
    """
    angle = na.linspace(45, 405, axis="vertex", num=5) * u.deg
    shift = (_wavelength() - 550 * u.nm) * drift
    return na.Cartesian2dVectorArray(
        x=0.5 * u.deg * np.cos(angle) + shift,
        y=0.5 * u.deg * np.sin(angle) + 0 * shift,
    )


class AbstractTestAbstractFieldStopModel(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test__call__(self, a: optika.radiometry.AbstractFieldStopModel):
        scene = _scene()
        result = a(scene)
        assert isinstance(result, na.AbstractScalar)
        for ax in ("field_x", "field_y"):
            assert ax in na.shape(result)

        # the center of the field is inside the field of view at every
        # wavelength, and the corners of the scene are not
        center = scene.position[dict(field_x=2, field_y=2)]
        corner = scene.position[dict(field_x=0, field_y=0)]
        inside = a(na.SpectralPositionalVectorArray(scene.wavelength, center))
        outside = a(na.SpectralPositionalVectorArray(scene.wavelength, corner))
        assert np.all(inside)
        assert not np.any(outside)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            550 * u.nm,
            na.linspace(480, 620, axis="wavelength", num=5) * u.nm,
        ],
    )
    def test_wire(
        self,
        a: optika.radiometry.AbstractFieldStopModel,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.wire(wavelength)
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert "wire" in na.shape(result)
        assert "vertex" not in na.shape(result)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.radiometry.ApertureFieldStopModel(
            aperture=optika.apertures.CircularAperture(0.5 * u.deg),
        ),
    ],
)
class TestApertureFieldStopModel(
    AbstractTestAbstractFieldStopModel,
):
    def test_wire_ignores_the_wavelength(
        self,
        a: optika.radiometry.ApertureFieldStopModel,
    ):
        result = a.wire(na.linspace(480, 620, axis="wavelength", num=5) * u.nm)
        assert "wavelength" not in na.shape(result)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.radiometry.PolynomialFieldStopModel(
            wavelength=_wavelength(),
            vertices=_vertices(drift),
            axis_wavelength="wavelength",
            degree=degree,
        )
        for drift in [0 * _drift, _drift]
        for degree in [1, 2, 5]
    ],
)
class TestPolynomialFieldStopModel(
    AbstractTestAbstractFieldStopModel,
):
    def test_shape(self, a: optika.radiometry.PolynomialFieldStopModel):
        super().test_shape(a)
        assert a.axis_wavelength not in a.shape
        assert "vertex" not in a.shape

    def test_fit(self, a: optika.radiometry.PolynomialFieldStopModel):
        assert isinstance(a.fit, na.PolynomialFitFunctionArray)

    def test_polygon(self, a: optika.radiometry.PolynomialFieldStopModel):
        # at the wavelengths it was given, the outline is the one it was given
        wavelength = a.wavelength
        result = a.polygon(wavelength)
        assert isinstance(result, optika.apertures.PolygonalAperture)
        assert np.allclose(result.vertices.x, a.vertices.x, rtol=0, atol=1e-9 * u.deg)
        assert np.allclose(result.vertices.y, a.vertices.y, rtol=0, atol=1e-9 * u.deg)

    def test_wire_follows_the_wavelength(
        self,
        a: optika.radiometry.PolynomialFieldStopModel,
    ):
        # outlined at a single wavelength, there is one outline, where the
        # drift puts it: the scene wavelengths are not the ones it was given
        for wavelength in [480 * u.nm, 575 * u.nm, 620 * u.nm]:
            wire = a.wire(wavelength)
            assert "wavelength" not in na.shape(wire)
            center = (wire.x.max("wire") + wire.x.min("wire")) / 2
            drift = a.vertices.x.ptp(a.axis_wavelength).max() / (100 * u.nm)
            expected = (wavelength - 550 * u.nm) * drift
            assert np.allclose(center, expected, rtol=0, atol=1e-9 * u.deg)


def test_field_stop_model_follows_a_field_of_view_across_the_scene():
    """
    A point of the scene lies inside a drifting field of view at the
    wavelengths the field of view has drifted over it, and outside at the
    others.
    """
    drift = 0.01 * u.deg / u.nm
    model = optika.radiometry.PolynomialFieldStopModel(
        wavelength=_wavelength(),
        vertices=_vertices(drift),
        axis_wavelength="wavelength",
    )
    wavelength = na.linspace(460, 640, axis="wavelength", num=10) * u.nm
    point = na.Cartesian2dVectorArray(0.6, 0) * u.deg

    result = model(na.SpectralPositionalVectorArray(wavelength, point))

    # the square's edge sits at 0.5 * cos(45 deg) = 0.354 deg from its
    # center, so the point is inside once the drift passes 0.246 deg
    expected = (wavelength - 550 * u.nm) * drift > (0.6 - 0.5 / np.sqrt(2)) * u.deg
    assert np.all(result == expected)
