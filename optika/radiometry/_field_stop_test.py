import tracemalloc
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


def _vertices(
    drift: u.Quantity,
    radius: u.Quantity | na.AbstractScalar = 0.5 * u.deg,
) -> na.Cartesian2dVectorArray:
    """
    The corners of a square field of view drifting along :math:`x` with
    wavelength at the given rate.

    Parameters
    ----------
    drift
        How fast the field of view moves with wavelength.
    radius
        The distance from the center of the square to its corners.
    """
    angle = na.linspace(45, 405, axis="vertex", num=5) * u.deg
    shift = (_wavelength() - 550 * u.nm) * drift
    return na.Cartesian2dVectorArray(
        x=radius * np.cos(angle) + shift,
        y=radius * np.sin(angle) + 0 * shift,
    )


_radius_uncertain = na.UniformUncertainScalarArray(0.5 * u.deg, 0.01 * u.deg)
"""A field of view whose size is uncertain by a fiftieth of itself."""


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
        optika.radiometry.ApertureFieldStopModel(
            aperture=optika.apertures.CircularAperture(_radius_uncertain),
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
    ]
    + [
        optika.radiometry.PolynomialFieldStopModel(
            wavelength=_wavelength(),
            vertices=_vertices(drift, radius=_radius_uncertain),
            axis_wavelength="wavelength",
        )
        for drift in [0 * _drift, _drift]
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


def test_polynomial_field_stop_does_not_copy_the_vertices_for_every_point() -> None:
    """
    A field of view which moves with wavelength is a different polygon at
    each wavelength, and testing a scene against it takes memory in
    proportion to the points of the scene, not to the points times the
    vertices, even with the scene in a different unit than the vertices.
    """
    wavelength = na.linspace(500, 600, axis="wavelength", num=3) * u.nm
    corners = na.linspace(0, 360, axis="vertex", num=81) * u.deg
    drift = (wavelength - 550 * u.nm) * (0.01 * u.deg / u.nm)
    model = optika.radiometry.PolynomialFieldStopModel(
        wavelength=wavelength,
        vertices=na.Cartesian2dVectorArray(
            x=0.5 * u.deg * np.cos(corners) + drift,
            y=0.5 * u.deg * np.sin(corners),
        ),
        axis_wavelength="wavelength",
    )
    coordinates = na.SpectralPositionalVectorArray(
        wavelength=na.linspace(480, 620, axis="scene_wavelength", num=3) * u.nm,
        position=na.Cartesian2dVectorArray(
            x=na.linspace(-3600, 3600, axis="scene_x", num=200) * u.arcsec,
            y=na.linspace(-3600, 3600, axis="scene_y", num=200) * u.arcsec,
        ),
    )

    # compile the test of the points before measuring
    model(coordinates)

    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        result = model(coordinates)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not tracing:
            tracemalloc.stop()

    # the field of view moves across the scene, so the test is not empty
    assert np.any(result) and not np.all(result)

    # room for four float64 copies of the points, where a copy of the
    # vertices for every point takes two times 81 of them
    assert peak - baseline < 4 * 8 * result.size


def test_polynomial_field_stop_tests_every_point_along_an_axis_of_one_polygon():
    """
    An outline which is one polygon along an axis the points vary along is
    the outline of every point along it, not only of the first.
    """
    model = optika.radiometry.PolynomialFieldStopModel(
        wavelength=_wavelength(),
        vertices=_vertices(_drift) + na.ScalarArray.zeros(dict(channel=1)) * u.deg,
        axis_wavelength="wavelength",
    )
    coordinates = na.SpectralPositionalVectorArray(
        wavelength=550 * u.nm,
        position=na.Cartesian2dVectorArray(
            x=na.linspace(-0.8, 0.8, axis="channel", num=3) * u.deg,
            y=0 * u.deg,
        ),
    )

    result = model(coordinates)

    # the square's edges sit at 0.5 * cos(45 deg) = 0.354 deg from its center
    expected = np.abs(coordinates.position.x) < 0.5 * u.deg / np.sqrt(2)
    assert np.any(expected) and not np.all(expected)
    assert np.all(result == expected)
