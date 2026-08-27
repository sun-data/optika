import pytest
import matplotlib.axes
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import test_mixins
from . import test_propagators

surfaces = [
    optika.surfaces.Surface(),
    optika.surfaces.Surface(
        name="test_surface",
        sag=optika.sags.SphericalSag(radius=1000 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.RectangularAperture(half_width=10 * u.mm),
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
    ),
    optika.surfaces.Surface(
        name="test_surface",
        sag=optika.sags.SphericalSag(radius=1000 * u.mm),
        material=optika.materials.Mirror(),
        aperture=optika.apertures.RectangularAperture(half_width=10 * u.mm),
        aperture_mechanical=optika.apertures.RectangularAperture(11 * u.mm),
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
        rulings=optika.rulings.Rulings(spacing=1 * u.um, diffraction_order=1),
    ),
]


class AbstractTestAbstractSurface(
    test_mixins.AbstractTestDxfWritable,
    test_mixins.AbstractTestPlottable,
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
    test_propagators.AbstractTestAbstractLightPropagator,
):
    def test_name(self, a: optika.surfaces.AbstractSurface):
        if a.name is not None:
            assert isinstance(a.name, str)

    def test_sag(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.sag, optika.sags.AbstractSag)

    def test_material(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.material, optika.materials.AbstractMaterial)

    def test_aperture(self, a: optika.surfaces.AbstractSurface):
        if a.aperture is not None:
            assert isinstance(a.aperture, optika.apertures.AbstractAperture)

    def test_aperture_mechanical(self, a: optika.surfaces.AbstractSurface):
        if a.aperture_mechanical is not None:
            assert isinstance(a.aperture_mechanical, optika.apertures.AbstractAperture)

    def test_rulings(self, a: optika.surfaces.AbstractSurface):
        if a.rulings is not None:
            assert isinstance(a.rulings, optika.rulings.AbstractRulings)

    def test_is_field_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_field_stop, bool)

    def test_is_pupil_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_pupil_stop, bool)

    def test_is_stop(self, a: optika.surfaces.AbstractSurface):
        assert isinstance(a.is_stop, bool)

    class TestPlot(
        test_mixins.AbstractTestPlottable.TestPlot,
    ):
        def test_plot(
            self,
            a: optika.surfaces.AbstractSurface,
            ax: None | matplotlib.axes.Axes | na.ScalarArray,
            transformation: None | na.transformations.AbstractTransformation,
        ):
            if a.aperture is not None:
                if na.unit_normalized(a.aperture.wire()).is_equivalent(u.mm):
                    super().test_plot(
                        a=a,
                        ax=ax,
                        transformation=transformation,
                    )


@pytest.mark.parametrize("a", surfaces)
class TestSurface(
    AbstractTestAbstractSurface,
):
    pass


def test_plot_kwargs_plot():
    """A surface can carry the keywords it is to be drawn with."""
    color = "tab:red"
    surface = optika.surfaces.Surface(
        aperture=optika.apertures.CircularAperture(10 * u.mm),
        kwargs_plot=dict(color=color),
    )

    fig, ax = plt.subplots()
    surface.plot(ax=ax, components=("x", "y"))
    colors = [line.get_color() for line in ax.lines]
    plt.close(fig)

    assert colors
    assert all(c == color for c in colors)


_substrate = optika.materials.Mirror(
    substrate=optika.materials.Layer(chemical="SiO2", thickness=20 * u.mm),
)


def _surface(material) -> optika.surfaces.Surface:
    return optika.surfaces.Surface(
        sag=optika.sags.SphericalSag(radius=-400 * u.mm),
        material=material,
        aperture=optika.apertures.RegularPolygonalAperture(
            radius=50 * u.mm,
            num_vertices=8,
        ),
    )


def test_plot_substrate():
    """A surface whose material has a substrate is given a thickness."""
    fig, ax = plt.subplots()
    result = _surface(_substrate).plot(ax=ax, components=("z", "x"))
    plt.close(fig)

    assert "substrate" in result
    assert "back" in result["substrate"]
    assert "edges" in result["substrate"]


def test_plot_no_substrate():
    """A surface without one is drawn exactly as it was before."""
    fig, ax = plt.subplots()
    result = _surface(optika.materials.Mirror()).plot(ax=ax, components=("z", "x"))
    plt.close(fig)

    assert "substrate" not in result


def test_plot_substrate_behind_the_surface():
    """
    The substrate lies on the far side of the surface from its normal.

    A fixed sign would be right for a surface facing one way and wrong for one
    facing the other, so the side is taken from the surface itself.
    """
    surface = _surface(_substrate)
    thickness = surface.material.substrate.thickness
    normal = surface.sag.normal(na.Cartesian3dVectorArray() * u.mm)

    fig, ax = plt.subplots()
    result = surface.plot(ax=ax, components=("z", "x"))

    lines = np.atleast_1d(result["substrate"]["back"].ndarray)
    drawn = np.concatenate([np.asarray(a.get_xdata()) for a in lines.flat])
    plt.close(fig)

    # the back of the substrate is a thickness away, on the side the normal
    # points away from
    assert drawn == pytest.approx(-np.sign(na.value(normal.z)) * na.value(thickness))


def test_plot_substrate_uses_the_aperture_transformation():
    """
    The wall of the substrate is built where the faces of it are.

    An aperture can be placed within its surface. `wire` accounts for that and
    `vertices` does not, so a wall built from the raw vertices is drawn beside
    the optic instead of on it.
    """
    offset = 17 * u.mm
    surface = optika.surfaces.Surface(
        material=_substrate,
        aperture=optika.apertures.RectangularAperture(
            half_width=10 * u.mm,
            transformation=na.transformations.Cartesian3dTranslation(x=offset),
        ),
    )

    fig, ax = plt.subplots()
    result = surface.plot(ax=ax, components=("x", "z"))

    def extent(artist) -> tuple[float, float]:
        lines = np.atleast_1d(artist.ndarray if hasattr(artist, "ndarray") else artist)
        x = np.concatenate([np.asarray(a.get_xdata()) for a in lines.flat])
        return float(x.min()), float(x.max())

    back = extent(result["substrate"]["back"])
    edges = extent(result["substrate"]["edges"])
    plt.close(fig)

    # the wall meets the back of the substrate rather than sitting beside it
    assert back[0] == pytest.approx(edges[0], abs=1e-9)
    assert back[1] == pytest.approx(edges[1], abs=1e-9)

    # and both are where the aperture was put, not where it was defined
    assert back[0] == pytest.approx(offset.value - 10, abs=1e-9)


def test_plot_substrate_3d():
    """
    Seen in 3D the substrate is a solid: a filled back, and a wall around it.

    The wall is drawn a panel at a time rather than as one artist, since
    matplotlib sorts a 3D axes by one depth per artist, and a wall which
    wraps around the optic is nearer the viewer on one side than the other.
    """
    surface = _surface(_substrate)
    num_vertices = surface.aperture.num_vertices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = surface.plot(ax=ax, components=("x", "y", "z"))["substrate"]
    plt.close(fig)

    assert "wall" in result
    assert "edges" not in result

    for artist in [result["back"], result["wall"]]:
        for a in np.atleast_1d(na.as_named_array(artist).ndarray).flat:
            assert isinstance(a, mpl_toolkits.mplot3d.art3d.Poly3DCollection)

    # one panel of the wall for each edge of the aperture
    assert na.shape(result["wall"]) == dict(vertex=num_vertices)


def test_plot_substrate_unit():
    """The substrate is drawn in the unit asked for, like the rest of the surface."""
    surface = _surface(_substrate)
    thickness = na.value(surface.material.substrate.thickness)

    fig, ax = plt.subplots()
    result = surface.plot(ax=ax, components=("z", "x"), unit=u.um)["substrate"]

    lines = np.atleast_1d(na.as_named_array(result["back"]).ndarray)
    drawn = np.concatenate([np.asarray(a.get_xdata()) for a in lines.flat])
    plt.close(fig)

    # micrometres, so a thousand times the number millimetres would give
    assert np.abs(drawn) == pytest.approx(1000 * thickness)


def test_plot_substrate_without_an_aperture():
    """A surface with no aperture has no edge to hang a substrate on."""
    fig, ax = plt.subplots()
    result = optika.surfaces.Surface(material=_substrate).plot(
        ax=ax,
        components=("z", "x"),
    )
    plt.close(fig)

    assert result["substrate"] == dict()


def test_plot_substrate_without_vertices():
    """
    An aperture with no corners is still given a wall.

    A circle has no vertices to build one from, so the samples along its edge
    are used instead. Without this its substrate is a back face with nothing
    joining it to the front.
    """
    surface = optika.surfaces.Surface(
        sag=optika.sags.SphericalSag(radius=-400 * u.mm),
        material=_substrate,
        aperture=optika.apertures.CircularAperture(50 * u.mm),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    result = surface.plot(ax=ax, components=("x", "y", "z"))["substrate"]
    plt.close(fig)

    assert "wall" in result
    assert na.shape(result["wall"])["vertex"] > 2
