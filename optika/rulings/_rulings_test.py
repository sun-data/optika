import pytest
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins
from optika.rays._tests import test_ray_vectors

_wavelength = na.linspace(100, 300, axis="wavelength", num=11) * u.AA


class AbstractTestAbstractRulings(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test_diffraction_order(self, a: optika.rulings.AbstractRulings):
        assert np.issubdtype(na.get_dtype(a.diffraction_order), int)

    def test_spacing(self, a: optika.rulings.AbstractRulings):
        result = a.spacing
        types = (u.Quantity, na.AbstractScalar, optika.rulings.AbstractRulingSpacing)
        assert isinstance(result, types)

    def test_spacing_(self, a: optika.rulings.AbstractRulings):
        result = a.spacing_
        assert isinstance(result, optika.rulings.AbstractRulingSpacing)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=test_ray_vectors.rays,
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_rulings(
        self,
        a: optika.rulings.AbstractRulings,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.incident_effective(
            rays=rays,
            normal=normal,
        )

        assert isinstance(result, optika.rays.RayVectorArray)

        assert not np.all(result == rays)

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=test_ray_vectors.rays,
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_efficiency(
        self,
        a: optika.rulings.AbstractRulings,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.efficiency(
            rays=rays,
            normal=normal,
        )

        assert np.all(result >= 0)
        assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.Rulings(
            spacing=1 * u.um,
            diffraction_order=1,
        ),
        optika.rulings.Rulings(
            spacing=1 * u.um,
            diffraction_order=na.ScalarArray(np.array([-1, 0, 1]), axes="m"),
        ),
    ],
)
class TestRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.MeasuredRulings(
            spacing=1 * u.um,
            diffraction_order=1,
            efficiency_measured=na.FunctionArray(
                inputs=na.SpectralDirectionalVectorArray(
                    wavelength=_wavelength,
                    direction=na.Cartesian3dVectorArray(0, 0, 1),
                ),
                outputs=np.exp(-np.square(_wavelength / (10 * u.AA)) / 2),
            ),
        )
    ],
)
class TestMeasuredRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.SinusoidalRulings(
            spacing=1 * u.um,
            depth=10 * u.nm,
            diffraction_order=na.ScalarArray(np.array([0, 1, 2]), axes="m"),
        ),
    ],
)
class TestSinusoidalRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.SquareRulings(
            spacing=1 * u.um,
            depth=10 * u.nm,
            diffraction_order=na.ScalarArray(np.array([0, 1, 2]), axes="m"),
        ),
    ],
)
class TestSquareRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.SawtoothRulings(
            spacing=1 * u.um,
            depth=10 * u.nm,
            diffraction_order=na.ScalarArray(np.array([0, 1, 2]), axes="m"),
        ),
    ],
)
class TestSawtoothRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.TriangularRulings(
            spacing=1 * u.um,
            depth=10 * u.nm,
            diffraction_order=na.ScalarArray(np.array([0, 1, 2]), axes="m"),
        ),
    ],
)
class TestTriangularRulings(
    AbstractTestAbstractRulings,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.rulings.RectangularRulings(
            spacing=1 * u.um,
            depth=10 * u.nm,
            ratio_duty=0.3,
            diffraction_order=na.ScalarArray(np.array([0, 1, 2]), axes="m"),
        ),
    ],
)
class TestRectangularRulings(
    AbstractTestAbstractRulings,
):
    pass


def _rays(
    wavelength: u.Quantity,
    direction: None | na.AbstractCartesian3dVectorArray = None,
) -> optika.rays.RayVectorArray:
    if direction is None:
        direction = na.Cartesian3dVectorArray(0, 0, 1)
    return optika.rays.RayVectorArray(
        wavelength=wavelength,
        position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        direction=direction,
    )


_normal = na.Cartesian3dVectorArray(0, 0, -1)
_orders = na.ScalarArray(np.arange(-100, 101), axes="m")

# a fine grating, for which the diffracted orders leave at angles that
# matter, and a coarse one, for which every order of consequence leaves
# close to the normal and the thin-grating formulas conserve energy
_spacing_fine = 1 / (2200 / u.mm)
_spacing_coarse = 100 * u.um


def test_sinusoidal_rulings_efficiency_bessel():
    """
    The efficiency of sinusoidal rulings is the square of the Bessel
    function, per Table 1 of Magnusson and Gaylord (1978), not the Bessel
    function itself, of the phase modulation of a relief grating, which
    for the first order at normal incidence is the depth times half of
    one plus the cosine of the angle the order leaves at.
    """
    depth = 42 * u.nm
    wavelength = 150 * u.nm
    rulings = optika.rulings.SinusoidalRulings(
        spacing=_spacing_fine,
        depth=depth,
        diffraction_order=1,
    )

    result = rulings.efficiency(_rays(wavelength), _normal)

    sin_beta = (wavelength / _spacing_fine).to_value(u.dimensionless_unscaled)
    cos_beta = np.sqrt(1 - np.square(sin_beta))
    gamma = (np.pi * depth / wavelength).to_value(u.dimensionless_unscaled)
    gamma = gamma * (1 + cos_beta) / 2
    expected = np.square(scipy.special.jv(1, 2 * gamma))
    assert np.isclose(result, expected)


def test_sinusoidal_rulings_efficiency_glass():
    """
    On a transmissive surface the phase modulation is the index contrast
    across the relief, so a grating carrying light from vacuum into glass
    is much weaker than the same relief on a mirror.
    """
    depth = 42 * u.nm
    wavelength = 150 * u.nm
    n2 = 1.5
    rulings = optika.rulings.SinusoidalRulings(
        spacing=_spacing_fine,
        depth=depth,
        diffraction_order=1,
    )

    result = rulings.efficiency(
        rays=_rays(wavelength),
        normal=_normal,
        index_refraction_new=n2,
        is_mirror=False,
    )

    sin_beta_1 = (wavelength / _spacing_fine).to_value(u.dimensionless_unscaled)
    cos_beta = np.sqrt(1 - np.square(sin_beta_1 / n2))
    gamma = (np.pi * depth / wavelength).to_value(u.dimensionless_unscaled)
    gamma = gamma * np.abs(1 - n2 * cos_beta) / 2
    expected = np.square(scipy.special.jv(1, 2 * gamma))
    assert np.isclose(result, expected)

    result_mirror = rulings.efficiency(_rays(wavelength), _normal)
    assert result < result_mirror


@pytest.mark.parametrize(
    argnames="rulings",
    argvalues=[
        optika.rulings.SinusoidalRulings(
            spacing=_spacing_fine,
            depth=42 * u.nm,
            diffraction_order=1,
        ),
        optika.rulings.SquareRulings(
            spacing=_spacing_fine,
            depth=42 * u.nm,
            diffraction_order=1,
        ),
        optika.rulings.SawtoothRulings(
            spacing=_spacing_fine,
            depth=42 * u.nm,
            diffraction_order=1,
        ),
        optika.rulings.TriangularRulings(
            spacing=_spacing_fine,
            depth=42 * u.nm,
            diffraction_order=1,
        ),
        optika.rulings.RectangularRulings(
            spacing=_spacing_fine,
            depth=42 * u.nm,
            diffraction_order=1,
            ratio_duty=0.3,
        ),
    ],
)
def test_efficiency_reciprocal(rulings: optika.rulings.AbstractRulings):
    """
    Sending the diffracted light back along its path returns it to the
    incident direction with the same efficiency, since the phase
    modulation of a relief grating is symmetric in the angles of
    incidence and diffraction.
    """
    wavelength = 150 * u.nm
    alpha = 30 * u.deg
    direction = na.Cartesian3dVectorArray(
        x=np.sin(alpha),
        y=0,
        z=np.cos(alpha),
    )
    rays = _rays(wavelength, direction)

    result = rulings.efficiency(rays, _normal)

    rays_effective = rulings.incident_effective(rays, _normal)
    direction_diffracted = optika.materials.snells_law(
        direction=rays_effective.direction,
        index_refraction=1,
        index_refraction_new=1,
        is_mirror=True,
        normal=_normal,
    )
    rays_reversed = _rays(wavelength, -direction_diffracted)

    result_reversed = rulings.efficiency(rays_reversed, _normal)

    assert np.isclose(result, result_reversed)


def test_efficiency_evanescent():
    """An order that cannot propagate carries no light."""
    rulings = optika.rulings.SinusoidalRulings(
        spacing=_spacing_fine,
        depth=42 * u.nm,
        diffraction_order=5,
    )
    result = rulings.efficiency(_rays(150 * u.nm), _normal)
    assert np.isfinite(result)
    assert result == 0


@pytest.mark.parametrize(
    argnames="rulings",
    argvalues=[
        optika.rulings.SinusoidalRulings(
            spacing=_spacing_coarse,
            depth=42 * u.nm,
            diffraction_order=_orders,
        ),
        optika.rulings.TriangularRulings(
            spacing=_spacing_coarse,
            depth=42 * u.nm,
            diffraction_order=_orders,
        ),
    ],
)
def test_efficiency_conserved(rulings: optika.rulings.AbstractRulings):
    """
    A thin phase grating absorbs nothing, so the efficiency summed over
    all orders is one. Only the profiles whose efficiency falls off faster
    than the square of the order can be checked to high precision with a
    finite sum, and only for a grating coarse enough that every order of
    consequence leaves close to the normal.
    """
    wavelength = na.ScalarArray([100, 150, 200, 400] * u.nm, axes="w")
    result = rulings.efficiency(_rays(wavelength), _normal).sum("m")
    assert np.allclose(result, 1)


def test_sawtooth_rulings_efficiency_blazed():
    """
    A sawtooth profile a whole wave deep sends all of the light into the
    blazed order. This is a removable singularity of the formula.
    """
    wavelength = 150 * u.nm
    rulings = optika.rulings.SawtoothRulings(
        spacing=_spacing_coarse,
        depth=wavelength / 2,
        diffraction_order=_orders,
    )
    result = rulings.efficiency(_rays(wavelength), _normal)
    assert np.isfinite(result).all()
    assert np.isclose(result.sum("m"), 1)
    assert np.isclose(result.max("m"), 1)


def test_triangular_rulings_efficiency_resonant():
    """
    A triangular profile a whole wave deep puts a quarter of the light
    into each of the two orders its slopes are blazed for. This is a
    removable singularity of the formula.
    """
    wavelength = 150 * u.nm
    rulings = optika.rulings.TriangularRulings(
        spacing=_spacing_coarse,
        depth=wavelength / 2,
        diffraction_order=_orders,
    )
    result = rulings.efficiency(_rays(wavelength), _normal)
    assert np.isfinite(result).all()
    assert np.isclose(result.sum("m"), 1)
    assert np.isclose(result[dict(m=100 - 2)], 1 / 4)
    assert np.isclose(result[dict(m=100 + 2)], 1 / 4)
