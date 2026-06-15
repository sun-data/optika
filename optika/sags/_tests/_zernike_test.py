import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc_test import AbstractTestAbstractSag


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ZernikeSag(
            coefficients=[0, 0, 0, 200e-6] * u.mm,
            radius=50 * u.mm,
        ),
        optika.sags.ZernikeSag(
            base=optika.sags.SphericalSag(radius=500 * u.mm),
            coefficients=[100, 50, 50, 200, 30, 30, 10, 10, 5, 5, 20] * u.nm,
            radius=50 * u.mm,
        ),
        optika.sags.ZernikeSag(
            base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
            coefficients=na.ScalarArray(
                ndarray=[0, 0, 0, 100, 0, 50] * u.nm,
                axes="zernike",
            ),
            radius=25 * u.mm,
            transformation=na.transformations.Cartesian3dTranslation(
                z=10 * u.mm,
            ),
        ),
    ],
)
class TestZernikeSag(
    AbstractTestAbstractSag,
):
    pass


def test_coefficients_invalid_axis():
    sag = optika.sags.ZernikeSag(
        coefficients=na.ScalarArray([0, 0, 0, 1] * u.um, axes="not_zernike"),
        radius=50 * u.mm,
    )
    with pytest.raises(ValueError):
        sag(na.Cartesian3dVectorArray() * u.mm)


def test_defocus_closed_form():
    """
    A pure-defocus ZernikeSag over a flat profile should equal the analytic
    form of the Noll-normalized defocus polynomial.
    """
    c = 100 * u.nm
    radius = 50 * u.mm

    sag = optika.sags.ZernikeSag(
        coefficients=[0 * u.nm, 0 * u.nm, 0 * u.nm, c],
        radius=radius,
    )

    position = na.Cartesian3dVectorArray(
        x=na.linspace(-50, 50, axis="x", num=11) * u.mm,
        y=na.linspace(-50, 50, axis="y", num=11) * u.mm,
        z=0 * u.mm,
    )

    rho2 = np.square(position.xy.length / radius)
    expected = c * np.sqrt(3) * (2 * rho2 - 1)

    assert np.allclose(sag(position), expected)


def test_defocus_focal_shift():
    """
    Adding a Zernike defocus term to a paraboloid yields another exact
    paraboloid with a focal length given by

    .. math::

        \\frac{1}{f'} = \\frac{1}{f} + \\frac{8 \\sqrt{3} c}{R^2},

    so collimated rays should still focus perfectly, at the shifted focus.
    """
    f = 1000 * u.mm
    radius = 25 * u.mm
    c = 1 * u.um

    sag = optika.sags.ZernikeSag(
        base=optika.sags.ParabolicSag(focal_length=f),
        coefficients=[0 * u.um, 0 * u.um, 0 * u.um, c],
        radius=radius,
    )

    position = na.Cartesian3dVectorArray(
        x=na.linspace(-radius, radius, axis="rx", num=11),
        y=na.linspace(-radius, radius, axis="ry", num=11),
        z=2 * f,
    )
    rays = optika.rays.RayVectorArray(
        wavelength=500 * u.nm,
        position=position,
        direction=na.Cartesian3dVectorArray(0, 0, -1),
    )

    rays = sag.intercept(rays)
    normal = sag.normal(rays.position)
    direction = rays.direction
    direction = direction - 2 * (direction @ normal) * normal

    f_perturbed = 1 / (1 / f + 8 * np.sqrt(3) * c / np.square(radius))
    z_focus = f_perturbed - np.sqrt(3) * c

    def rms_spot(z: u.Quantity) -> u.Quantity:
        t = (z - rays.position.z) / direction.z
        spot = rays.position + direction * t
        return np.sqrt(np.mean(np.square(spot.x) + np.square(spot.y)))

    assert rms_spot(z_focus) < 100 * u.nm
    assert rms_spot(f) > 50 * u.um
