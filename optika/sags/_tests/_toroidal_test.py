import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ToroidalSag(
            radius=radius,
            radius_of_rotation=2 * radius_of_rotation,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for radius_of_rotation in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestToroidalSag(
    AbstractTestAbstractSag,
):
    pass


@pytest.mark.parametrize(
    argnames="radius",
    argvalues=[
        500 * u.mm,
        -500 * u.mm,
    ],
)
def test_toroidal_matches_spherical(radius: u.Quantity):
    """
    A toroid whose two radii are equal is a sphere, so its sag and normal
    should match those of :class:`optika.sags.SphericalSag`, whatever the sign
    of the radius.
    """
    position = (
        na.Cartesian3dVectorArray(
            x=na.linspace(-20, 20, axis="x", num=5),
            y=na.linspace(-20, 20, axis="y", num=5),
            z=0,
        )
        * u.mm
    )

    toroid = optika.sags.ToroidalSag(
        radius=radius,
        radius_of_rotation=radius,
    )
    sphere = optika.sags.SphericalSag(radius=radius)

    assert np.allclose(toroid(position), sphere(position))
    assert np.allclose(toroid.normal(position), sphere.normal(position))


@pytest.mark.parametrize("radius_sign", [1, -1])
@pytest.mark.parametrize("radius_of_rotation_sign", [1, -1])
def test_toroidal_normal_matches_gradient(
    radius_sign: int,
    radius_of_rotation_sign: int,
):
    """
    The normal of a toroid should agree with the finite-difference gradient
    of its sag for every combination of the signs of its two radii.
    """
    sag = optika.sags.ToroidalSag(
        radius=radius_sign * 400 * u.mm,
        radius_of_rotation=radius_of_rotation_sign * 800 * u.mm,
    )
    x = na.linspace(-10, 10, axis="x", num=5) * u.mm
    y = na.linspace(-10, 10, axis="y", num=5) * u.mm
    h = 1e-4 * u.mm

    def z(x, y):
        return sag(na.Cartesian3dVectorArray(x, y, 0 * u.mm))

    dzdx = (z(x + h, y) - z(x - h, y)) / (2 * h)
    dzdy = (z(x, y + h) - z(x, y - h)) / (2 * h)
    expected = na.Cartesian3dVectorArray(
        x=dzdx.to(u.dimensionless_unscaled),
        y=dzdy.to(u.dimensionless_unscaled),
        z=-1 * u.dimensionless_unscaled,
    )
    expected = expected / expected.length

    normal = sag.normal(na.Cartesian3dVectorArray(x, y, 0 * u.mm))

    assert np.allclose(normal, expected, atol=1e-8)
    assert np.allclose(
        z(x, 0 * u.mm),
        np.square(x) / (2 * sag.radius_of_rotation),
        rtol=1e-3,
    )
