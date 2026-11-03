import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization

_position = (
    na.Cartesian3dVectorArray(
        x=na.linspace(-20, 20, axis="x", num=5),
        y=na.linspace(-30, 30, axis="y", num=7),
        z=0,
    )
    * u.mm
)
"""A grid of positions near the vertex at which to evaluate the sags."""


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


@pytest.mark.parametrize("radius", [500 * u.mm, -500 * u.mm])
def test_sphere(radius: u.Quantity):
    """
    A toroid whose two radii are equal is a sphere.

    Parameters
    ----------
    radius
        The radius of curvature of both the toroid and the sphere.
    """
    a = optika.sags.ToroidalSag(
        radius=radius,
        radius_of_rotation=radius,
    )
    b = optika.sags.SphericalSag(radius=radius)
    assert np.allclose(a(_position), b(_position))
    assert np.allclose(a.normal(_position), b.normal(_position))


@pytest.mark.parametrize("radius", [400 * u.mm, -400 * u.mm])
@pytest.mark.parametrize("radius_of_rotation", [800 * u.mm, -800 * u.mm])
def test_normal_finite_difference(
    radius: u.Quantity,
    radius_of_rotation: u.Quantity,
):
    """
    The normal of a toroid is the normal of its sag, computed by central
    finite differences, for every combination of the signs of its radii.

    Parameters
    ----------
    radius
        The minor radius of the toroid.
    radius_of_rotation
        The major radius of the toroid.
    """
    a = optika.sags.ToroidalSag(
        radius=radius,
        radius_of_rotation=radius_of_rotation,
    )
    h = 1e-4 * u.mm
    dx = na.Cartesian3dVectorArray(h, 0 * u.mm, 0 * u.mm)
    dy = na.Cartesian3dVectorArray(0 * u.mm, h, 0 * u.mm)
    dzdx = (a(_position + dx) - a(_position - dx)) / (2 * h)
    dzdy = (a(_position + dy) - a(_position - dy)) / (2 * h)
    normal = na.Cartesian3dVectorArray(dzdx, dzdy, -1)
    normal = normal / normal.length
    assert np.allclose(a.normal(_position), normal)


@pytest.mark.parametrize("radius", [400 * u.mm, -400 * u.mm])
@pytest.mark.parametrize("radius_of_rotation", [800 * u.mm, -800 * u.mm])
def test_sag_vertex(
    radius: u.Quantity,
    radius_of_rotation: u.Quantity,
):
    """
    Near its vertex, the sag of a toroid is that of a paraboloid with the
    curvatures of its two radii.

    Parameters
    ----------
    radius
        The minor radius of the toroid.
    radius_of_rotation
        The major radius of the toroid.
    """
    a = optika.sags.ToroidalSag(
        radius=radius,
        radius_of_rotation=radius_of_rotation,
    )
    position = _position / 100
    x = position.x
    y = position.y
    expected = np.square(x) / (2 * radius_of_rotation) + np.square(y) / (2 * radius)
    assert np.allclose(a(position), expected, rtol=1e-6, atol=1e-12 * u.mm)
