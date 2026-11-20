import warnings
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import AbstractTestAbstractSag, radius_parameterization


class AbstractTestAbstractConicSag(
    AbstractTestAbstractSag,
):
    def test_radius(self, a: optika.sags):
        assert isinstance(na.as_named_array(a.radius), na.ScalarLike)
        assert na.unit_normalized(a.radius).is_equivalent(u.mm)

    def test_conic(self, a: optika.sags):
        assert isinstance(na.as_named_array(a.conic), na.ScalarLike)
        assert na.unit_normalized(a.conic).is_equivalent(u.dimensionless_unscaled)


def conic_parameterization() -> list[u.Quantity | na.AbstractScalar]:
    nominals = [
        0 * u.dimensionless_unscaled,
        na.ScalarLinearSpace(0, 1, axis="conic", num=4),
    ]
    widths = [
        None,
        0.1 * u.dimensionless_unscaled,
    ]
    return [
        nominal if width is None else na.NormalUncertainScalarArray(nominal, width)
        for nominal in nominals
        for width in widths
    ]


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ConicSag(
            radius=radius,
            conic=conic,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for conic in conic_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestConicSag(
    AbstractTestAbstractConicSag,
):
    pass


def test_intercept_grazing_conic():
    """
    Regression test for the closed-form conic intercept: on the steep flank of
    a grazing-incidence conic, the intercept must land on the surface and on the
    same sheet as the vertex.  An iterative root-finder can otherwise converge to
    the far / wrong-sheet root, which corrupts the resulting image.
    """
    # a steeply-curved hyperboloid, like a grazing-incidence secondary mirror
    sag = optika.sags.ConicSag(radius=-0.7 * u.mm, conic=-1.0007)

    # rays parallel to the axis, grazing the flank far from the vertex
    azimuth = na.linspace(0, 2 * np.pi, axis="ray", num=64, endpoint=False) * u.rad
    radius = 68 * u.mm
    rays = optika.rays.RayVectorArray(
        position=na.Cartesian3dVectorArray(
            x=radius * np.cos(azimuth),
            y=radius * np.sin(azimuth),
            z=-2000 * u.mm,
        ),
        direction=na.Cartesian3dVectorArray(0, 0, 1),
    )

    position = sag.intercept(rays).position

    # the intercept lies on the surface ...
    assert np.allclose(sag(position), position.z)

    # ... and on the same sheet of the conic as the vertex
    c = 1 / sag.radius
    r2 = np.square(position.x) + np.square(position.y)
    assert np.all((position.z * (c * r2 - position.z)) >= 0)


@pytest.mark.parametrize("conic", [-1.5, -1, -0.9999, 0, 0.5])
@pytest.mark.parametrize("angle", [0, 1e-160, 1e-9, 1e-7, 1e-5, 1e-3, 1e-2] * u.rad)
@pytest.mark.parametrize("distance", [1000, 1234.567, 100_000] * u.mm)
def test_intercept_of_nearly_axial_rays(
    conic: float,
    angle: u.Quantity,
    distance: u.Quantity,
) -> None:
    r"""
    The intercept of a ray travelling nearly parallel to the axis of a conic
    must lie on the vertex sheet of the conic, near the vertex, to within the
    precision of a double.

    The textbook form of the quadratic formula finds the nearer root of such
    a ray by subtracting two nearly equal numbers when the conic is nearly
    a paraboloid, which left intercepts off the surface.
    The test of which sheet a root is on, :math:`z (c r^2 - z) \geq 0`,
    reduces to :math:`-z^2 \geq 0` on the axis, so the rounding of :math:`z`
    rejected the near root of a ray on or near the axis.
    The ray then missed the conic or, for a paraboloid, took the far root,
    which can be more than 1e25 mm away.
    That depends on where the ray starts, so the rays start at several
    distances and are aimed at the aperture at :math:`z = 0`.
    From far away, the two crossings of a closed conic are close together
    compared with their distance, which costs precision in proportion to the
    square of the distance, so the tolerance grows that way.
    A slope of 1e-160 rad makes :math:`A` subnormal for the paraboloid.
    """
    sag = optika.sags.ConicSag(radius=2000 * u.mm, conic=conic)

    radius = na.linspace(-50, 50, axis="radius", num=21) * u.mm
    direction = na.Cartesian3dVectorArray(
        x=np.sin(angle),
        y=0.5 * np.sin(angle),
        z=np.cos(angle),
    )
    direction = direction / direction.length
    target = na.Cartesian3dVectorArray(x=radius, y=0.3 * radius, z=0 * u.mm)
    rays = optika.rays.RayVectorArray(
        wavelength=63 * u.nm,
        position=target - direction * distance / direction.z,
        direction=direction,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        position = sag.intercept(rays).position

    atol = 1e-11 * u.mm * np.square(distance / (1000 * u.mm))
    assert np.all(np.isfinite(position))
    assert np.allclose(sag(position), position.z, rtol=0, atol=atol)
    assert np.all(np.abs(position.z) < 10 * u.mm)
