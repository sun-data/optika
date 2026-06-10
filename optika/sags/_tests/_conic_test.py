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
