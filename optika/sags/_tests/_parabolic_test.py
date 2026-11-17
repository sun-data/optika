import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ..._tests import test_mixins
from ._abc_test import radius_parameterization, positions
from ._conic_test import AbstractTestAbstractConicSag


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.ParabolicSag(
            focal_length=radius / 2,
            transformation=transformation,
        )
        for radius in radius_parameterization()
        for transformation in test_mixins.transformation_parameterization
    ],
)
class TestParabolicSag(
    AbstractTestAbstractConicSag,
):

    @pytest.mark.parametrize("position", positions)
    def test_normal(
        self,
        a: optika.sags.AbstractSag,
        position: na.AbstractCartesian3dVectorArray,
    ):
        super().test_normal(a, position)

        result = a.normal(position)

        result_expected = super(type(a), a).normal(position)

        assert np.allclose(result, result_expected)


@pytest.mark.parametrize("focal_length", [-1 * u.mm, 1 * u.mm, -1000 * u.mm])
@pytest.mark.parametrize("radius", [10 * u.mm, 76 * u.mm])
@pytest.mark.parametrize("angle", [-4 * u.deg, 4 * u.deg])
def test_intercept_of_ray_on_the_surface(
    focal_length: u.Quantity,
    radius: u.Quantity,
    angle: u.Quantity,
):
    """
    A ray which starts on the paraboloid must be intercepted where it already
    is, not at the surface's other crossing.

    A paraboloid seen at grazing incidence is crossed twice by the same line,
    and a ray launched from the surface is the degenerate case of that: one
    root is zero and the other is far away.  Choosing between the roots by the
    sign of the focal length, rather than by their distance from the ray,
    moves such a ray to the opposite side of the paraboloid whenever it
    travels toward the axis on a steep flank.
    :class:`optika.systems.SequentialSystem` meets exactly that case when it
    propagates its solved stop rays back through the surface they were
    launched from, and the resulting entrance pupil is silently wrong rather
    than obviously so.
    """
    sag = optika.sags.ParabolicSag(focal_length=focal_length)

    position = na.Cartesian3dVectorArray(x=radius, y=0 * u.mm, z=0 * u.mm)
    position = position.replace(z=sag(position))

    direction = na.Cartesian3dVectorArray(
        x=np.sin(angle),
        y=0,
        z=np.cos(angle),
    )

    rays = optika.rays.RayVectorArray(
        wavelength=500 * u.nm,
        position=position,
        direction=direction,
    )

    result = sag.intercept(rays)

    assert np.allclose(result.position, position)


@pytest.mark.parametrize(
    argnames="focal_length",
    argvalues=[-1000 * u.mm, -100 * u.mm, 100 * u.mm, 1000 * u.mm],
)
def test_intercept_matches_the_conic_intercept(focal_length: u.Quantity):
    """
    A paraboloid is a conic with a conic constant of -1, so the closed-form
    intercept here must agree with the general conic intercept.

    Every ray in this test strikes the paraboloid, since the two
    implementations report a miss differently: this one gives not-a-number and
    :class:`optika.sags.ConicSag` gives infinity.
    """
    sag = optika.sags.ParabolicSag(focal_length=focal_length)
    sag_conic = optika.sags.ConicSag(radius=2 * focal_length, conic=-1)

    radius = na.linspace(10, 100, axis="radius", num=11) * u.mm
    azimuth = na.linspace(0, 360, axis="azimuth", num=7) * u.deg

    position = na.Cartesian3dVectorArray(
        x=radius * np.cos(azimuth),
        y=radius * np.sin(azimuth),
        z=-500 * u.mm,
    )

    angle = na.linspace(-4, 4, axis="angle", num=5) * u.deg
    direction = na.Cartesian3dVectorArray(
        x=np.sin(angle),
        y=0,
        z=np.cos(angle),
    )

    rays = optika.rays.RayVectorArray(
        wavelength=500 * u.nm,
        position=position,
        direction=direction,
    )

    result = sag.intercept(rays)
    result_expected = sag_conic.intercept(rays)

    assert np.allclose(result.position, result_expected.position)
