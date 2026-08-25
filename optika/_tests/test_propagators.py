import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import optika.propagators
import optika.rays._tests.test_ray_vectors


class AbstractTestAbstractPropagator(
    abc.ABC,
):
    pass


class AbstractTestAbstractRayPropagator(
    AbstractTestAbstractPropagator,
):
    @pytest.mark.parametrize("rays", optika.rays._tests.test_ray_vectors.rays)
    def test_propagate_rays(
        self,
        a: optika.propagators.AbstractRayPropagator,
        rays: optika.rays.AbstractRayVectorArray,
    ):
        result = a.propagate_rays(rays)

        assert isinstance(result, optika.rays.AbstractRayVectorArray)
        assert not np.all(result == rays)


class AbstractTestAbstractLightPropagator(
    AbstractTestAbstractRayPropagator,
):
    pass


_surface = optika.surfaces.Surface(
    sag=optika.sags.SphericalSag(radius=-100 * u.mm),
    transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
)
"""One thing which knows how to propagate rays."""

_rays = optika.rays.RayVectorArray(
    wavelength=500 * u.nm,
    position=na.Cartesian3dVectorArray(0, 1, 0) * u.mm,
    direction=na.Cartesian3dVectorArray(0, 0, 1),
)


def test_propagate_rays_one():
    """A lone propagator is taken as a sequence of one, not iterated over."""
    result = optika.propagators.propagate_rays(_surface, _rays)

    expected = optika.propagators.propagate_rays([_surface], _rays)
    assert np.all(result.position == expected.position)


def test_accumulate_rays_one():
    """The same, for the version which keeps the rays at every propagator."""
    result = optika.propagators.accumulate_rays(_surface, _rays, axis="surface")

    assert na.shape(result) == dict(surface=1)
