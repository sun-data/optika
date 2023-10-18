import pytest
import abc
import numpy as np
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
