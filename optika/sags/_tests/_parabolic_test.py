import pytest
import numpy as np
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
