import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


class AbstractTestAbstractSlopeErrorParameters(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestShaped,
):
    def test_step_size(self, a: optika.metrology.AbstractSlopeErrorParameters):
        assert np.issubdtype(na.get_dtype(a.step_size), float)
        assert na.unit_normalized(a.step_size).is_equivalent(u.mm)

    def test_kernel_size(self, a: optika.metrology.AbstractSlopeErrorParameters):
        assert np.issubdtype(na.get_dtype(a.kernel_size), float)
        assert na.unit_normalized(a.kernel_size).is_equivalent(u.mm)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.metrology.SlopeErrorParameters(
            step_size=4 * u.mm,
            kernel_size=2 * u.mm,
        )
    ],
)
class TestSlopeErrorParameters(
    AbstractTestAbstractSlopeErrorParameters,
):
    pass
