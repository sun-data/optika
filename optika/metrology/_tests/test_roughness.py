import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


class AbstractTestAbstractRoughnessParameters(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestShaped,
):
    def test_period_min(self, a: optika.metrology.AbstractRoughnessParameters):
        assert np.issubdtype(na.get_dtype(a.period_min), float)
        assert na.unit_normalized(a.period_min).is_equivalent(u.mm)

    def test_period_max(self, a: optika.metrology.AbstractRoughnessParameters):
        assert np.issubdtype(na.get_dtype(a.period_max), float)
        assert na.unit_normalized(a.period_max).is_equivalent(u.mm)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.metrology.RoughnessParameters(
            period_min=2 * u.mm,
            period_max=4 * u.mm,
        )
    ],
)
class TestRoughnessParameters(
    AbstractTestAbstractRoughnessParameters,
):
    pass
