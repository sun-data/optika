import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests.test_surfaces import AbstractTestAbstractSurface


class AbstractTestAbstractImagingSensor(
    AbstractTestAbstractSurface,
):
    def test_num_pixel(self, a: optika.sensors.AbstractImagingSensor):
        result = a.num_pixel
        assert isinstance(result, na.AbstractCartesian2dVectorArray)
        assert np.issubdtype(na.get_dtype(result.x), int)
        assert np.issubdtype(na.get_dtype(result.y), int)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.IdealImagingSensor(
            name="test sensor",
            width_pixel=15 * u.um,
            num_pixel=na.Cartesian2dVectorArray(2048, 1024),
        )
    ],
)
class TestIdealImagingSensor(
    AbstractTestAbstractImagingSensor,
):
    pass
