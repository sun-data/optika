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

    def test_timedelta_exposure(self, a: optika.sensors.AbstractImagingSensor):
        result = a.timedelta_exposure
        assert result >= 0 * u.s

    @pytest.mark.parametrize(
        argnames="rays",
        argvalues=[
            optika.rays.RayVectorArray(
                intensity=100 * u.photon / u.s,
                position=na.Cartesian2dVectorLinearSpace(
                    start=-10 * u.mm,
                    stop=10 * u.mm,
                    axis=na.Cartesian2dVectorArray("x", "y"),
                    num=11,
                )
            )
        ],
    )
    def test_readout(
        self,
        a: optika.sensors.AbstractImagingSensor,
        rays: optika.rays.RayVectorArray,
    ):
        result = a.readout(rays)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.Cartesian2dVectorArray)
        assert isinstance(result.outputs, na.AbstractScalar)
        assert result.outputs.unit.is_equivalent(u.electron)


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
