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
                position=na.Cartesian3dVectorArray() * u.mm,
            ),
            optika.rays.RayVectorArray(
                intensity=na.random.poisson(100, shape_random=dict(t=11)) * u.erg / u.s,
                wavelength=500 * u.nm,
                position=na.Cartesian3dVectorArray(
                    x=na.random.uniform(-1, 1, shape_random=dict(t=11)) * u.mm,
                    y=na.random.uniform(-1, 1, shape_random=dict(t=11)) * u.mm,
                    z=0 * u.mm,
                ),
            ),
        ],
    )
    def test_readout(
        self,
        a: optika.sensors.AbstractImagingSensor,
        rays: optika.rays.RayVectorArray,
    ):
        result = a.readout(rays)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert isinstance(result.outputs, na.AbstractScalar)
        assert result.outputs.unit.is_equivalent(u.electron)
        assert a.axis_pixel.x in result.outputs.shape
        assert a.axis_pixel.y in result.outputs.shape


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.ImagingSensor(
            name="test sensor",
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(2048, 1024),
            transformation=na.transformations.Cartesian3dTranslation(x=1 * u.mm),
        ),
    ],
)
class TestImagingSensor(
    AbstractTestAbstractImagingSensor,
):
    pass
