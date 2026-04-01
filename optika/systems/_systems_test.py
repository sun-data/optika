import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .._tests import test_mixins


class AbstractTestAbstractSystem(
    test_mixins.AbstractTestPlottable,
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTransformable,
    test_mixins.AbstractTestShaped,
):

    @pytest.mark.parametrize(
        argnames="scene",
        argvalues=[
            na.FunctionArray(
                inputs=na.SpectralPositionalVectorArray(
                    wavelength=na.linspace(
                        start=530 * u.nm,
                        stop=531 * u.nm,
                        axis="wavelength",
                        num=2,
                    ),
                    position=na.Cartesian2dVectorLinearSpace(
                        start=-1,
                        stop=+1,
                        axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                        num=11,
                    ),
                ),
                outputs=na.random.uniform(
                    low=0 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
                    high=100 * u.photon / u.cm**2 / u.arcsec**2 / u.s / u.nm,
                    shape_random=dict(field_x=10, field_y=10),
                ),
            )
        ],
    )
    def test_image(
        self,
        a: optika.systems.AbstractSystem,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
    ):
        result = a.image(scene)
        assert isinstance(result, na.FunctionArray)
        assert isinstance(result.inputs, na.SpectralPositionalVectorArray)
        assert isinstance(result.outputs, na.AbstractScalar)
        assert np.all(result.inputs.wavelength > 0 * u.nm)
        assert na.unit_normalized(result.inputs.position).is_equivalent(u.mm)
        assert result.outputs.sum() != (0 * u.electron)
