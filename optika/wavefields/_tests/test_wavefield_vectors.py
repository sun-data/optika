import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


wavefields = [
    optika.wavefields.WavefieldVectorArray(
        wavelength=500 * u.nm,
        position=na.Cartesian3dVectorArray(0, 1, 2) * u.mm,
        amplitude=1 + 0j,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        area=1 * u.mm**2,
    ),
    optika.wavefields.WavefieldVectorArray(
        wavelength=500 * u.nm,
        position=na.Cartesian3dVectorLinearSpace(
            start=-10 * u.mm,
            stop=10 * u.mm,
            axis="s",
            num=5,
        ).explicit,
        amplitude=np.exp(1j * na.linspace(0, np.pi, axis="s", num=5)),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        area=4 * u.mm**2,
    ),
]


@pytest.mark.parametrize("array", wavefields)
class TestWavefieldVectorArray:

    def test_amplitude(self, array: optika.wavefields.AbstractWavefieldVectorArray):
        amplitude = na.as_named_array(array.amplitude)
        assert isinstance(amplitude, na.AbstractScalar)
        assert na.unit_normalized(amplitude).is_equivalent(
            u.dimensionless_unscaled
        )

    def test_normal(self, array: optika.wavefields.AbstractWavefieldVectorArray):
        normal = array.normal
        assert isinstance(normal, na.AbstractCartesian3dVectorArray)
        assert np.allclose(normal.length, 1)

    def test_area(self, array: optika.wavefields.AbstractWavefieldVectorArray):
        area = na.as_named_array(array.area)
        assert isinstance(area, na.AbstractScalar)
        assert na.unit_normalized(area).is_equivalent(u.mm**2)

    def test_translation(
        self,
        array: optika.wavefields.AbstractWavefieldVectorArray,
    ):
        transformation = na.transformations.Cartesian3dTranslation(
            x=1 * u.mm,
            z=2 * u.mm,
        )
        result = transformation(array)

        assert isinstance(result, optika.wavefields.AbstractWavefieldVectorArray)
        assert np.all(result.position.x == array.position.x + 1 * u.mm)
        assert np.all(result.position.z == array.position.z + 2 * u.mm)

        # Translations should not affect the surface normal.
        assert np.all(result.normal == array.normal)

        assert result.amplitude is array.amplitude
        assert result.wavelength is array.wavelength
        assert result.area is array.area

    def test_rotation(
        self,
        array: optika.wavefields.AbstractWavefieldVectorArray,
    ):
        transformation = na.transformations.Cartesian3dRotationY(90 * u.deg)
        result = transformation(array)

        assert isinstance(result, optika.wavefields.AbstractWavefieldVectorArray)

        matrix = na.Cartesian3dYRotationMatrixArray(90 * u.deg)

        # Rotations should rotate both the position and the normal.
        assert np.allclose(result.position, matrix @ array.position)
        assert np.allclose(result.normal, matrix @ array.normal)

        assert result.amplitude is array.amplitude
        assert result.wavelength is array.wavelength
        assert result.area is array.area
