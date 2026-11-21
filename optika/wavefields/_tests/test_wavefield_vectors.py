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
        assert na.unit_normalized(amplitude).is_equivalent(u.dimensionless_unscaled)

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


_wavefield = wavefields[0]
"""A single wavefield, for the operations which take one."""

_vector_2d = na.Cartesian2dVectorArray(1, 1) * u.mm
"""A vector which a wavefield cannot be combined with, having the wrong components."""


def test_type_explicit() -> None:
    """The explicit form of a wavefield is a WavefieldVectorArray."""
    assert _wavefield.type_explicit is optika.wavefields.WavefieldVectorArray

    explicit = _wavefield.explicit
    assert isinstance(explicit, optika.wavefields.WavefieldVectorArray)
    assert np.all(explicit.position == _wavefield.position)
    assert np.all(explicit.amplitude == _wavefield.amplitude)


def test_type_matrix() -> None:
    """There is no matrix of wavefields."""
    with pytest.raises(NotImplementedError):
        _wavefield.type_matrix


def test_from_scalar() -> None:
    """A wavefield made from a scalar has that scalar for each component."""
    scalar = na.ScalarArray(2.0)
    result = optika.wavefields.WavefieldVectorArray.from_scalar(scalar)
    components = [
        "wavelength",
        "position",
        "amplitude",
        "normal",
        "area",
        "index_refraction",
        "unvignetted",
    ]
    for component in components:
        assert getattr(result, component) is scalar


def test__array_matmul__cartesian3d_matrix() -> None:
    """A matrix on the right turns the positions and the normals."""
    matrix = na.Cartesian3dXRotationMatrixArray(30 * u.deg)
    result = _wavefield @ matrix

    assert np.all(result.position == _wavefield.position @ matrix)
    assert np.all(result.normal == _wavefield.normal @ matrix)
    assert result.amplitude is _wavefield.amplitude
    assert result.wavelength is _wavefield.wavelength
    assert result.area is _wavefield.area


def test__array_matmul__scalar() -> None:
    """Multiplying by a scalar is left to the base class, which can do it."""
    result = _wavefield @ 2

    assert np.all(result.position == 2 * _wavefield.position)
    assert np.all(result.normal == 2 * _wavefield.normal)


@pytest.mark.parametrize(
    argnames="x1,x2",
    argvalues=[
        (_wavefield, _vector_2d),
        (_vector_2d, _wavefield),
    ],
)
def test__array_matmul__unsupported(x1: na.AbstractArray, x2: na.AbstractArray) -> None:
    """A wavefield and a vector of other components cannot be multiplied."""
    assert x1 @ x2 is None


def test__array_matmul__neither() -> None:
    """
    Asked about two things which are not wavefields, a wavefield declines to
    answer.

    Only reachable by calling the method directly, since matmul is dispatched
    to this class only when one of its operands is a wavefield.
    """
    assert _wavefield.__array_matmul__(1, 2) is NotImplemented


def test__array_add__reversed() -> None:
    """A vector on the left moves the positions of a wavefield."""
    vector = na.Cartesian3dVectorArray(1, 2, 3) * u.mm
    result = vector + _wavefield

    assert np.all(result.position == vector + _wavefield.position)
    assert result.normal is _wavefield.normal
    assert result.amplitude is _wavefield.amplitude
    assert result.wavelength is _wavefield.wavelength
    assert result.area is _wavefield.area


@pytest.mark.parametrize(
    argnames="x1,x2",
    argvalues=[
        (_wavefield, _vector_2d),
        (_vector_2d, _wavefield),
    ],
)
def test__array_add__unsupported(x1: na.AbstractArray, x2: na.AbstractArray) -> None:
    """A wavefield and a vector of other components cannot be added."""
    with pytest.raises(TypeError):
        x1 + x2


def test__array_add__neither() -> None:
    """
    Asked about two things which are not wavefields, a wavefield declines to
    answer.

    Only reachable by calling the method directly, as with matmul above.
    """
    assert _wavefield.__array_add__(1, 2) is NotImplemented
