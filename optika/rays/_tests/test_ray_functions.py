import astropy.units as u
import named_arrays as na
import optika


def test_type_explicit():
    """
    The explicit form of a ray function is a ray function.

    :mod:`named_arrays` asks an array which class to build when it needs a
    concrete one, and the answer for any ray function is this class.
    """
    result = optika.rays.RayFunctionArray(
        inputs=optika.vectors.ObjectVectorArray(wavelength=500 * u.nm),
        outputs=optika.rays.RayVectorArray(
            wavelength=500 * u.nm,
            position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        ),
    )

    assert result.type_explicit is optika.rays.RayFunctionArray
