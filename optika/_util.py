from typing import Any
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "shape",
    "direction",
    "angles",
]


_shaped_type = (
    na.AbstractArray,
    na.transformations.AbstractTransformation,
    optika.mixins.Shaped,
)


def shape(a: Any) -> dict[str, int]:
    """
    Return the array shape of the given object.

    If the given object is an instance of :class:`named_arrays.AbstractArray`,
    :class:`named_arrays.transformations.AbstractTransformation`, or
    :class:`optika.mixins.Shaped`, ``a.shape`` will be returned.
    Otherwise an empty dictionary will be returned.

    Parameters
    ----------
    a
        The object to find the shape of.
    """
    if isinstance(a, _shaped_type):
        return a.shape
    else:
        return dict()


def direction(
    angles: na.AbstractCartesian2dVectorArray,
) -> na.Cartesian3dVectorArray:
    r"""
    Given a 2D vector of azimuth and elevation angles, convert to a
    3D vector of direction cosines.

    Parameters
    ----------
    angles
        A vector of azimuth and elevation angles.

    Notes
    -----
    If the azimuth and elevation angles are taken to :math:`\phi_x` and
    :math:`\phi_y`, the direction cosines :math:`\vec{d}` can be found by

    .. math::

        \vec{d} = R_y(\phi_x) R_x(\phi_y) \hat{z}

    where :math:`R_x(\theta)` and :math:`R_y(\theta)` are the rotation matrices
    about the :math:`x` and :math:`y` axes respectively.

    See Also
    --------
    :func:`angles` : Inverse of this function
    """
    return na.Cartesian3dVectorArray(
        x=-np.cos(angles.y) * np.sin(angles.x),
        y=-np.sin(angles.y),
        z=+np.cos(angles.y) * np.cos(angles.x),
    )


def angles(
    direction: na.AbstractCartesian3dVectorArray,
) -> na.Cartesian2dVectorArray:
    """
    Convert a 3D vector of direction cosines to a 2D vector of azimuth and
    elevation angles.

    Parameters
    ----------
    direction
        A vector of direction cosines.

    See Also
    --------
    :func:`direction` : Inverse of this function
    """
    if na.unit(direction) is None:
        direction = direction << u.dimensionless_unscaled
    return na.Cartesian2dVectorArray(
        x=-np.arctan2(direction.x, direction.z).to(u.deg),
        y=-np.arcsin(direction.y / direction.length).to(u.deg),
    )
