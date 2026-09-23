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


def _shape_efficiency_measured(
    measurement: na.FunctionArray[na.SpectralDirectionalVectorArray, na.AbstractScalar],
    axis_angle: None | str = None,
) -> dict[str, int]:
    """
    The shape of a measured efficiency, less the axes that
    :func:`_interp_efficiency_measured` interpolates over.

    Parameters
    ----------
    measurement
        A function array mapping wavelength and angle of incidence to the
        measured efficiency.
    axis_angle
        The logical axis along which the angle of incidence varies, or
        :obj:`None` if the efficiency was measured at a single angle.
    """
    axes = list(shape(measurement.inputs.wavelength))
    if axis_angle is not None:
        axes.append(axis_angle)
    result = shape(measurement.outputs)
    for ax in axes:
        result.pop(ax, None)
    return result


def _interp_efficiency_measured(
    measurement: na.FunctionArray[na.SpectralDirectionalVectorArray, na.AbstractScalar],
    rays: "optika.rays.RayVectorArray",
    normal: na.AbstractCartesian3dVectorArray,
    axis_angle: None | str = None,
) -> na.ScalarLike:
    """
    Interpolate a measured efficiency onto the wavelengths and angles of
    incidence of the given rays.

    The measurement is interpolated linearly in wavelength and, if it was
    made at more than one angle, linearly in the angle of incidence, the
    angle between each ray and the surface normal.
    Outside the measured range of either, the efficiency is held at the
    nearest measurement.

    Parameters
    ----------
    measurement
        A function array mapping wavelength and angle of incidence to the
        measured efficiency.
    rays
        The rays at which to evaluate the efficiency.
    normal
        The vector normal to the surface.
    axis_angle
        The logical axis along which the angle of incidence varies.
        If :obj:`None`, the efficiency was measured at a single angle and is
        used at every angle of incidence, and
        ``measurement.inputs.direction`` is otherwise ignored.
        If not :obj:`None`, ``measurement.inputs.direction`` must be scalar
        angles of incidence, increasing along this axis, and
        ``measurement.inputs.wavelength`` may vary along it too, so that each
        angle can have its own wavelength samples.
    """

    wavelength = measurement.inputs.wavelength
    direction = na.as_named_array(measurement.inputs.direction)
    efficiency = measurement.outputs

    if axis_angle is None:
        if direction.size != 1:
            raise ValueError(
                "the efficiency was measured at more than one direction, "
                "specify the axis along which the angle of incidence varies"
            )
        if wavelength.ndim != 1:
            raise ValueError(
                f"wavelength must be one dimensional, got shape {wavelength.shape}"
            )
        return na.interp(
            x=rays.wavelength,
            xp=wavelength,
            fp=efficiency,
        )

    if not isinstance(direction, na.AbstractScalar):
        raise TypeError(
            "to interpolate over the angle of incidence, the directions must be "
            f"given as scalar angles of incidence, got {type(direction)}"
        )

    direction = direction.explicit

    if axis_angle not in direction.shape:
        raise ValueError(
            f"the angles of incidence, with shape {direction.shape}, "
            f"have no axis {axis_angle!r}"
        )

    unit = na.unit_normalized(direction)
    if not unit.is_equivalent(u.deg):
        raise ValueError(f"angles of incidence must be angles, got unit {unit}")

    step = (
        direction[{axis_angle: slice(1, None)}]
        - direction[{axis_angle: slice(None, ~0)}]
    )
    if not np.all(step > 0):
        raise ValueError(
            f"angles of incidence must increase along {axis_angle!r}, "
            f"got {direction}"
        )

    num = direction.shape[axis_angle]

    d = rays.direction
    cos_angle = np.abs(d @ normal) / (d.length * normal.length)
    angle = (np.arccos(np.minimum(cos_angle, 1)) << u.rad).to(unit).value

    result = 0
    for i in range(num):
        index = {axis_angle: i}

        wavelength_i = wavelength[index]
        if wavelength_i.ndim != 1:
            raise ValueError(
                "for each angle of incidence, wavelength must be one dimensional, "
                f"got shape {wavelength_i.shape}"
            )

        # linear interpolation in angle, as the weight this measurement
        # receives: np.interp of a one-hot vector is the hat function
        # centered on it, clamped at the ends of the measured range
        weight = na.interp(
            x=angle,
            xp=direction.value,
            fp=na.ScalarArray(np.eye(num)[i], axes=axis_angle),
            axis=axis_angle,
        )

        efficiency_i = na.interp(
            x=rays.wavelength,
            xp=wavelength_i,
            fp=efficiency[index],
        )

        result = result + weight * efficiency_i

    return result
