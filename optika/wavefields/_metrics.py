"""Scalar image-quality metrics of sampled point-spread functions."""

import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "encircled_energy_radius",
    "ensquared_energy",
    "fwhm",
]


def _centroid(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray,
    axis: tuple[str, str],
) -> na.AbstractCartesian2dVectorArray:
    return (position * intensity).sum(axis) / intensity.sum(axis)


def encircled_energy_radius(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
    fraction: float = 0.5,
) -> u.Quantity | na.AbstractScalar:
    """
    The radius from the intensity-weighted centroid which encloses a given
    fraction of the total energy of a sampled point-spread function.

    Parameters
    ----------
    intensity
        The intensity of the point-spread function at each sample.
    position
        The position of each sample.
    axis
        The two logical axes of the grid of samples.
    fraction
        The fraction of the total energy to enclose.
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy
    centroid = _centroid(intensity, position, axis)
    radius = (position - centroid).length

    shape = na.shape_broadcasted(intensity, radius)
    intensity = na.broadcast_to(intensity, shape)
    radius = na.broadcast_to(radius, shape)

    shape_outer = {ax: n for ax, n in shape.items() if ax not in axis}
    unit = na.unit(radius)
    result = na.ScalarArray(
        ndarray=np.empty(tuple(shape_outer.values())),
        axes=tuple(shape_outer),
    )
    if unit is not None:
        result = result * unit

    for index in na.ndindex(shape_outer):
        r = radius[index].ndarray.ravel()
        w = intensity[index].ndarray.ravel()
        order = np.argsort(r)
        energy = np.cumsum(w[order])
        energy = energy / energy[~0]
        result[index] = r[order][np.searchsorted(energy, fraction)]

    if not shape_outer:
        result = result.ndarray

    return result


def ensquared_energy(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
    width: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The fraction of the total energy of a sampled point-spread function
    inside a square of the given width centered on the intensity-weighted
    centroid.

    This is useful for checking if an optical system is pixel-limited:
    if the ensquared energy in one pixel is large, the detector
    (and not the optics) sets the resolution.

    Parameters
    ----------
    intensity
        The intensity of the point-spread function at each sample.
    position
        The position of each sample.
    axis
        The two logical axes of the grid of samples.
    width
        The width of the square.
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy
    centroid = _centroid(intensity, position, axis)
    offset = position - centroid

    inside = (np.abs(offset.x) < width / 2) & (np.abs(offset.y) < width / 2)

    return (intensity * inside).sum(axis) / intensity.sum(axis)


def fwhm(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
) -> na.AbstractCartesian2dVectorArray:
    """
    The full width at half maximum of a sampled point-spread function along
    each axis, measured from the cuts through the brightest sample.

    Parameters
    ----------
    intensity
        The intensity of the point-spread function at each sample.
    position
        The position of each sample.
    axis
        The two logical axes of the grid of samples.
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy

    shape = na.shape_broadcasted(intensity, position.x, position.y)
    intensity = na.broadcast_to(intensity, shape)
    x = na.broadcast_to(position.x, shape)
    y = na.broadcast_to(position.y, shape)

    shape_outer = {ax: n for ax, n in shape.items() if ax not in axis}
    unit = na.unit(x)

    result = na.Cartesian2dVectorArray(
        x=na.ScalarArray(
            ndarray=np.empty(tuple(shape_outer.values())),
            axes=tuple(shape_outer),
        ),
        y=na.ScalarArray(
            ndarray=np.empty(tuple(shape_outer.values())),
            axes=tuple(shape_outer),
        ),
    )
    if unit is not None:
        result = result * unit

    def _fwhm_of_cut(cut: np.ndarray, coordinate: np.ndarray) -> float:
        index_peak = int(np.argmax(cut))
        half = cut[index_peak] / 2
        left = np.interp(
            half,
            cut[: index_peak + 1],
            coordinate[: index_peak + 1],
        )
        right = np.interp(
            half,
            cut[index_peak:][::-1],
            coordinate[index_peak:][::-1],
        )
        return right - left

    for index in na.ndindex(shape_outer):
        cut = intensity[index]
        index_peak = np.unravel_index(
            np.argmax(cut.ndarray),
            cut.ndarray.shape,
        )
        index_peak = dict(zip(cut.axes, index_peak))

        cut_x = cut[{axis[1]: index_peak[axis[1]]}].ndarray
        coordinate_x = x[index][{axis[1]: index_peak[axis[1]]}].ndarray
        cut_y = cut[{axis[0]: index_peak[axis[0]]}].ndarray
        coordinate_y = y[index][{axis[0]: index_peak[axis[0]]}].ndarray

        result.x[index] = _fwhm_of_cut(cut_x, coordinate_x)
        result.y[index] = _fwhm_of_cut(cut_y, coordinate_y)

    if not shape_outer:
        result = na.Cartesian2dVectorArray(
            x=result.x.ndarray,
            y=result.y.ndarray,
        )

    return result
