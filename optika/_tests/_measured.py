"""
Measurements and a reference interpolation shared by the tests of measured
efficiencies.
"""

import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "wavelength_channel",
    "efficiency_channel",
    "interp_each",
]

_wavelength = na.linspace(100, 300, axis="wavelength", num=11) * u.AA
"""The wavelengths of the first channel."""

wavelength_channel = _wavelength + na.linspace(0, 5, axis="channel", num=2) * u.AA
"""Wavelength samples which differ for each of two channels."""

efficiency_channel = np.exp(-np.square((_wavelength - 200 * u.AA) / (100 * u.AA)) / 2)
"""
An efficiency which is the same at each sample of every channel, and so is
different at the same wavelength in each, since the samples are not.
"""


def interp_each(
    x: u.Quantity | na.AbstractScalar,
    xp: na.AbstractScalar,
    fp: na.AbstractScalar,
    axis: str,
    axis_each: str,
) -> na.AbstractScalar:
    """
    Interpolate along `axis` one index of `axis_each` at a time.

    A reference for interpolations which handle the other axes by
    broadcasting, since this one hands :func:`named_arrays.interp` a single
    set of samples at a time.

    Parameters
    ----------
    x
        The points to interpolate at.
    xp
        The sample points.
    fp
        The values at the sample points.
    axis
        The logical axis to interpolate along.
    axis_each
        The logical axis to interpolate one index of at a time.
    """
    shape = na.broadcast_shapes(na.shape(xp), na.shape(fp))
    xp = na.broadcast_to(xp, shape)
    fp = na.broadcast_to(fp, shape)
    return na.stack(
        [
            na.interp(x, xp[{axis_each: i}], fp[{axis_each: i}], axis=axis)
            for i in range(shape[axis_each])
        ],
        axis=axis_each,
    )
