from __future__ import annotations
from typing import Sequence
import itertools
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "rayleigh_sommerfeld",
]


def rayleigh_sommerfeld(
    wavefield: optika.wavefields.AbstractWavefieldVectorArray,
    position: na.AbstractCartesian3dVectorArray,
    axis: str | Sequence[str],
    chunk_size: int = 1024,
) -> na.AbstractScalar:
    r"""
    Evaluate the Rayleigh-Sommerfeld diffraction integral of the first kind
    to find the complex amplitude of the given wavefield at the given
    destination points.

    Parameters
    ----------
    wavefield
        The source wavefield, sampled at a discrete set of points on a
        surface, expressed in the same coordinate system as `position`.
    position
        The destination points at which to evaluate the diffracted wavefield.
    axis
        The one or more logical axes of the source wavefield over which the
        integral is evaluated.
    chunk_size
        The maximum number of destination points to consider simultaneously.
        The peak memory usage of this function is approximately
        :math:`16 \, N_\text{chunk} N_\text{source} N_\text{broadcast}`
        bytes, where :math:`N_\text{broadcast}` is the number of elements
        along the axes shared by `wavefield` and `position`
        (wavelength, field angle, etc.).

    Notes
    -----
    The Rayleigh-Sommerfeld diffraction integral of the first kind is

    .. math::

        E(\mathbf{r}_2) = \frac{n}{i \lambda} \int_S E(\mathbf{r}_1)
            \frac{e^{i k r_{12}}}{r_{12}} \cos \theta \, dA,

    where :math:`E(\mathbf{r}_1)` is the source wavefield,
    :math:`r_{12} = |\mathbf{r}_2 - \mathbf{r}_1|` is the distance between
    the source and destination points,
    :math:`k = 2 \pi n / \lambda` is the wavenumber in the current medium,
    :math:`n` is the index of refraction,
    :math:`\lambda` is the vacuum wavelength,
    and :math:`\theta` is the angle between :math:`\mathbf{r}_2 - \mathbf{r}_1`
    and the surface normal at the source point.

    The integral is evaluated as a discrete sum over the source samples.
    To bound memory usage, the destination points are split into chunks of at
    most `chunk_size` points, and the outer product of source and destination
    points is never materialized in full.
    """
    if isinstance(axis, str):
        axis = (axis,)
    axis = tuple(axis)

    shape_src = wavefield.shape
    for ax in axis:
        if ax not in shape_src:
            raise ValueError(
                f"axis {ax!r} is not in the wavefield's shape, {shape_src}."
            )

    unit = u.mm

    position_src = wavefield.position.to(unit).value
    xs = position_src.x
    ys = position_src.y
    zs = position_src.z

    normal = wavefield.normal
    nx = normal.x
    ny = normal.y
    nz = normal.z

    wavelength = na.as_named_array(wavefield.wavelength).to(unit).value
    index_refraction = na.as_named_array(wavefield.index_refraction)

    k = 2 * np.pi * index_refraction / wavelength

    amp = wavefield.amplitude * wavefield.area.to(unit**2).value

    # The sampling convention for surface normals allows them to point away
    # from the destination surface (e.g. after a reflection), so compute the
    # orientation of the obliquity factor once for the whole integral using
    # the centroid of the destination points.
    axis_dst = tuple(ax for ax in position.shape if ax not in shape_src)
    centroid_dst = position.mean(axis_dst) if axis_dst else position
    dot = (centroid_dst - wavefield.position) @ normal
    axis_mean = tuple(ax for ax in axis if ax in na.shape(dot))
    if axis_mean:
        dot = dot.mean(axis_mean)
    sign = np.sign(na.as_named_array(dot).value)

    position_dst = position.to(unit).value

    shape_dst = {ax: position.shape[ax] for ax in axis_dst}

    shape_src_broadcast = {
        ax: num for ax, num in shape_src.items() if ax not in axis
    }
    shape_result = na.broadcast_shapes(shape_src_broadcast, position.shape)

    result = na.ScalarArray(
        ndarray=np.empty(tuple(shape_result.values()), dtype=complex),
        axes=tuple(shape_result),
    )

    # Greedily choose a chunk length along each destination axis (starting
    # from the last) so that the total number of destination points per chunk
    # is at most `chunk_size`.
    chunks = dict()
    budget = max(chunk_size, 1)
    for ax in reversed(tuple(shape_dst)):
        chunks[ax] = min(shape_dst[ax], budget)
        budget = max(budget // shape_dst[ax], 1)

    starts = itertools.product(
        *[range(0, shape_dst[ax], chunks[ax]) for ax in shape_dst]
    )

    expression = (
        "amp"
        " * ((xd - xs) * nx + (yd - ys) * ny + (zd - zs) * nz)"
        " * exp(1j * k * sqrt((xd - xs)**2 + (yd - ys)**2 + (zd - zs)**2))"
        " / ((xd - xs)**2 + (yd - ys)**2 + (zd - zs)**2)"
    )

    for start in starts:
        index = {
            ax: slice(i, i + chunks[ax])
            for ax, i in zip(shape_dst, start)
        }

        chunk = position_dst[index]

        term = na.numexpr.evaluate(
            ex=expression,
            local_dict=dict(
                amp=amp,
                xs=xs,
                ys=ys,
                zs=zs,
                nx=nx,
                ny=ny,
                nz=nz,
                k=k,
                xd=chunk.x,
                yd=chunk.y,
                zd=chunk.z,
            ),
        )

        result[index] = term.sum(axis)

    return result * sign * index_refraction / (1j * wavelength)
