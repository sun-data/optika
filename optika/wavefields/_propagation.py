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
    chunk_size: int = 2**24,
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
        The samples where its
        :attr:`~optika.wavefields.AbstractWavefieldVectorArray.unvignetted`
        is :obj:`False` are left out.
    position
        The destination points at which to evaluate the diffracted wavefield.
    axis
        The one or more logical axes of the source wavefield over which the
        integral is evaluated.
    chunk_size
        The maximum number of terms of the discrete sum to evaluate at once,
        the product of the number of source points, the number of
        destination points, and the number of elements along the other axes
        of `wavefield` and `position` (wavelength, field angle, etc.).
        The destination points are split into chunks small enough to keep to
        it, but each chunk has at least one point, so the peak memory usage
        of this function is about :math:`16` bytes times the larger of
        `chunk_size` and the number of terms for one destination point.

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
    most `chunk_size` terms, and the outer product of source and destination
    points is never materialized in full.

    The normal at a source point may point either way, so the sign of the
    obliquity factor is chosen once, from the area-weighted mean of
    :math:`(\mathbf{r}_2 - \mathbf{r}_1) \cdot \hat{\mathbf{n}}` over the
    samples which are not vignetted, with :math:`\mathbf{r}_2` the centroid
    of the destination points.
    """
    result, _ = _rayleigh_sommerfeld(
        wavefield=wavefield,
        position=position,
        axis=axis,
        chunk_size=chunk_size,
    )
    return result


def _rayleigh_sommerfeld(
    wavefield: optika.wavefields.AbstractWavefieldVectorArray,
    position: na.AbstractCartesian3dVectorArray,
    axis: str | Sequence[str],
    chunk_size: int = 2**24,
    normal_destination: None | na.AbstractCartesian3dVectorArray = None,
) -> tuple[na.AbstractScalar, None | na.AbstractScalar]:
    r"""
    The diffraction integral of :func:`rayleigh_sommerfeld`, and the same
    integral with each term weighted by the cosine between the direction from
    the source point to the destination point and the given normal at the
    destination point, from which the flux of the wavefield through a surface
    with that normal follows.

    The gradient of the field at the destination point is :math:`i k` times
    the second integral, :math:`F`, to leading order in
    :math:`1 / (k r_{12})`, so the flux of the field through the surface is
    the real part of :math:`E^* F`, in the units of :math:`|E|^2`.

    Parameters
    ----------
    wavefield
        The source wavefield, as in :func:`rayleigh_sommerfeld`.
    position
        The destination points, as in :func:`rayleigh_sommerfeld`.
    axis
        The logical axes of the source wavefield to integrate over.
    chunk_size
        The maximum number of terms of each sum to evaluate at once, as in
        :func:`rayleigh_sommerfeld`.
        With a normal, each term takes twice the memory, so the chunks hold
        half as many terms.
    normal_destination
        The unit normal of the destination surface at each destination point,
        or :obj:`None` to leave out the second integral.
        Its orientation does not matter: the second integral is signed so
        that the flux of the light which travels from the source to the
        destination is positive, from the area-weighted mean of
        :math:`(\mathbf{r}_2 - \mathbf{r}_1) \cdot \hat{\mathbf{m}}`, with
        :math:`\mathbf{r}_2` the centroid of the destination points and
        :math:`\hat{\mathbf{m}}` the mean normal.

    Returns
    -------
    The two integrals, the second :obj:`None` if `normal_destination` is
    :obj:`None`.
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

    area = np.where(wavefield.unvignetted, wavefield.area, 0 * unit**2)
    area = area.to(unit**2).value
    mask = area > 0
    amp = np.where(mask, wavefield.amplitude * area, 0)

    # The sampling convention for surface normals allows them to point away
    # from the destination surface (e.g. after a reflection), so compute the
    # orientation of the obliquity factor once for the whole integral using
    # the centroid of the destination points, weighted by the area of the
    # samples so that those which are not part of the wavefield are left out.
    shape_position = position.shape
    if normal_destination is not None:
        shape_position = na.broadcast_shapes(shape_position, normal_destination.shape)

    axis_dst = tuple(ax for ax in shape_position if ax not in shape_src)
    axis_centroid = tuple(ax for ax in axis_dst if ax in position.shape)
    centroid_dst = position.mean(axis_centroid) if axis_centroid else position
    dot = ((centroid_dst - wavefield.position) @ normal).to(unit).value
    dot = na.as_named_array(np.where(mask, dot * area, 0))
    axis_sum = tuple(ax for ax in axis if ax in dot.shape)
    if axis_sum:
        dot = dot.sum(axis_sum)
    sign = np.sign(dot)

    # The orientation of the normal of the destination surface, from the mean
    # normal in the same way, which is constant on a flat surface.
    if normal_destination is not None:
        axis_mean = tuple(ax for ax in axis_dst if ax in normal_destination.shape)
        normal_mean = normal_destination
        if axis_mean:
            normal_mean = normal_destination.mean(axis_mean)
        dot = ((centroid_dst - wavefield.position) @ normal_mean).to(unit).value
        dot = na.as_named_array(np.where(mask, dot * area, 0))
        if axis_sum:
            dot = dot.sum(axis_sum)
        sign_normal = np.sign(dot)

    position_dst = position.to(unit).value

    shape_dst = {ax: shape_position[ax] for ax in axis_dst}

    shape_src_broadcast = {ax: num for ax, num in shape_src.items() if ax not in axis}
    shape_result = na.broadcast_shapes(shape_src_broadcast, shape_position)

    # The number of terms of the sum for each destination point.
    shape_other = {ax: num for ax, num in shape_position.items() if ax not in axis_dst}
    num_terms = int(np.prod(list(na.broadcast_shapes(shape_src, shape_other).values())))

    result = na.ScalarArray(
        ndarray=np.empty(tuple(shape_result.values()), dtype=complex),
        axes=tuple(shape_result),
    )
    result_normal = None
    if normal_destination is not None:
        result_normal = na.ScalarArray(
            ndarray=np.empty(tuple(shape_result.values()), dtype=complex),
            axes=tuple(shape_result),
        )

    # Greedily choose a chunk length along each destination axis (starting
    # from the last) so that the total number of terms per chunk is at most
    # `chunk_size`.
    chunks = dict()
    budget = max(chunk_size // num_terms // (1 if normal_destination is None else 2), 1)
    for ax in reversed(tuple(shape_dst)):
        chunks[ax] = min(shape_dst[ax], budget)
        budget = max(budget // shape_dst[ax], 1)

    starts = itertools.product(
        *[range(0, shape_dst[ax], chunks[ax]) for ax in shape_dst]
    )

    # The samples which are left out are selected away rather than multiplied
    # by zero, so that their positions and normals do not matter.
    # The mask is given to numexpr as a number, one or zero.
    expression = (
        "where("
        "mask > 0.5,"
        " amp"
        " * ((xd - xs) * nx + (yd - ys) * ny + (zd - zs) * nz)"
        " * exp(1j * k * sqrt((xd - xs)**2 + (yd - ys)**2 + (zd - zs)**2))"
        " / ((xd - xs)**2 + (yd - ys)**2 + (zd - zs)**2),"
        " 0j"
        ")"
    )

    # Each term of the second integral is a term of the first times the
    # cosine between the direction to the destination point and its normal.
    expression_normal = (
        "where("
        "mask > 0.5,"
        " term"
        " * ((xd - xs) * mx + (yd - ys) * my + (zd - zs) * mz)"
        " / sqrt((xd - xs)**2 + (yd - ys)**2 + (zd - zs)**2),"
        " 0j"
        ")"
    )

    for start in starts:
        index = {ax: slice(i, i + chunks[ax]) for ax, i in zip(shape_dst, start)}

        chunk = position_dst[index]

        term = na.numexpr.evaluate(
            ex=expression,
            local_dict=dict(
                mask=np.where(mask, 1.0, 0.0),
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

        if normal_destination is not None:
            chunk_normal = normal_destination[index]
            term = na.numexpr.evaluate(
                ex=expression_normal,
                local_dict=dict(
                    mask=np.where(mask, 1.0, 0.0),
                    term=term,
                    xs=xs,
                    ys=ys,
                    zs=zs,
                    mx=chunk_normal.x,
                    my=chunk_normal.y,
                    mz=chunk_normal.z,
                    xd=chunk.x,
                    yd=chunk.y,
                    zd=chunk.z,
                ),
            )
            result_normal[index] = term.sum(axis)

    factor = index_refraction / (1j * wavelength)
    result = result * sign * factor
    if result_normal is not None:
        result_normal = result_normal * sign * sign_normal * factor
    return result, result_normal
