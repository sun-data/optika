"""Scalar image-quality metrics of sampled point-spread functions."""

import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "encircled_energy_radius",
    "ensquared_energy",
    "fwhm",
    "mtf",
]


def _centroid(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray,
    axis: tuple[str, str],
) -> na.AbstractCartesian2dVectorArray:
    return (position * intensity).sum(axis) / intensity.sum(axis)


def _spacing(
    position: na.AbstractCartesian2dVectorArray,
    shape: dict[str, int],
    axis: tuple[str, str],
) -> na.Cartesian2dVectorArray:
    """
    The spacing of a regular grid of samples along each of its two axes.

    Parameters
    ----------
    position
        The position of each sample.
    shape
        The shape of the samples, broadcast against their intensity.
    axis
        The two logical axes of the grid of samples.
        :math:`x` must change only along the first and :math:`y` only along
        the second.

    Raises
    ------
    ValueError
        If the grid does not have at least two samples along each of `axis`,
        or the samples are not on a regular grid.
    """
    for ax in axis:
        if shape.get(ax, 0) < 2:
            raise ValueError(
                f"the samples must make a grid of at least two along each of "
                f"{axis}, but their shape is {shape}."
            )
    num_x = shape[axis[0]]
    num_y = shape[axis[1]]

    x = na.broadcast_to(position.x, shape)
    y = na.broadcast_to(position.y, shape)
    origin = {axis[0]: 0, axis[1]: 0}
    spacing_x = x[{axis[0]: 1, axis[1]: 0}] - x[origin]
    spacing_y = y[{axis[0]: 0, axis[1]: 1}] - y[origin]
    grid_x = x[origin] + spacing_x * na.arange(0, num_x, axis=axis[0])
    grid_y = y[origin] + spacing_y * na.arange(0, num_y, axis=axis[1])
    tolerance = 1e-6
    regular_x = np.all(np.abs(x - grid_x) <= tolerance * np.abs(spacing_x))
    regular_y = np.all(np.abs(y - grid_y) <= tolerance * np.abs(spacing_y))
    if not (regular_x and regular_y):
        raise ValueError(
            f"the samples must be on a regular grid, with x changing only "
            f"along {axis[0]!r} and y only along {axis[1]!r}."
        )

    return na.Cartesian2dVectorArray(spacing_x, spacing_y)


def _area_cell(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray,
    axis: tuple[str, str],
) -> na.AbstractScalar:
    """
    The area of a cell of a regular grid of samples.

    Parameters
    ----------
    intensity
        The intensity of the point-spread function at each sample.
    position
        The position of each sample.
    axis
        The two logical axes of the grid of samples.
    """
    shape = na.shape_broadcasted(intensity, position.x, position.y)
    spacing = _spacing(position, shape, axis)
    return np.abs(spacing.x * spacing.y)


def _dimensionless(a: na.AbstractScalar) -> na.AbstractScalar:
    """
    Strip the unit of a dimensionless quantity.

    Parameters
    ----------
    a
        A dimensionless quantity, or an array without a unit.
    """
    if na.unit(a) is not None:
        a = a.to(u.dimensionless_unscaled).value
    return a


def encircled_energy_radius(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
    fraction: float = 0.5,
    normalize: bool = True,
) -> u.Quantity | na.AbstractScalar:
    """
    The radius from the intensity-weighted centroid which encloses a given
    fraction of the total energy of a sampled point-spread function.

    The radius is that of the first sample, in order of distance from the
    centroid, at which the enclosed energy reaches `fraction`, or
    not-a-number if it does not reach it on the grid.

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
    normalize
        If :obj:`True` (the default), the energy is a fraction of the energy
        on the grid.
        If :obj:`False`, `intensity` must be the intensity per unit area,
        normalized to the power of the whole image, like the outputs of
        :meth:`optika.systems.SequentialSystem.psf`, and the energy is a
        fraction of the power of the whole image, so the light which lands
        off the grid counts against it.
        The samples must then be on a regular grid, like those of
        :func:`mtf`.
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy
    if not normalize:
        intensity = intensity * _area_cell(intensity, position, axis)
    centroid = _centroid(intensity, position, axis)
    radius = (position - centroid).length

    shape = na.shape_broadcasted(intensity, radius)
    axis_flat = f"_{axis[0]}_{axis[1]}"
    radius = na.broadcast_to(radius, shape).combine_axes(axis, axis_flat)
    intensity = na.broadcast_to(intensity, shape).combine_axes(axis, axis_flat)

    order = np.argsort(radius, axis=axis_flat)
    radius = radius[order]
    energy = np.cumsum(intensity[order], axis=axis_flat)
    if normalize:
        energy = energy / energy[{axis_flat: ~0}]
    energy = _dimensionless(energy)

    num = radius.shape[axis_flat]
    index = na.searchsorted(energy, fraction, axis=axis_flat)
    reached = index < num
    index = np.minimum(index, num - 1)
    result = radius[{axis_flat: index}]
    result = np.where(reached, result, np.nan * na.unit_normalized(result))

    if not result.shape:
        result = result.ndarray

    return result


def ensquared_energy(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
    width: u.Quantity | na.AbstractScalar,
    normalize: bool = True,
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
    normalize
        If :obj:`True` (the default), the energy is a fraction of the energy
        on the grid.
        If :obj:`False`, `intensity` must be the intensity per unit area,
        normalized to the power of the whole image, like the outputs of
        :meth:`optika.systems.SequentialSystem.psf`, and the energy is a
        fraction of the power of the whole image, so the light which lands
        off the grid counts against it.
        The samples must then be on a regular grid, like those of
        :func:`mtf`.
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy
    centroid = _centroid(intensity, position, axis)
    offset = position - centroid

    inside = (np.abs(offset.x) < width / 2) & (np.abs(offset.y) < width / 2)

    result = (intensity * inside).sum(axis)
    if normalize:
        result = result / intensity.sum(axis)
    else:
        result = result * _area_cell(intensity, position, axis)

    return _dimensionless(result)


def fwhm(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
) -> na.AbstractCartesian2dVectorArray:
    """
    The full width at half maximum of a sampled point-spread function along
    each axis, measured from the cuts through the brightest sample.

    On each side of the peak, the cut is followed outward to the first sample
    below half of the peak, and the position of half of the peak is
    interpolated linearly between it and the sample before it.
    The width is not-a-number if the cut does not fall below half of the peak
    on both sides within the samples.

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

    index_peak = np.argmax(intensity, axis=axis)
    index_x = {axis[1]: index_peak[axis[1]]}
    index_y = {axis[0]: index_peak[axis[0]]}

    result = na.Cartesian2dVectorArray(
        x=_fwhm_of_cut(
            cut=intensity[index_x],
            coordinate=x[index_x],
            index_peak=index_peak[axis[0]],
            axis=axis[0],
        ),
        y=_fwhm_of_cut(
            cut=intensity[index_y],
            coordinate=y[index_y],
            index_peak=index_peak[axis[1]],
            axis=axis[1],
        ),
    )

    if not result.shape:
        result = na.Cartesian2dVectorArray(
            x=result.x.ndarray,
            y=result.y.ndarray,
        )

    return result


def _fwhm_of_cut(
    cut: na.AbstractScalar,
    coordinate: na.AbstractScalar,
    index_peak: na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    """
    The full width at half maximum of one-dimensional cuts through a peak.

    Parameters
    ----------
    cut
        The intensity along each cut.
    coordinate
        The position of each sample along each cut.
    index_peak
        The index of the peak of each cut along `axis`.
    axis
        The logical axis along each cut.
    """
    num = cut.shape[axis]
    index = na.arange(0, num, axis=axis)
    half = cut[{axis: index_peak}] / 2
    below = cut < half

    index_right = np.where(below & (index > index_peak), index, num).min(axis)
    index_left = np.where(below & (index < index_peak), index, -1).max(axis)
    found = (index_right < num) & (index_left >= 0)
    index_right = np.minimum(index_right, num - 1)
    index_left = np.maximum(index_left, 0)

    def crossing(
        index_inside: na.AbstractScalar,
        index_outside: na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The position of half of the peak, interpolated between a sample
        above it and the next sample outward, below it.

        Parameters
        ----------
        index_inside
            The index of the sample above half of the peak.
        index_outside
            The index of the sample below half of the peak.
        """
        cut_inside = cut[{axis: index_inside}]
        cut_outside = cut[{axis: index_outside}]
        coordinate_inside = coordinate[{axis: index_inside}]
        coordinate_outside = coordinate[{axis: index_outside}]
        step = cut_outside - cut_inside
        step = np.where(step != 0, step, 1 * na.unit_normalized(step))
        fraction = (half - cut_inside) / step
        return coordinate_inside + fraction * (coordinate_outside - coordinate_inside)

    right = crossing(np.maximum(index_right - 1, 0), index_right)
    left = crossing(np.minimum(index_left + 1, num - 1), index_left)

    nan = np.nan * na.unit_normalized(coordinate)
    return np.where(found, right - left, nan)


def mtf(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
    normalize: bool = True,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
    r"""
    The modulation transfer function of a sampled point-spread function.

    This is the modulus of the two-dimensional Fourier transform of the
    intensity, normalized to one at zero frequency, or to the power of the
    whole image, on a grid of spatial frequencies centered on zero.

    The samples must be on a regular grid, with :math:`x` changing only along
    the first of `axis` and :math:`y` only along the second, like the grid of
    :meth:`optika.systems.SequentialSystem.psf`.
    The frequencies are spaced by one over the width of the grid and reach
    one over twice the spacing of the samples, so the grid must be fine
    enough to resolve the point-spread function.
    Normalized to one at zero frequency, the MTF is that of the light on the
    grid, so the grid must also be wide enough to hold nearly all of the
    light.
    Normalized to the power of the whole image, with ``normalize=False``, the
    light which lands off the grid lowers the MTF at every frequency but
    zero by the fraction of the power it carries, which is accurate as long
    as that light is spread over a region much wider than the grid, as the
    light which the figure errors of the surfaces scatter far from the image
    is.

    A point-spread function from :meth:`~optika.systems.SequentialSystem.psf`
    is computed from a finite grid of rays, which aliases it at a distance
    from the image set by the spacing of the rays (see the notes of
    :meth:`~optika.systems.SequentialSystem.wavefield`), so check that the
    MTF has converged by increasing the number of rays.

    Parameters
    ----------
    intensity
        The intensity of the point-spread function at each sample.
    position
        The position of each sample.
    axis
        The two logical axes of the grid of samples.
        The frequencies are along the same two axes.
    normalize
        If :obj:`True` (the default), the MTF is normalized to one at zero
        frequency.
        If :obj:`False`, `intensity` must be the intensity per unit area,
        normalized to the power of the whole image, like the outputs of
        :meth:`optika.systems.SequentialSystem.psf`, and the MTF at zero
        frequency is the fraction of the power which lands on the grid.

    Raises
    ------
    ValueError
        If the samples are not on a regular grid along `axis`.

    Examples
    --------
    The MTF of a Gaussian point-spread function with a standard deviation
    :math:`\sigma` is :math:`\exp(-2 \pi^2 \sigma^2 f^2)`.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        sigma = 5 * u.um
        num = 201
        position = na.Cartesian2dVectorLinearSpace(
            start=-40 * u.um,
            stop=40 * u.um,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=num,
        ).explicit
        intensity = np.exp(-np.square(position.length / sigma) / 2)

        mtf = optika.wavefields.mtf(
            intensity=intensity,
            position=position,
            axis=("x", "y"),
        )

        cut = mtf[dict(y=num // 2)]
        frequency = cut.inputs.x.to(1 / u.mm)
        expected = np.exp(-2 * np.square(np.pi * sigma * frequency))

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(frequency, cut.outputs, ax=ax, label="mtf()")
            na.plt.plot(frequency, expected, ax=ax, linestyle="--", label="exact")
            ax.set_xlim(0, 100)
            ax.set_xlabel(f"spatial frequency ({ax.get_xlabel()})")
            ax.set_ylabel("MTF")
            ax.legend()
    """
    if isinstance(position, na.AbstractCartesian3dVectorArray):
        position = position.xy

    shape = na.shape_broadcasted(intensity, position.x, position.y)
    spacing = _spacing(position, shape, axis)
    spacing_x = spacing.x
    spacing_y = spacing.y
    num_x = shape[axis[0]]
    num_y = shape[axis[1]]

    intensity = na.broadcast_to(intensity, shape)
    if normalize:
        intensity = intensity / intensity.sum(axis)
    else:
        intensity = intensity * np.abs(spacing_x * spacing_y)
    intensity = _dimensionless(intensity)

    spectrum = np.fft.fftn(intensity, axes={axis[0]: axis[0], axis[1]: axis[1]})
    modulus = np.abs(spectrum)
    modulus = np.roll(modulus, num_x // 2, axis=axis[0])
    modulus = np.roll(modulus, num_y // 2, axis=axis[1])

    index_x = na.arange(0, num_x, axis=axis[0]) - num_x // 2
    index_y = na.arange(0, num_y, axis=axis[1]) - num_y // 2

    return na.FunctionArray(
        inputs=na.Cartesian2dVectorArray(
            x=index_x / (num_x * spacing_x),
            y=index_y / (num_y * spacing_y),
        ),
        outputs=modulus,
    )
