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


def mtf(
    intensity: na.AbstractScalar,
    position: na.AbstractCartesian2dVectorArray | na.AbstractCartesian3dVectorArray,
    axis: tuple[str, str],
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.AbstractScalar]:
    r"""
    The modulation transfer function of a sampled point-spread function.

    This is the modulus of the two-dimensional Fourier transform of the
    intensity, normalized to one at zero frequency, on a grid of spatial
    frequencies centered on zero.

    The samples must be on a regular grid, with :math:`x` changing only along
    the first of `axis` and :math:`y` only along the second, like the grid of
    :meth:`optika.systems.SequentialSystem.psf`.
    The frequencies are spaced by one over the width of the grid and reach
    one over twice the spacing of the samples, so the grid must be wide
    enough to hold nearly all of the light, and fine enough to resolve the
    point-spread function.

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

    intensity = na.broadcast_to(intensity, shape)
    intensity = intensity / intensity.sum(axis)
    if na.unit(intensity) is not None:
        intensity = intensity.to(u.dimensionless_unscaled).value

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
