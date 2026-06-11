import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


def _aperture_wavefield(
    half_width: u.Quantity,
    wavelength: u.Quantity,
    num: int,
    amplitude_function,
    seed: int = 42,
) -> optika.wavefields.WavefieldVectorArray:
    """A flat, square aperture at the origin with the given amplitude function."""
    grid = na.Cartesian2dVectorStratifiedRandomSpace(
        start=-half_width,
        stop=half_width,
        axis=na.Cartesian2dVectorArray("sx", "sy"),
        num=num,
        seed=seed,
    ).explicit

    return optika.wavefields.WavefieldVectorArray(
        wavelength=wavelength,
        position=na.Cartesian3dVectorArray(grid.x, grid.y, 0 * half_width),
        amplitude=amplitude_function(grid),
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        area=np.square(2 * half_width / num),
    )


def _focusing_phase(
    grid: na.AbstractCartesian2dVectorArray,
    wavelength: u.Quantity,
    focal_length: u.Quantity,
) -> na.AbstractScalar:
    """The complex phase of a perfect lens with the given focal length."""
    distance = np.sqrt(np.square(grid.length) + np.square(focal_length))
    phase = (distance / wavelength).to_value(u.dimensionless_unscaled)
    return np.exp(-2j * np.pi * phase)


def test_airy_pattern():
    """
    A uniformly-illuminated circular aperture with a perfect focusing phase
    should produce an Airy pattern with its first intensity zero at a radius
    of :math:`1.22 \\lambda f / D`.
    """
    wavelength = 500 * u.nm
    radius = 5 * u.mm
    focal_length = 1000 * u.mm

    def amplitude(grid: na.AbstractCartesian2dVectorArray) -> na.AbstractScalar:
        phase = _focusing_phase(grid, wavelength, focal_length)
        return np.where(grid.length <= radius, phase, 0)

    wavefield = _aperture_wavefield(
        half_width=radius,
        wavelength=wavelength,
        num=101,
        amplitude_function=amplitude,
    )

    radius_airy = (1.22 * wavelength * focal_length / (2 * radius)).to(u.mm)

    x = na.linspace(0, 2 * radius_airy, axis="dx", num=121)
    position = na.Cartesian3dVectorArray(x, 0 * u.mm, focal_length)

    amplitude_detector = optika.wavefields.rayleigh_sommerfeld(
        wavefield=wavefield,
        position=position,
        axis=("sx", "sy"),
    )

    intensity = np.square(np.abs(amplitude_detector))
    intensity = (intensity / intensity.max()).ndarray

    minima = [
        i
        for i in range(1, len(intensity) - 1)
        if intensity[i] < intensity[i - 1] and intensity[i] < intensity[i + 1]
    ]
    assert minima

    radius_zero = x.ndarray[minima[0]]
    assert np.abs(radius_zero - radius_airy) < 0.02 * radius_airy

    # The first zero should be dark and the peak should be on axis.
    assert intensity[minima[0]] < 1e-3
    assert np.argmax(intensity) == 0


def test_gaussian_apodization():
    """
    A Gaussian-apodized aperture with waist :math:`w` and a perfect focusing
    phase should produce a Gaussian point-spread function with standard
    deviation :math:`\\sigma = \\lambda f / (2 \\pi w)`.
    """
    wavelength = 500 * u.nm
    waist = 1 * u.mm
    focal_length = 1000 * u.mm

    def amplitude(grid: na.AbstractCartesian2dVectorArray) -> na.AbstractScalar:
        apodization = np.exp(
            -np.square(grid.length / waist).to_value(u.dimensionless_unscaled)
        )
        return apodization * _focusing_phase(grid, wavelength, focal_length)

    wavefield = _aperture_wavefield(
        half_width=3 * waist,
        wavelength=wavelength,
        num=101,
        amplitude_function=amplitude,
    )

    sigma = (wavelength * focal_length / (2 * np.pi * waist)).to(u.mm)

    x = na.linspace(-4 * sigma, 4 * sigma, axis="dx", num=161)
    position = na.Cartesian3dVectorArray(x, 0 * u.mm, focal_length)

    amplitude_detector = optika.wavefields.rayleigh_sommerfeld(
        wavefield=wavefield,
        position=position,
        axis=("sx", "sy"),
    )

    intensity = np.square(np.abs(amplitude_detector))

    sigma_measured = np.sqrt(
        (intensity * np.square(x)).sum() / intensity.sum()
    )

    assert np.abs(sigma_measured - sigma) < 0.03 * sigma


def test_energy_conservation():
    """
    The total energy on a detector plane that captures essentially all of
    the diffracted light should equal the total energy of the source.
    """
    wavelength = 500 * u.nm
    waist = 1 * u.mm
    distance = 2000 * u.mm

    def amplitude(grid: na.AbstractCartesian2dVectorArray) -> na.AbstractScalar:
        return (
            np.exp(-np.square(grid.length / waist).to_value(u.dimensionless_unscaled))
            + 0j
        )

    wavefield = _aperture_wavefield(
        half_width=3 * waist,
        wavelength=wavelength,
        num=151,
        amplitude_function=amplitude,
        seed=7,
    )

    energy_source = (np.square(np.abs(wavefield.amplitude)) * wavefield.area).sum()

    half_width_detector = 6 * u.mm
    num_detector = 151
    grid_detector = na.Cartesian2dVectorLinearSpace(
        start=-half_width_detector,
        stop=half_width_detector,
        axis=na.Cartesian2dVectorArray("dx", "dy"),
        num=num_detector,
    ).explicit
    area_detector = np.square(2 * half_width_detector / num_detector)

    amplitude_detector = optika.wavefields.rayleigh_sommerfeld(
        wavefield=wavefield,
        position=na.Cartesian3dVectorArray(
            x=grid_detector.x,
            y=grid_detector.y,
            z=distance,
        ),
        axis=("sx", "sy"),
    )

    energy_detector = (
        np.square(np.abs(amplitude_detector)) * area_detector
    ).sum()

    assert np.abs(energy_detector / energy_source - 1) < 0.01


def test_chunk_invariance():
    """The result should not depend on the chunk size."""
    wavelength = 500 * u.nm
    radius = 1 * u.mm

    def amplitude(grid: na.AbstractCartesian2dVectorArray) -> na.AbstractScalar:
        return np.where(grid.length <= radius, 1 + 0j, 0)

    wavefield = _aperture_wavefield(
        half_width=radius,
        wavelength=wavelength,
        num=21,
        amplitude_function=amplitude,
    )

    position = na.Cartesian3dVectorLinearSpace(
        start=na.Cartesian3dVectorArray(-1, -1, 99) * u.mm,
        stop=na.Cartesian3dVectorArray(1, 1, 101) * u.mm,
        axis=na.Cartesian3dVectorArray("dx", "dy", "dz"),
        num=na.Cartesian3dVectorArray(7, 6, 5),
    ).explicit

    results = [
        optika.wavefields.rayleigh_sommerfeld(
            wavefield=wavefield,
            position=position,
            axis=("sx", "sy"),
            chunk_size=chunk_size,
        )
        for chunk_size in (1, 17, 10**6)
    ]

    assert np.allclose(results[0], results[1], rtol=1e-9)
    assert np.allclose(results[0], results[2], rtol=1e-9)


def test_invalid_axis():
    wavefield = optika.wavefields.WavefieldVectorArray(
        wavelength=500 * u.nm,
        position=na.Cartesian3dVectorArray(0, 0, 0) * u.mm,
        normal=na.Cartesian3dVectorArray(0, 0, -1),
        area=1 * u.mm**2,
    )
    with pytest.raises(ValueError):
        optika.wavefields.rayleigh_sommerfeld(
            wavefield=wavefield,
            position=na.Cartesian3dVectorArray(0, 0, 1) * u.mm,
            axis="missing",
        )
