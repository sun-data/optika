"""Tests of the physical-optics propagation methods of SequentialSystem."""

import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


def _telescope(
    sag_primary: None | optika.sags.AbstractSag = None,
    z_sensor: u.Quantity = 0 * u.mm,
) -> optika.systems.SequentialSystem:
    """An on-axis parabolic telescope with focal length 200 mm and D=40 mm."""
    if sag_primary is None:
        sag_primary = optika.sags.ParabolicSag(focal_length=-200 * u.mm)

    primary = optika.surfaces.Surface(
        name="primary",
        sag=sag_primary,
        aperture=optika.apertures.CircularAperture(20 * u.mm),
        material=optika.materials.Mirror(),
        is_pupil_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(z=200 * u.mm),
    )
    sensor = optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=20 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        num_pixel=na.Cartesian2dVectorArray(128, 128),
        timedelta_exposure=1 * u.s,
        is_field_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(z=z_sensor),
    )
    return optika.systems.SequentialSystem(
        surfaces=[primary],
        sensor=sensor,
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=500 * u.nm,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
                centers=True,
            ),
        ),
    )


def _rms_radius(
    psf: na.FunctionArray,
    axis: tuple[str, str] = ("wavefield_x", "wavefield_y"),
) -> u.Quantity:
    """The intensity-weighted RMS radius of a point-spread function."""
    intensity = psf.outputs
    position = psf.inputs.position
    centroid = (position * intensity).sum(axis) / intensity.sum(axis)
    r2 = np.square((position - centroid).length)
    return np.sqrt((r2 * intensity).sum(axis) / intensity.sum(axis))


def test_psf_airy():
    """
    The on-axis Huygens PSF of an ideal parabolic telescope should be an
    Airy pattern with its first zero at a radius of
    :math:`1.22 \\lambda f / D`.
    """
    system = _telescope()

    psf = system.psf(
        field=na.Cartesian2dVectorArray(0, 0),
        width_sensor=0.02 * u.mm,
        num=101**2,
        num_sensor=51,
        seed=42,
    )

    intensity = psf.outputs

    assert np.allclose(intensity.sum(), 1)

    index_peak = np.unravel_index(
        np.argmax(intensity.ndarray),
        intensity.ndarray.shape,
    )
    index_center = (51 // 2, 51 // 2)
    assert index_peak == index_center

    cut = intensity[dict(wavefield_y=51 // 2)].ndarray
    cut = cut / cut.max()
    minima = [
        i
        for i in range(51 // 2 + 1, 50)
        if cut[i] < cut[i - 1] and cut[i] < cut[i + 1]
    ]
    assert minima

    x = psf.inputs.position.x
    if "wavefield_y" in x.shape:
        x = x[dict(wavefield_y=51 // 2)]
    x = x.ndarray
    radius_zero = np.abs(x[minima[0]] - x[51 // 2])

    radius_airy = (1.22 * 500 * u.nm * 200 / 40).to(u.mm)

    # The grid resolution limits how precisely the zero can be located.
    resolution = 0.02 * u.mm / 50
    assert np.abs(radius_zero - radius_airy) < resolution

    assert cut[minima[0]] < 1e-2


def test_wavefield_centroid_matches_raytrace():
    """
    For an off-axis field angle, the centroid of the Huygens PSF should match
    the centroid of a geometric raytrace to within a fraction of the Airy
    radius.
    """
    system = _telescope()
    field = na.Cartesian2dVectorArray(0.5, 0)

    wavefield = system.wavefield(
        field=field,
        width_sensor=0.02 * u.mm,
        num=101**2,
        num_sensor=51,
        seed=11,
    )
    intensity = np.square(np.abs(wavefield.outputs.amplitude))
    position = wavefield.outputs.position.xy
    centroid = (position * intensity).sum() / intensity.sum()

    rayfunction = system.rayfunction(field=field)
    where = rayfunction.outputs.unvignetted
    position_rays = rayfunction.outputs.position.xy
    centroid_rays = (position_rays * where).sum() / where.sum()

    radius_airy = (1.22 * 500 * u.nm * 200 / 40).to(u.mm)

    assert np.abs(centroid.x - centroid_rays.x) < radius_airy / 2
    assert np.abs(centroid.y - centroid_rays.y) < radius_airy / 2


def test_psf_zernike_defocus():
    """
    Adding a Zernike defocus term to the primary mirror shifts the focus by
    an analytic amount: the PSF at the shifted sensor position should be much
    sharper than at the nominal position.
    """
    focal_length = 200 * u.mm
    radius = 20 * u.mm
    c = 0.5 * u.um

    sag = optika.sags.ZernikeSag(
        sag=optika.sags.ParabolicSag(focal_length=-focal_length),
        coefficients=[0 * u.um, 0 * u.um, 0 * u.um, -c],
        radius=radius,
    )

    focal_length_perturbed = 1 / (
        1 / focal_length + 8 * np.sqrt(3) * c / np.square(radius)
    )
    z_focus = focal_length - focal_length_perturbed

    kwargs = dict(
        field=na.Cartesian2dVectorArray(0, 0),
        width_sensor=0.2 * u.mm,
        num=101**2,
        num_sensor=51,
        seed=3,
    )

    psf_nominal = _telescope(sag_primary=sag).psf(**kwargs)
    psf_shifted = _telescope(sag_primary=sag, z_sensor=z_focus).psf(**kwargs)

    # Both PSFs are normalized to unit total energy over the same window,
    # so the peak intensity measures the concentration of the light.
    peak_nominal = psf_nominal.outputs.max()
    peak_shifted = psf_shifted.outputs.max()

    assert peak_shifted > 5 * peak_nominal

    # Refocusing should also reduce the geometric spot size.
    rms_nominal = _rms_radius(psf_nominal)
    rms_shifted = _rms_radius(psf_shifted)

    assert rms_shifted < rms_nominal


def test_wavefield_multiwavelength():
    """The wavefield should broadcast over wavelength and field axes."""
    system = _telescope()

    wavelength = na.linspace(450, 550, axis="wavelength", num=2) * u.nm
    field = na.Cartesian2dVectorArray(
        x=na.linspace(-0.5, 0.5, axis="field", num=2),
        y=0,
    )

    wavefield = system.wavefield(
        wavelength=wavelength,
        field=field,
        width_sensor=0.05 * u.mm,
        num=31**2,
        num_sensor=11,
        seed=5,
    )

    amplitude = wavefield.outputs.amplitude
    assert "wavelength" in amplitude.shape
    assert "field" in amplitude.shape
    assert "wavefield_x" in amplitude.shape
    assert "wavefield_y" in amplitude.shape
    assert np.all(np.isfinite(np.abs(amplitude.ndarray)))


def test_psf_newtonian():
    """
    The Huygens PSF of the Newtonian telescope from the SequentialSystem
    docstring, which exercises fold mirrors (the obliquity sign) and central
    obscurations (beam-footprint sampling).
    """
    primary_mirror_z = 200 * u.mm
    fold_mirror_z = 50 * u.mm
    sensor_x = 50 * u.mm

    primary_mirror = optika.surfaces.Surface(
        name="mirror",
        sag=optika.sags.ParabolicSag(
            focal_length=-(primary_mirror_z - fold_mirror_z + sensor_x),
        ),
        aperture=optika.apertures.RectangularAperture(40 * u.mm),
        material=optika.materials.Mirror(),
        is_pupil_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(
            z=primary_mirror_z,
        ),
    )
    fold_mirror = optika.surfaces.Surface(
        name="fold_mirror",
        aperture=optika.apertures.RectangularAperture(25 * u.mm),
        material=optika.materials.Mirror(),
        transformation=na.transformations.TransformationList([
            na.transformations.Cartesian3dRotationY((90 + 45) * u.deg),
            na.transformations.Cartesian3dTranslation(z=fold_mirror_z),
        ]),
    )
    obscuration = optika.surfaces.Surface(
        name="obscuration",
        aperture=dataclasses.replace(fold_mirror.aperture, inverted=True),
        transformation=fold_mirror.transformation,
    )
    sensor = optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=20 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        num_pixel=na.Cartesian2dVectorArray(128, 128),
        timedelta_exposure=1 * u.s,
        transformation=na.transformations.TransformationList([
            na.transformations.Cartesian3dRotationY(-90 * u.deg),
            na.transformations.Cartesian3dTranslation(
                x=-sensor_x,
                z=fold_mirror_z,
            ),
        ]),
        is_field_stop=True,
    )
    system = optika.systems.SequentialSystem(
        surfaces=[
            optika.surfaces.Surface(name="front"),
            obscuration,
            primary_mirror,
            fold_mirror,
        ],
        sensor=sensor,
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=500 * u.nm,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
                centers=True,
            ),
        ),
    )

    field = na.Cartesian2dVectorArray(0, 0)

    wavefield = system.wavefield(
        field=field,
        width_sensor=0.05 * u.mm,
        num=101**2,
        num_sensor=51,
        seed=9,
    )
    intensity = np.square(np.abs(wavefield.outputs.amplitude))
    position = wavefield.outputs.position.xy
    centroid = (position * intensity).sum() / intensity.sum()

    rayfunction = system.rayfunction(field=field)
    where = rayfunction.outputs.unvignetted
    position_rays = rayfunction.outputs.position.xy
    centroid_rays = (position_rays * where).sum() / where.sum()

    assert np.abs(centroid.x - centroid_rays.x) < 5 * u.um
    assert np.abs(centroid.y - centroid_rays.y) < 5 * u.um


def test_wavefield_grating():
    """
    For a concave diffraction grating used in first order, the centroid of
    the Huygens PSF should match the centroid of a geometric raytrace.
    This exercises the ruling phase function of the wave-propagation path,
    including its sign convention: an incorrect sign would displace the
    Huygens spot by twice the (millimeter-scale) dispersion offset.
    """
    grating = optika.surfaces.Surface(
        name="grating",
        sag=optika.sags.SphericalSag(radius=-400 * u.mm),
        aperture=optika.apertures.CircularAperture(15 * u.mm),
        material=optika.materials.Mirror(),
        rulings=optika.rulings.Rulings(
            spacing=20 * u.um,
            diffraction_order=1,
        ),
        is_pupil_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(z=200 * u.mm),
    )
    sensor = optika.sensors.ImagingSensor(
        name="sensor",
        width_pixel=50 * u.um,
        axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
        num_pixel=na.Cartesian2dVectorArray(256, 256),
        timedelta_exposure=1 * u.s,
        is_field_stop=True,
    )
    system = optika.systems.SequentialSystem(
        surfaces=[grating],
        sensor=sensor,
        grid_input=optika.vectors.ObjectVectorArray(
            wavelength=500 * u.nm,
            field=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                num=5,
                centers=True,
            ),
            pupil=na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
                num=5,
                centers=True,
            ),
        ),
    )

    field = na.Cartesian2dVectorArray(0, 0)

    wavefield = system.wavefield(
        field=field,
        width_sensor=0.2 * u.mm,
        num=151**2,
        num_sensor=51,
        seed=4,
    )
    intensity = np.square(np.abs(wavefield.outputs.amplitude))
    position = wavefield.outputs.position.xy
    centroid = (position * intensity).sum() / intensity.sum()

    rayfunction = system.rayfunction(field=field)
    where = rayfunction.outputs.unvignetted
    position_rays = rayfunction.outputs.position.xy
    centroid_rays = (position_rays * where).sum() / where.sum()

    assert np.abs(centroid.x - centroid_rays.x) < 2 * u.um
    assert np.abs(centroid.y - centroid_rays.y) < 2 * u.um


def test_wavefield_invalid():
    system = _telescope()

    with pytest.raises(ValueError, match="width_sensor"):
        system.wavefield(field=na.Cartesian2dVectorArray(0, 0))

    with pytest.raises(ValueError, match="num"):
        system.wavefield(
            field=na.Cartesian2dVectorArray(0, 0),
            width_sensor=0.02 * u.mm,
            num=[100, 100],
        )


def test_wavefield_samples_decentered_aperture():
    """
    The stratified samples must cover an aperture that is decentered by its
    internal transformation.

    Regression test: the polygonal-aperture bounding boxes used to ignore
    the internal transformation, so the sampling grid covered the
    untranslated footprint and only its overlap with the true aperture
    received nonzero amplitude (an effective pupil far smaller than the
    real one).
    """
    half_width = 5 * u.mm
    decenter = 12 * u.mm
    surface = optika.surfaces.Surface(
        name="decentered",
        aperture=optika.apertures.RectangularAperture(
            half_width=half_width,
            transformation=na.transformations.Cartesian3dTranslation(x=decenter),
        ),
    )

    samples = surface.wavefield_samples(axis=("sx", "sy"), num=51**2, seed=0)

    shape = na.shape_broadcasted(
        samples.position.x,
        samples.position.y,
        samples.amplitude,
    )
    where = np.abs(na.broadcast_to(samples.amplitude, shape).ndarray) > 0
    x = na.broadcast_to(samples.position.x, shape).ndarray[where]
    y = na.broadcast_to(samples.position.y, shape).ndarray[where]

    assert np.any(where)
    assert x.min() < decenter - 0.9 * half_width
    assert x.max() > decenter + 0.9 * half_width
    assert y.min() < -0.9 * half_width
    assert y.max() > +0.9 * half_width
