"""Tests of the physical-optics propagation methods of SequentialSystem."""

import dataclasses
import inspect
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
        num_pupil=101,
        num_sensor=51,
    )

    intensity = psf.outputs

    index_peak = np.unravel_index(
        np.argmax(intensity.ndarray),
        intensity.ndarray.shape,
    )
    index_center = (51 // 2, 51 // 2)
    assert index_peak == index_center

    cut = intensity[dict(wavefield_y=51 // 2)].ndarray
    cut = cut / cut.max()
    minima = [
        i for i in range(51 // 2 + 1, 50) if cut[i] < cut[i - 1] and cut[i] < cut[i + 1]
    ]
    assert minima

    x = psf.inputs.position.x.ndarray
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
        num_pupil=101,
        num_sensor=51,
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


@pytest.mark.parametrize(
    argnames="tilted",
    argvalues=["system", "object"],
)
def test_wavefield_transformation(tilted: str) -> None:
    """
    The transformation of a system turns its optics but not its object, so
    tilting the test telescope, or tilting the object instead, moves the
    image of a star on the old axis by :math:`f \\tan \\theta`.
    The Huygens PSF must land where the rays do, with most of the light of
    the PSF of the telescope untilted.
    """
    angle = 0.25 * u.deg
    focal_length = 200 * u.mm
    rotation = na.transformations.Cartesian3dRotationY(angle)
    if tilted == "system":
        system = dataclasses.replace(_telescope(), transformation=rotation)
    else:
        system = dataclasses.replace(
            _telescope(),
            object=optika.surfaces.Surface(name="object", transformation=rotation),
        )
    field = na.Cartesian2dVectorArray(0, 0) * u.deg
    kwargs = dict(
        field=field,
        normalized_field=False,
        width_sensor=0.05 * u.mm,
        num_pupil=101,
        num_sensor=51,
    )

    psf = system.psf(**kwargs)
    intensity = psf.outputs
    position = psf.inputs.position
    centroid = (position * intensity).sum() / intensity.sum()

    rayfunction = system.rayfunction(field=field, normalized_field=False)
    where = rayfunction.outputs.unvignetted
    position_rays = rayfunction.outputs.position.xy
    centroid_rays = (position_rays * where).sum() / where.sum()

    shift = focal_length * np.tan(angle)
    assert np.abs(np.abs(centroid_rays.x) - shift) < 0.01 * shift

    radius_airy = (1.22 * 500 * u.nm * 200 / 40).to(u.mm)
    assert np.abs(centroid.x - centroid_rays.x) < radius_airy / 2
    assert np.abs(centroid.y - centroid_rays.y) < radius_airy / 2

    # the grid is centered on the rays, so check that the light is there
    # too, rather than a noise floor centered by the grid
    psf_untilted = _telescope().psf(**kwargs)
    assert intensity.max() > psf_untilted.outputs.max() / 2


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
        base=optika.sags.ParabolicSag(focal_length=-focal_length),
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
        num_pupil=101,
        num_sensor=51,
    )

    psf_nominal = _telescope(sag_primary=sag).psf(**kwargs)
    psf_shifted = _telescope(sag_primary=sag, z_sensor=z_focus).psf(**kwargs)

    # Both PSFs are normalized to the power of the same beam, so the peak
    # intensity measures the concentration of the light.
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
        num_pupil=31,
        num_sensor=11,
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
    obscurations.
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
        transformation=na.transformations.TransformationList(
            [
                na.transformations.Cartesian3dRotationY((90 + 45) * u.deg),
                na.transformations.Cartesian3dTranslation(z=fold_mirror_z),
            ]
        ),
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
        transformation=na.transformations.TransformationList(
            [
                na.transformations.Cartesian3dRotationY(-90 * u.deg),
                na.transformations.Cartesian3dTranslation(
                    x=-sensor_x,
                    z=fold_mirror_z,
                ),
            ]
        ),
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
        num_pupil=101,
        num_sensor=51,
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
        num_pupil=151,
        num_sensor=51,
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

    # A field given as an angle needs neither a field stop nor a sensor to
    # be found, so these reach the checks of the wavefield itself.
    kwargs = dict(
        field=na.Cartesian2dVectorArray(0, 0) * u.deg,
        normalized_field=False,
        width_sensor=0.02 * u.mm,
    )

    primary = dataclasses.replace(system.surfaces[0], aperture=None)
    system_open = dataclasses.replace(system, surfaces=[primary])
    with pytest.raises(ValueError, match="at least one surface"):
        system_open.wavefield(**kwargs)

    system_blind = dataclasses.replace(system, sensor=None)
    with pytest.raises(ValueError, match="must have a sensor"):
        system_blind.wavefield(**kwargs)


def test_wavefield_default_field() -> None:
    """
    Without a field, the wavefield is computed at every field point of the
    input grid of the system, as a :class:`WavefieldFunctionArray`.
    """
    system = _telescope()

    wavefield = system.wavefield(
        width_sensor=0.02 * u.mm,
        num_pupil=11,
        num_sensor=5,
    )

    assert isinstance(wavefield, optika.wavefields.WavefieldFunctionArray)
    assert wavefield.type_explicit is optika.wavefields.WavefieldFunctionArray
    assert type(wavefield.inputs) is optika.vectors.SceneVectorArray
    assert wavefield.outputs.amplitude.shape == dict(
        field_x=5,
        field_y=5,
        wavefield_x=5,
        wavefield_y=5,
    )


def test_wavefield_position_sensor() -> None:
    """
    The wavefield can be evaluated at any positions on the sensor.
    Positions which are not a grid along both axes have no area of their own.
    """
    system = _telescope()
    field = na.Cartesian2dVectorArray(0, 0)
    position_sensor = na.Cartesian2dVectorArray(
        x=na.linspace(-2, 2, axis="p", num=3) * u.um,
        y=0 * u.um,
    )

    wavefield = system.wavefield(
        field=field,
        position_sensor=position_sensor,
        num_pupil=101,
    )
    grid = system.wavefield(
        field=field,
        width_sensor=0.004 * u.mm,
        num_pupil=101,
        num_sensor=3,
    )

    assert wavefield.outputs.area == 0 * u.mm**2
    assert np.allclose(
        wavefield.outputs.amplitude,
        grid.outputs.amplitude[dict(wavefield_y=1)].combine_axes(["wavefield_x"], "p"),
    )


@pytest.mark.parametrize(
    argnames="moved,image",
    argvalues=[
        (None, -0.5 * u.mm),
        ("object", -0.7 * u.mm),
        ("system", -0.5 * u.mm),
    ],
)
def test_psf_finite_object(moved: None | str, image: u.Quantity) -> None:
    """
    A spherical mirror images a point at its center of curvature onto
    itself, turned around the axis, without aberration.
    The Huygens PSF of a point at a finite distance must land on its image,
    where the rays do, with the object moved by a transformation of its own,
    or with the mirror and the sensor tilted about the center of curvature
    by a transformation of the system.
    """
    mirror = optika.surfaces.Surface(
        name="mirror",
        sag=optika.sags.SphericalSag(radius=-200 * u.mm),
        aperture=optika.apertures.CircularAperture(20 * u.mm),
        material=optika.materials.Mirror(),
        is_pupil_stop=True,
        transformation=na.transformations.Cartesian3dTranslation(z=200 * u.mm),
    )
    obj = optika.surfaces.Surface(
        name="object",
        aperture=optika.apertures.RectangularAperture(1 * u.mm),
    )
    transformation = None
    if moved == "object":
        obj = dataclasses.replace(
            obj,
            transformation=na.transformations.Cartesian3dTranslation(x=0.2 * u.mm),
        )
    elif moved == "system":
        transformation = na.transformations.Cartesian3dRotationY(0.25 * u.deg)
    system = dataclasses.replace(
        _telescope(),
        surfaces=[mirror],
        object=obj,
        transformation=transformation,
    )
    assert not system.object_is_at_infinity

    field = na.Cartesian2dVectorArray(0.5, 0) * u.mm
    psf = system.psf(
        field=field,
        normalized_field=False,
        width_sensor=0.05 * u.mm,
        num_pupil=101,
        num_sensor=51,
    )
    intensity = psf.outputs
    centroid = (psf.inputs.position * intensity).sum() / intensity.sum()

    rayfunction = system.rayfunction(field=field, normalized_field=False)
    where = rayfunction.outputs.unvignetted
    position_rays = rayfunction.outputs.position.xy
    centroid_rays = (position_rays * where).sum() / where.sum()
    assert np.abs(centroid_rays.x - image) < 1 * u.um

    radius_airy = (1.22 * 500 * u.nm * 200 / 40).to(u.mm)
    assert np.abs(centroid.x - centroid_rays.x) < radius_airy / 2
    assert np.abs(centroid.y - centroid_rays.y) < radius_airy / 2

    # the grid holds nearly all of the power of the beam
    area = np.square(0.05 * u.mm / 50)
    assert intensity.sum() * area > 0.95


def test_raytrace_optical_path_focus() -> None:
    """
    By Fermat's principle, every ray of a plane wave which a paraboloid brings
    to a focus takes the same optical path to it.
    """
    system = _telescope()
    pupil = na.Cartesian2dVectorLinearSpace(
        start=-1,
        stop=1,
        axis=na.Cartesian2dVectorArray("px", "py"),
        num=11,
        centers=True,
    )
    rays = system.raytrace(
        field=na.Cartesian2dVectorArray(0, 0),
        pupil=pupil,
        optical_path=True,
    ).outputs[{system.axis_surface: ~0}]

    where = rays.unvignetted
    assert np.any(where)

    optical_path = rays.optical_path
    reference = (optical_path * where).sum() / where.sum()
    assert np.all(np.abs(np.where(where, optical_path - reference, 0)) < 1e-9 * u.mm)


def test_wavefield_energy() -> None:
    """
    The wavefield on the sensor carries the power of the incident beam:
    nearly all of the power of rays of unit intensity through the aperture
    lands in a window of many Airy radii around the image.
    """
    system = _telescope()

    wavefield = system.wavefield(
        field=na.Cartesian2dVectorArray(0, 0),
        width_sensor=0.1 * u.mm,
        num_pupil=101,
        num_sensor=201,
    )
    outputs = wavefield.outputs
    power = (np.square(np.abs(outputs.amplitude)) * outputs.area).sum()
    power_expected = np.pi * np.square(20 * u.mm)

    ratio = (power / power_expected).to(u.dimensionless_unscaled)
    assert 0.98 < ratio < 1.001


@pytest.mark.parametrize("radius", [1.2 * u.mm, 0.4 * u.mm])
def test_wavefield_energy_narrow_beam(radius: u.Quantity) -> None:
    """
    An aperture in front of the telescope, smaller than its pupil stop, limits
    the beam to a small part of the pupil, and the grid of rays still covers
    all of it, even when only one ray of the coarser trace gets through.
    """
    system = _telescope()
    front = optika.surfaces.Surface(
        name="front",
        aperture=optika.apertures.CircularAperture(radius),
        transformation=na.transformations.Cartesian3dTranslation(z=100 * u.mm),
    )
    system = dataclasses.replace(system, surfaces=[front, *system.surfaces])

    wavefield = system.wavefield(
        field=na.Cartesian2dVectorArray(0, 0) * u.deg,
        normalized_field=False,
        width_sensor=2 * u.mm,
        num_pupil=101,
        num_sensor=101,
    )
    outputs = wavefield.outputs
    power = (np.square(np.abs(outputs.amplitude)) * outputs.area).sum()
    power_expected = np.pi * np.square(radius)

    ratio = (power / power_expected).to(u.dimensionless_unscaled)
    assert 0.97 < ratio < 1.01


def test_psf_encircled_energy() -> None:
    """
    The Huygens PSF is the irradiance per unit power of the beam, so its
    integral over the circle of each of the first three dark rings of an
    Airy pattern is the fraction of the power inside that ring,
    :math:`1 - J_0^2(v)`, where :math:`v` is a zero of :math:`J_1`.
    """
    import scipy.special

    width = 20 * u.um
    num = 101
    psf = _telescope().psf(
        field=na.Cartesian2dVectorArray(0, 0),
        width_sensor=width,
        num_pupil=101,
        num_sensor=num,
    )

    zero = na.ScalarArray(scipy.special.jn_zeros(1, 3), axes="ring")
    radius = zero * 500 * u.nm * 200 / 40 / np.pi
    inside = psf.inputs.position.length < radius
    area = np.square(width / (num - 1))
    result = (psf.outputs * inside).sum(("wavefield_x", "wavefield_y")) * area

    expected = 1 - np.square(scipy.special.j0(zero))
    assert np.allclose(result.to(u.dimensionless_unscaled), expected, atol=2e-3)


def test_psf_sensor_tilted() -> None:
    """
    A tilted sensor spreads the image over more of its area, so the
    integral of :math:`|E|^2` over it grows by about one over the cosine of
    the tilt, but the integral of the PSF over a grid stretched along the
    tilt by one over the cosine, which holds the same part of the image, is
    the same as that of a sensor which is not tilted.
    """
    angle = 30 * u.deg
    width = 50 * u.um
    num = 101
    axis = ("wavefield_x", "wavefield_y")

    def integral(tilted: bool) -> tuple[na.AbstractScalar, na.AbstractScalar]:
        """
        The integrals of the PSF and of :math:`|E|^2` over a grid of sensor
        positions which is stretched along :math:`y` by one over the cosine
        of the tilt of the sensor.

        Parameters
        ----------
        tilted
            Whether the sensor is tilted about the image.
        """
        angle_sensor = angle if tilted else 0 * u.deg
        stretch = 1 / np.cos(angle_sensor)
        system = _telescope()
        sensor = dataclasses.replace(
            system.sensor,
            transformation=na.transformations.Cartesian3dRotationX(angle_sensor),
        )
        system = dataclasses.replace(system, sensor=sensor)
        half = na.Cartesian2dVectorArray(width / 2, stretch * width / 2)
        kwargs = dict(
            field=na.Cartesian2dVectorArray(0, 0) * u.deg,
            normalized_field=False,
            position_sensor=na.Cartesian2dVectorLinearSpace(
                start=-half,
                stop=half,
                axis=na.Cartesian2dVectorArray(*axis),
                num=num,
            ).explicit,
            num_pupil=61,
            axis=axis,
        )
        area = np.square(width / (num - 1)) * stretch
        psf = system.psf(**kwargs)
        wavefield = system.wavefield(**kwargs)
        irradiance = np.square(np.abs(wavefield.outputs.amplitude))
        return psf.outputs.sum(axis) * area, irradiance.sum(axis) * area

    psf, irradiance = integral(tilted=False)
    psf_tilted, irradiance_tilted = integral(tilted=True)

    ratio = (irradiance_tilted / irradiance).to(u.dimensionless_unscaled)
    assert np.allclose(ratio, 1 / np.cos(angle), rtol=2e-3)

    assert 0.97 < psf.to(u.dimensionless_unscaled) < 1
    assert np.allclose(psf_tilted, psf, rtol=2e-4)


def test_psf_vignetted_field() -> None:
    """
    The PSF of a field from which no ray reaches the last surface before the
    sensor is zero, and leaves the PSF of the other fields as they are.
    """
    system = _telescope()
    stop = optika.surfaces.Surface(
        name="stop",
        aperture=optika.apertures.CircularAperture(2 * u.mm),
        transformation=na.transformations.Cartesian3dTranslation(z=10 * u.mm),
    )
    system = dataclasses.replace(system, surfaces=[*system.surfaces, stop])
    field = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0, 1]), axes="field") * u.deg,
        y=0 * u.deg,
    )
    kwargs = dict(
        normalized_field=False,
        width_sensor=0.02 * u.mm,
        num_pupil=31,
        num_sensor=11,
    )

    psf = system.psf(field=field, **kwargs)
    psf_center = system.psf(field=field[dict(field=0)], **kwargs)

    assert np.all(np.isfinite(psf.outputs))
    assert np.all(psf.outputs[dict(field=1)] == 0)
    assert np.allclose(psf.outputs[dict(field=0)], psf_center.outputs)

    assert type(psf.inputs) is optika.vectors.ImageVectorArray
    assert np.all(psf.inputs.field == field)


def test_wavefield_line_cut() -> None:
    """
    A grid of sensor positions with one position along one of its axes is a
    line, which has no area.
    """
    system = _telescope()
    position_sensor = na.Cartesian2dVectorLinearSpace(
        start=na.Cartesian2dVectorArray(-2, 0) * u.um,
        stop=na.Cartesian2dVectorArray(2, 0) * u.um,
        axis=na.Cartesian2dVectorArray("wavefield_x", "wavefield_y"),
        num=na.Cartesian2dVectorArray(5, 1),
    ).explicit

    wavefield = system.wavefield(
        field=na.Cartesian2dVectorArray(0, 0),
        position_sensor=position_sensor,
        num_pupil=31,
    )

    assert wavefield.outputs.area == 0 * u.mm**2
    assert np.all(np.isfinite(wavefield.outputs.amplitude))


def test_wavefield_rays_missing_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A ray which is vignetted before the sensor, and so has no position on it,
    such as one which misses a curved surface, does not change the wavefield
    or the grid of sensor positions centered on the image.
    """
    system = _telescope()
    kwargs = dict(
        field=na.Cartesian2dVectorArray(0, 0),
        width_sensor=0.02 * u.mm,
        num_pupil=31,
        num_sensor=11,
    )
    wavefield_expected = system.wavefield(**kwargs)

    raytrace = optika.systems.SequentialSystem.raytrace

    def raytrace_missing(
        self: optika.systems.SequentialSystem,
        *args,
        **kwargs,
    ) -> optika.rays.RayFunctionArray:
        """Put the rays vignetted before the sensor nowhere on it."""
        result = raytrace(self, *args, **kwargs)
        axis = kwargs["axis"]
        rays = result.outputs
        num = rays.shape[axis]
        index = na.arange(0, num, axis=axis)
        missing = (index == num - 1) & ~rays.unvignetted[{axis: ~1}]
        nan = na.Cartesian3dVectorArray(np.nan, np.nan, np.nan) * u.mm
        rays.position = np.where(missing, nan, rays.position)
        return result

    monkeypatch.setattr(
        optika.systems.SequentialSystem,
        "raytrace",
        raytrace_missing,
    )

    wavefield = system.wavefield(**kwargs)

    assert np.all(np.isfinite(wavefield.outputs.amplitude))
    assert np.allclose(wavefield.outputs.position, wavefield_expected.outputs.position)
    assert np.allclose(
        wavefield.outputs.amplitude,
        wavefield_expected.outputs.amplitude,
    )


def test_wavefield_figure_error_tilt() -> None:
    """
    A figure error which tilts the primary mirror does not deflect the rays,
    but it moves the Huygens PSF to where the rays of a mirror which is tilted
    by the same amount land.
    """
    focal_length = 200 * u.mm
    radius = 20 * u.mm
    slope = 5e-6
    base = optika.sags.ParabolicSag(focal_length=-focal_length)

    grid = na.Cartesian2dVectorLinearSpace(
        start=-25 * u.mm,
        stop=25 * u.mm,
        axis=na.Cartesian2dVectorArray("map_x", "map_y"),
        num=11,
    ).explicit
    sag_map = optika.sags.FigureErrorSag(
        base=base,
        figure_error=na.FunctionArray(grid, slope * grid.x),
        axis=("map_x", "map_y"),
    )
    sag_tilt = optika.sags.ZernikeSag(
        base=base,
        coefficients=[0 * u.mm, slope * radius / 2],
        radius=radius,
    )

    # A physical field angle, since a normalized one is solved for the center
    # of the sensor, which is the field stop, and so would follow the tilt.
    field = na.Cartesian2dVectorArray(0, 0) * u.deg
    kwargs = dict(
        field=field,
        normalized_field=False,
        position_sensor=na.Cartesian2dVectorLinearSpace(
            start=-15 * u.um,
            stop=15 * u.um,
            axis=na.Cartesian2dVectorArray("wavefield_x", "wavefield_y"),
            num=61,
        ).explicit,
        num_pupil=101,
    )
    psf = _telescope(sag_primary=sag_map).psf(**kwargs)
    psf_expected = _telescope(sag_primary=sag_tilt).psf(**kwargs)

    peak = psf_expected.outputs.max()
    assert np.all(np.abs(psf.outputs - psf_expected.outputs) < 1e-6 * peak)

    def centroid_rays(sag: optika.sags.AbstractSag) -> na.Cartesian2dVectorArray:
        """The centroid of the rays on the sensor of the telescope."""
        rayfunction = _telescope(sag_primary=sag).rayfunction(
            field=field,
            normalized_field=False,
        )
        where = rayfunction.outputs.unvignetted
        return (rayfunction.outputs.position.xy * where).sum() / where.sum()

    # the map does not deflect the rays, but the tilted mirror does
    assert np.all(centroid_rays(sag_map).length < 1 * u.nm)
    shift = centroid_rays(sag_tilt)
    assert np.abs(shift.x - 2 * slope * focal_length) < 1 * u.nm
    assert np.abs(shift.y) < 1 * u.nm

    # the window cuts off more of the far side of the PSF, which pulls its
    # centroid toward the center of the window by a few tens of nanometers
    centroid = (psf.inputs.position * psf.outputs).sum() / psf.outputs.sum()
    assert np.abs(centroid.x - shift.x) < 0.05 * u.um
    assert np.abs(centroid.y - shift.y) < 0.01 * u.um


def test_derivative() -> None:
    """
    The derivative is a central difference inside the grid and a one-sided
    difference at its ends and next to a missing point.
    """
    x = na.linspace(0, 5, axis="i", num=6) * u.mm
    a = na.Cartesian3dVectorArray(x=np.square(x) / u.mm, y=0 * u.mm, z=0 * u.mm)
    a.x[dict(i=2)] = np.nan * u.mm

    result = optika.systems._sequential._derivative(a, axis="i")

    result_expected = na.ScalarArray(
        ndarray=np.array([1, 1, np.nan, 7, 8, 9]) * u.mm,
        axes="i",
    )
    assert np.allclose(result.x, result_expected, equal_nan=True)


def test_psf_sensor_tilted_defocused() -> None:
    """
    On a sensor which is tilted and out of focus, each position is lit from
    a different direction, so the light is spread as the flux of the
    wavefield through the sensor, not as :math:`|E|^2`.
    The centroid of the PSF is that of the rays, while that of
    :math:`|E|^2` is about 1.7 um away from it.
    """
    angle = 45 * u.deg
    system = _telescope()
    sensor = dataclasses.replace(
        system.sensor,
        transformation=na.transformations.TransformationList(
            [
                na.transformations.Cartesian3dRotationX(angle),
                na.transformations.Cartesian3dTranslation(z=0.5 * u.mm),
            ]
        ),
    )
    system = dataclasses.replace(system, sensor=sensor)
    field = na.Cartesian2dVectorArray(0, 0) * u.deg
    axis = ("wavefield_x", "wavefield_y")

    width = 200 * u.um
    half = na.Cartesian2dVectorArray(width / 2, width / 2 / np.cos(angle))
    position = na.Cartesian2dVectorLinearSpace(
        start=-half,
        stop=half,
        axis=na.Cartesian2dVectorArray(*axis),
        num=81,
    ).explicit
    psf = system.psf(
        field=field,
        normalized_field=False,
        position_sensor=position,
        num_pupil=151,
        axis=axis,
    )
    y = psf.inputs.position.y
    centroid = (y * psf.outputs).sum() / psf.outputs.sum()

    rays = system.rayfunction(
        field=field,
        normalized_field=False,
        pupil=na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("pupil_x", "pupil_y"),
            num=301,
            centers=True,
        ),
    ).outputs
    unvignetted = rays.unvignetted
    centroid_rays = (rays.position.y * unvignetted).sum() / unvignetted.sum()

    assert np.abs(centroid - centroid_rays) < 0.2 * u.um


def test_wavefield_parameters() -> None:
    """
    :meth:`psf` passes its keyword arguments to ``_wavefield()``, whose
    parameters and their defaults must be those of :meth:`wavefield`.
    """
    public = inspect.signature(optika.systems.SequentialSystem.wavefield)
    private = inspect.signature(optika.systems.SequentialSystem._wavefield)
    for name, parameter in public.parameters.items():
        assert private.parameters[name].kind == parameter.kind
        assert private.parameters[name].default == parameter.default
