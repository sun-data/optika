import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize(
    argnames="telescope",
    argvalues=[
        optika.telescopes.Wolter1(
            focal_length=2 * u.m,
            radius=72 * u.mm,
            separation=500 * u.mm,
        ),
        optika.telescopes.Wolter1(
            focal_length=3 * u.m,
            radius=72 * u.mm,
            separation=500 * u.mm,
        ),
        optika.telescopes.Wolter1(
            focal_length=na.linspace(2, 3, axis="f", num=2) * u.m,
            radius=50 * u.mm,
            separation=400 * u.mm,
            length=100 * u.mm,
        ),
    ],
)
class TestWolter1:

    def test_grazing_angle(self, telescope: optika.telescopes.Wolter1):
        result = telescope.grazing_angle
        assert np.all(result > 0 * u.deg)
        assert np.all(result < 1 * u.deg)
        # the four reflections of 2 alpha each turn a marginal ray onto the focus
        assert np.allclose(
            np.tan(4 * result),
            telescope.radius / telescope.focal_length,
        )

    def test_surfaces(self, telescope: optika.telescopes.Wolter1):
        result = telescope.surfaces
        assert len(result) == 2
        primary, secondary = result
        assert isinstance(primary.sag, optika.sags.ParabolicSag)
        assert isinstance(secondary.sag, optika.sags.ConicSag)
        assert np.all(secondary.sag.conic < -1)
        for surface in result:
            assert isinstance(surface.material, optika.materials.Mirror)
            assert isinstance(surface.aperture, optika.apertures.AnnularAperture)
        # the secondary sits inside the beam converging on the primary's focus
        assert np.all(telescope.radius_secondary < telescope.radius)
        assert np.all(telescope.radius_secondary > 0 * u.mm)

    def test_focus(self, telescope: optika.telescopes.Wolter1):
        """
        An on-axis annulus of rays must converge to a point at the focus.
        """
        azimuth = na.linspace(0, 360, axis="azimuth", num=8, endpoint=False) * u.deg
        pupil = na.Cartesian2dVectorArray(
            x=telescope.radius * np.cos(azimuth),
            y=telescope.radius * np.sin(azimuth),
        )
        sensor = optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=10 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(256, 256),
            transformation=na.transformations.Cartesian3dTranslation(
                z=telescope.focal_length,
            ),
        )
        grid = optika.vectors.ObjectVectorArray(
            wavelength=15 * u.AA,
            field=na.Cartesian2dVectorArray(0, 0) * u.deg,
            pupil=pupil,
        )
        system = optika.systems.SequentialSystem(
            surfaces=telescope.surfaces,
            sensor=sensor,
            grid_input=grid,
        )
        rays = system.raytrace(
            wavelength=grid.wavelength,
            field=grid.field,
            pupil=grid.pupil,
            normalized_field=False,
            normalized_pupil=False,
        )
        final = rays.outputs[{system.axis_surface: ~0}]
        assert np.all(final.unvignetted)
        # the sensor is at the focus, so every ray lands on the axis
        assert np.all(final.position.xy.length < 1 * u.um)
        # and the rays never travel backward between surfaces (the input rays
        # start on the primary's annulus, so the first step is zero)
        z = rays.outputs.position.z
        assert np.all(np.diff(z, axis=system.axis_surface) > -1 * u.nm)
