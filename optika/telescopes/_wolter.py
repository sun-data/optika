from __future__ import annotations
import abc
import dataclasses
import astropy.units as u
import numpy as np
import named_arrays as na
import optika

__all__ = [
    "AbstractWolter1",
    "Wolter1",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractWolter1(
    optika.mixins.Printable,
):
    """
    An interface describing a Wolter Type-I telescope: a paraboloid followed
    by a confocal hyperboloid, both struck at grazing incidence.
    """

    @property
    @abc.abstractmethod
    def focal_length(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the illuminated annulus of the primary to the
        focus of the telescope.
        """

    @property
    @abc.abstractmethod
    def radius(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius of the illuminated annulus of the primary.
        """

    @property
    @abc.abstractmethod
    def separation(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial distance from the illuminated annulus of the primary to the
        point where the secondary intercepts the converging beam.
        """

    @property
    @abc.abstractmethod
    def length(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial length of each mirror.
        """

    @property
    @abc.abstractmethod
    def margin(self) -> u.Quantity | na.AbstractScalar:
        """
        The radial clearance added on both sides of each mirror's annulus.
        """

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """
        The material of both mirrors.
        """

    @property
    def _material(self) -> optika.materials.AbstractMaterial:
        material = self.material
        if material is None:
            material = optika.materials.Mirror()
        return material

    @property
    def grazing_angle(self) -> u.Quantity | na.AbstractScalar:
        r"""
        The grazing angle :math:`\alpha` at the primary.

        A ray entering parallel to the axis at the radius of the primary is
        bent by :math:`2 \alpha` at each mirror, so
        :math:`\tan 4\alpha = r / f`.
        """
        return np.arctan(self.radius / self.focal_length) / 4

    @property
    def focal_length_primary(self) -> u.Quantity | na.AbstractScalar:
        """
        The distance from the illuminated annulus of the primary to the focus
        of the paraboloid alone.
        """
        return self.radius / np.tan(2 * self.grazing_angle)

    @property
    def _parameter_primary(self) -> u.Quantity | na.AbstractScalar:
        """
        The focal length of the paraboloid measured from its own vertex.
        """
        return self.radius**2 / (4 * self.focal_length_primary)

    @property
    def z_focus_primary(self) -> u.Quantity | na.AbstractScalar:
        """
        The axial position of the focus of the paraboloid, which is the far
        focus of the hyperboloid.

        The paraboloid is translated so that its vertex sits one focal length
        (of the paraboloid) downstream of its own focus, which puts this focus
        one such focal length short of the vertex.
        """
        return self.focal_length_primary - self._parameter_primary

    @property
    def radius_secondary(self) -> u.Quantity | na.AbstractScalar:
        """
        The radius at which the secondary intercepts the converging beam.
        """
        f = self.z_focus_primary
        return self.radius * (f - self.separation) / f

    @property
    def halfwidth(self) -> u.Quantity | na.AbstractScalar:
        """
        Half of the radial extent of each mirror.

        A mirror of axial length :math:`L` struck at grazing angle
        :math:`\alpha` spans :math:`L \tan \alpha` in radius.
        """
        return self.length * np.tan(self.grazing_angle) / 2

    @property
    def primary(self) -> optika.surfaces.Surface:
        """
        The paraboloid.

        It opens toward :math:`-z` so that rays travelling toward :math:`+z`
        reflect inward, and it is placed so that its illuminated annulus sits
        at :math:`z = 0`.
        """
        halfwidth = self.halfwidth + self.margin
        return optika.surfaces.Surface(
            name="paraboloid",
            sag=optika.sags.ParabolicSag(focal_length=-self._parameter_primary),
            material=self._material,
            aperture=optika.apertures.AnnularAperture(
                radius_inner=self.radius - halfwidth,
                radius_outer=self.radius + halfwidth,
            ),
            transformation=na.transformations.Cartesian3dTranslation(
                z=self.focal_length_primary,
            ),
        )

    @property
    def secondary(self) -> optika.surfaces.Surface:
        """
        The hyperboloid.

        It is confocal with the paraboloid: its far focus is the focus of the
        paraboloid and its near focus is the focus of the telescope, so a
        beam converging on the former is re-imaged to a point at the latter.
        A conic is fixed by its two foci and one point it passes through,
        which here is the intercept of the converging beam at
        :attr:`separation`.
        """
        z_far = self.z_focus_primary
        z_near = self.focal_length
        z_intercept = self.separation
        radius_intercept = self.radius_secondary

        d_far = np.sqrt(radius_intercept**2 + (z_intercept - z_far) ** 2)
        d_near = np.sqrt(radius_intercept**2 + (z_intercept - z_near) ** 2)

        a = np.abs(d_far - d_near) / 2
        c = (z_far - z_near) / 2
        e = c / a
        radius_vertex = a * (e**2 - 1)
        conic = -(e**2)
        z_vertex = (z_near + z_far) / 2 - a

        halfwidth = self.halfwidth + self.margin
        return optika.surfaces.Surface(
            name="hyperboloid",
            sag=optika.sags.ConicSag(radius=-radius_vertex, conic=conic),
            material=self._material,
            aperture=optika.apertures.AnnularAperture(
                radius_inner=radius_intercept - halfwidth,
                radius_outer=radius_intercept + halfwidth,
            ),
            transformation=na.transformations.Cartesian3dTranslation(z=z_vertex),
        )

    @property
    def surfaces(self) -> list[optika.surfaces.Surface]:
        """
        The primary and the secondary, in the order light meets them.
        """
        return [self.primary, self.secondary]


@dataclasses.dataclass(eq=False, repr=False)
class Wolter1(
    AbstractWolter1,
):
    """
    A Wolter Type-I telescope: a paraboloid followed by a confocal
    hyperboloid, both struck at grazing incidence.

    Soft X-rays reflect efficiently only at grazing incidence, so an X-ray
    telescope illuminates its mirrors far from the axis, on their steep,
    nearly cylindrical flanks. The two reflections satisfy the Abbe sine
    condition far better than one, which is what gives the Wolter-I its
    usable field of view.

    The telescope is fixed by its focal length, the radius of the annulus
    it collects through, and where along the axis the secondary intercepts
    the beam. The grazing angle follows from the first two. The illuminated
    annulus of the primary sits at :math:`z = 0` and the focus at
    :math:`z = f`.

    Examples
    --------

    Build a 3 m telescope and draw its mirrors and the path of an annulus of
    rays in the :math:`z`-:math:`y` plane.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import numpy as np
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        telescope = optika.telescopes.Wolter1(
            focal_length=3 * u.m,
            radius=72 * u.mm,
            separation=500 * u.mm,
        )

        # A grazing mirror is a thin ring, so sample its pupil with a polar
        # grid rather than a rectilinear one.
        azimuth = na.linspace(0, 360, axis="azimuth", num=8, endpoint=False) * u.deg
        pupil = na.Cartesian2dVectorArray(
            x=telescope.radius * np.cos(azimuth),
            y=telescope.radius * np.sin(azimuth),
        )
        sensor = optika.sensors.ImagingSensor(
            name="sensor",
            width_pixel=10 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            num_pixel=na.Cartesian2dVectorArray(2048, 2048),
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

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
            system.plot(ax=ax, components=("z", "y"), plot_rays=False, color="black")
            na.plt.plot(
                rays.outputs.position,
                ax=ax,
                axis=system.axis_surface,
                components=("z", "y"),
                color="tab:blue",
            )
            ax.set_title(f"grazing angle {telescope.grazing_angle.to(u.deg):.2f}")
    """

    focal_length: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The distance from the illuminated annulus of the primary to the focus of
    the telescope.
    """

    radius: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The radius of the illuminated annulus of the primary.
    """

    separation: u.Quantity | na.AbstractScalar = 0 * u.mm
    """
    The axial distance from the illuminated annulus of the primary to the
    point where the secondary intercepts the converging beam.
    Both mirrors are axially long at grazing incidence, so this must be large
    enough that the two do not overlap in :math:`z`.
    """

    length: u.Quantity | na.AbstractScalar = 200 * u.mm
    """
    The axial length of each mirror.
    """

    margin: u.Quantity | na.AbstractScalar = 1 * u.mm
    """
    The radial clearance added on both sides of each mirror's annulus.
    On a grazing flank a little radius is a lot of axial length, so this is
    kept small by default: at a grazing angle of a third of a degree, one
    millimetre of radius is 170 mm along the axis.
    """

    material: None | optika.materials.AbstractMaterial = None
    """
    The material of both mirrors.
    If :obj:`None`, the default, a perfect :class:`optika.materials.Mirror`.
    """
