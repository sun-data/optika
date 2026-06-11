"""
Optical interfaces used to focus light.

The building block of an optical system.
"""

from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
from astropy import units as u
import named_arrays as na
import optika
from ezdxf.addons.r12writer import R12FastStreamWriter

__all__ = [
    "AbstractSurface",
    "Surface",
]

SagT = TypeVar(
    "SagT",
    bound=None | optika.sags.AbstractSag,
)
MaterialT = TypeVar(
    "MaterialT",
    bound=None | optika.materials.AbstractMaterial,
)
ApertureT = TypeVar(
    "ApertureT",
    bound=None | optika.apertures.AbstractAperture,
)
ApertureMechanicalT = TypeVar(
    "ApertureMechanicalT",
    bound=None | optika.apertures.AbstractAperture,
)
RulingsT = TypeVar(
    "RulingsT",
    bound=None | optika.rulings.AbstractRulings,
)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSurface(
    optika.mixins.DxfWritable,
    optika.mixins.Plottable,
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
    optika.propagators.AbstractLightPropagator,
    Generic[SagT, MaterialT, ApertureT, ApertureMechanicalT, RulingsT],
):
    """
    Interface describing a single optical interface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The human-readable name of this surface.
        """

    @property
    @abc.abstractmethod
    def sag(self) -> SagT:
        """
        The sag profile of this surface.
        """

    @property
    @abc.abstractmethod
    def material(self) -> MaterialT:
        """
        The optical material type of this surface.
        """

    @property
    @abc.abstractmethod
    def aperture(self) -> ApertureT:
        """
        The region of this surface which allows light to propagate.
        """

    @property
    @abc.abstractmethod
    def aperture_mechanical(self) -> ApertureMechanicalT:
        """
        The shape of the physical substrate containing this optical surface.
        """

    @property
    @abc.abstractmethod
    def rulings(self) -> RulingsT:
        """
        The optional ruling profile of this surface.
        """

    @property
    @abc.abstractmethod
    def is_field_stop(self) -> bool:
        """
        A flag controlling whether this surface should act as the field stop
        for the system
        """

    @property
    @abc.abstractmethod
    def is_pupil_stop(self) -> bool:
        """
        A flag controlling whether this surface should act as the pupil stop
        for the system
        """

    @property
    def is_stop(self) -> bool:
        """
        If this surface is pupil stop or the field stop, return :obj:`True`.
        """
        return self.is_field_stop or self.is_pupil_stop

    def propagate_rays(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> optika.rays.RayVectorArray:
        r"""
        Refract, reflect, and/or diffract the given rays off of this surface

        Parameters
        ----------
        rays
            A set of input rays that will interact with this surface.
        """
        sag = self.sag
        material = self.material
        aperture = self.aperture
        rulings = self.rulings
        transformation = self.transformation

        if transformation is not None:
            rays = transformation.inverse(rays)

        rays_1 = sag.propagate_rays(rays)

        position_1 = rays_1.position

        normal = sag.normal(position_1)

        if rulings is not None:
            rays_1 = rulings.incident_effective(
                rays=rays_1,
                normal=normal,
            )

        wavelength_1 = rays_1.wavelength
        a = rays_1.direction
        intensity_1 = rays_1.intensity
        n1 = rays_1.index_refraction

        position_2 = position_1
        n2 = material.index_refraction(rays_1)
        r = n1 / n2

        wavelength_2 = wavelength_1 / r

        b = optika.materials.snells_law(
            direction=a,
            index_refraction=n1,
            index_refraction_new=n2,
            is_mirror=material.is_mirror,
            normal=normal,
        )

        efficiency = material.efficiency(rays_1, normal)
        if rulings is not None:
            efficiency = efficiency * rulings.efficiency(rays_1, normal)

        intensity_2 = intensity_1 * efficiency
        attenuation_2 = material.attenuation(rays_1)

        rays_2 = dataclasses.replace(
            rays_1,
            wavelength=wavelength_2,
            position=position_2,
            direction=b,
            intensity=intensity_2,
            attenuation=attenuation_2,
            index_refraction=n2,
        )

        if aperture is not None:
            rays_2 = aperture.clip_rays(rays_2)

        if transformation is not None:
            rays_2 = transformation(rays_2)

        return rays_2

    def wavefield_samples(
        self,
        axis: tuple[str, str],
        num: int | na.AbstractCartesian2dVectorArray = 10_000,
        seed: None | int = None,
        bound: None
        | tuple[
            na.AbstractCartesian2dVectorArray,
            na.AbstractCartesian2dVectorArray,
        ] = None,
    ) -> optika.wavefields.WavefieldVectorArray:
        """
        Stratified-random samples of this surface's aperture, in local surface
        coordinates, with unit amplitude on unvignetted samples and zero
        amplitude elsewhere.

        The :attr:`~optika.wavefields.WavefieldVectorArray.wavelength` and
        :attr:`~optika.wavefields.WavefieldVectorArray.index_refraction`
        fields are left at their defaults for the caller to fill in.

        Parameters
        ----------
        axis
            The two logical axes of the new grid of samples.
        num
            The approximate total number of samples,
            or the number of samples along each axis.
        seed
            An optional seed for the random number generator.
        bound
            The lower and upper corners of the region to sample, in local
            surface coordinates.
            If :obj:`None` (the default), the bounding box of the aperture is
            used.
            This parameter is required for surfaces with inverted apertures
            (obscurations), since the aperture of such surfaces does not
            bound the beam.
        """
        aperture = self.aperture
        if aperture is None:
            raise ValueError(
                f"surface {self.name!r} must have an aperture to be sampled "
                f"by a wavefield."
            )

        if bound is None:
            if aperture.inverted:
                raise ValueError(
                    f"the aperture of surface {self.name!r} is inverted, "
                    f"so the sampling region must be given using the `bound` "
                    f"parameter."
                )
            lower = aperture.bound_lower.xy
            upper = aperture.bound_upper.xy
        else:
            lower, upper = bound

        if not na.unit_normalized(lower).is_equivalent(u.mm):
            raise ValueError(
                f"the aperture of surface {self.name!r} must be expressed in "
                f"physical units to be sampled by a wavefield."
            )

        if not isinstance(num, na.AbstractCartesian2dVectorArray):
            num_x = int(np.round(np.sqrt(num)))
            num = na.Cartesian2dVectorArray(num_x, num_x)

        grid = na.Cartesian2dVectorStratifiedRandomSpace(
            start=lower,
            stop=upper,
            axis=na.Cartesian2dVectorArray(*axis),
            num=num,
            seed=seed,
        ).explicit

        sag = self.sag
        if sag is None:
            sag = optika.sags.NoSag()

        position = na.Cartesian3dVectorArray(
            x=grid.x,
            y=grid.y,
            z=0 * na.unit_normalized(grid.x),
        )
        position.z = sag(position)

        where = aperture(position)

        normal = sag.normal(position)

        area = (upper.x - lower.x) * (upper.y - lower.y) / (num.x * num.y)
        area = area / np.abs(normal.z)

        return optika.wavefields.WavefieldVectorArray(
            position=position,
            amplitude=np.where(where, 1 + 0j, 0j),
            normal=normal,
            area=area,
            unvignetted=where,
        )

    def _interact_wavefield(
        self,
        wavefield: optika.wavefields.AbstractWavefieldVectorArray,
        direction: na.AbstractCartesian3dVectorArray,
    ) -> optika.wavefields.WavefieldVectorArray:
        """
        Apply this surface's material and rulings to a wavefield sampled on
        this surface.

        The amplitude is multiplied by the square root of the material
        efficiency (and of the ruling efficiency, if rulings are present),
        the ruling phase function is applied,
        and the index of refraction is updated to that of the new medium.

        Parameters
        ----------
        wavefield
            A wavefield sampled on this surface, expressed in system
            coordinates.
        direction
            The effective incidence direction of the light at each sample
            point, expressed in system coordinates.
        """
        material = self.material
        if material is None:
            material = optika.materials.Vacuum()

        sag = self.sag
        if sag is None:
            sag = optika.sags.NoSag()

        rulings = self.rulings
        transformation = self.transformation

        rays = optika.rays.RayVectorArray(
            wavelength=wavefield.wavelength,
            position=wavefield.position,
            direction=direction,
            index_refraction=wavefield.index_refraction,
        )
        if transformation is not None:
            rays = transformation.inverse(rays)

        normal = sag.normal(rays.position)

        amplitude = wavefield.amplitude

        if rulings is not None:
            spacing = rulings.spacing_
            position = rays.position

            # The phase shift imparted by the rulings is
            # 2 pi m N(r), where N(r) is the groove-counting function whose
            # gradient is the local groove density.
            # Integrate the groove density along the chord from the local
            # origin to each sample point using Gauss-Legendre quadrature,
            # which is exact for constant ruling spacing.
            nodes, weights = np.polynomial.legendre.leggauss(32)
            nodes = (nodes + 1) / 2
            weights = weights / 2

            grooves = 0
            for node, weight in zip(nodes, weights):
                kappa = spacing(node * position, normal)
                density = (position @ kappa) / np.square(kappa.length)
                grooves = grooves + weight * density
            grooves = grooves.to(u.dimensionless_unscaled).value

            # The same orientation convention as
            # optika.rulings.incident_effective.
            sign = np.sign(
                na.as_named_array(rays.direction @ normal).value
            )

            phase = 2 * np.pi * rulings.diffraction_order * grooves
            amplitude = amplitude * np.exp(1j * sign * phase)
            amplitude = amplitude * np.sqrt(rulings.efficiency(rays, normal))

        efficiency = material.efficiency(rays, normal)

        return dataclasses.replace(
            wavefield,
            amplitude=amplitude * np.sqrt(efficiency),
            index_refraction=material.index_refraction(rays),
        )

    def propagate_wavefield(
        self,
        wavefield: optika.wavefields.AbstractWavefieldVectorArray,
        axis: tuple[str, str],
        axis_new: tuple[str, str],
        num: int | na.AbstractCartesian2dVectorArray = 10_000,
        chunk_size: int = 1024,
        seed: None | int = None,
        bound: None
        | tuple[
            na.AbstractCartesian2dVectorArray,
            na.AbstractCartesian2dVectorArray,
        ] = None,
    ) -> optika.wavefields.WavefieldVectorArray:
        """
        Propagate the given wavefield to this surface by evaluating the
        Rayleigh-Sommerfeld diffraction integral at a new set of stratified
        random samples of this surface's aperture.

        Both the given and the resulting wavefield are expressed in system
        coordinates.

        Parameters
        ----------
        wavefield
            The wavefield sampled on the previous surface, expressed in
            system coordinates.
        axis
            The two logical axes of the samples of the given wavefield,
            which are summed over by the diffraction integral.
        axis_new
            The two logical axes of the new samples on this surface.
            Must be distinct from `axis`.
        num
            The approximate total number of samples on this surface,
            or the number of samples along each axis.
        chunk_size
            The maximum number of destination points considered
            simultaneously by the diffraction integral.
        seed
            An optional seed for the random number generator.
        bound
            The lower and upper corners of the region of this surface to
            sample, in local surface coordinates.
            If :obj:`None` (the default), the bounding box of this surface's
            aperture is used, except for inverted apertures (obscurations),
            where the footprint of the incoming wavefield is used instead.
        """
        for ax in axis_new:
            if ax in axis:
                raise ValueError(
                    f"`axis_new`, {axis_new}, must be distinct from "
                    f"`axis`, {axis}."
                )

        if (bound is None) and (self.aperture is not None):
            if self.aperture.inverted:
                # The aperture of an obscuration does not bound the beam,
                # so sample the footprint of the incoming wavefield instead.
                position_local = wavefield.position
                if self.transformation is not None:
                    position_local = self.transformation.inverse(position_local)
                position_local = position_local.xy
                lower = position_local.min()
                upper = position_local.max()
                padding = (upper - lower) / 10
                bound = (lower - padding, upper + padding)

        target = self.wavefield_samples(
            axis=axis_new,
            num=num,
            seed=seed,
            bound=bound,
        )

        if self.transformation is not None:
            target = self.transformation(target)

        amplitude = optika.wavefields.rayleigh_sommerfeld(
            wavefield=wavefield,
            position=target.position,
            axis=axis,
            chunk_size=chunk_size,
        )

        target.amplitude = target.amplitude * amplitude
        target.wavelength = wavefield.wavelength
        target.index_refraction = wavefield.index_refraction

        weight = np.square(np.abs(wavefield.amplitude))
        centroid = (wavefield.position * weight).sum(axis) / weight.sum(axis)
        direction = target.position - centroid
        direction = direction / direction.length

        return self._interact_wavefield(
            wavefield=target,
            direction=direction,
        )

    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
    ) -> dict[str, na.AbstractScalar]:
        sag = self.sag
        aperture = self.aperture
        aperture_mechanical = self.aperture_mechanical
        transformation_self = self.transformation
        kwargs_plot = self.kwargs_plot

        if transformation is not None:
            if transformation_self is not None:
                transformation = transformation @ transformation_self
        else:
            if transformation_self is not None:
                transformation = transformation_self

        if kwargs_plot is not None:
            kwargs = kwargs | kwargs_plot

        result = dict()

        if aperture is not None:
            result["aperture"] = aperture.plot(
                ax=ax,
                transformation=transformation,
                components=components,
                sag=sag,
                **kwargs,
            )

        if aperture_mechanical is not None:
            result["aperture_mechanical"] = aperture_mechanical.plot(
                ax=ax,
                transformation=transformation,
                components=components,
                sag=sag,
                **kwargs,
            )

        return result

    def _write_to_dxf(
        self,
        dxf: R12FastStreamWriter,
        unit: u.Unit,
        transformation: None | na.transformations.AbstractTransformation = None,
        **kwargs,
    ) -> None:

        if self.transformation is not None:
            if transformation is not None:
                transformation = transformation @ self.transformation
            else:
                transformation = self.transformation

        super()._write_to_dxf(
            dxf=dxf,
            unit=unit,
            transformation=transformation,
        )

        if self.aperture is not None:
            self.aperture._write_to_dxf(
                dxf=dxf,
                unit=unit,
                transformation=transformation,
                sag=self.sag,
            )

        if self.aperture_mechanical is not None:
            self.aperture_mechanical._write_to_dxf(
                dxf=dxf,
                unit=unit,
                transformation=transformation,
                sag=self.sag,
            )


@dataclasses.dataclass(eq=False, repr=False)
class Surface(
    AbstractSurface[SagT, MaterialT, ApertureT, ApertureMechanicalT, RulingsT],
):
    """
    Representation of a single optical interface.

    Composition of a sag profile, material type, aperture, and ruling specification (all optional).

    Examples
    --------

    Define a spherical mirror, with a rectangular aperture, :math:`z=50 \\; \\text{mm}` from the origin.

    Reflect a grid of collimated rays off of this mirror and measure their position
    at the origin.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # define a spherical reflective surface 50 mm from the origin
        mirror = optika.surfaces.Surface(
            sag=optika.sags.SphericalSag(radius=-100 * u.mm),
            material=optika.materials.Mirror(),
            aperture=optika.apertures.RectangularAperture(30 * u.mm),
            transformation=na.transformations.Cartesian3dTranslation(z=50 * u.mm),
        )

        # define a detector surface at the origin to capture the reflected rays
        detector=optika.surfaces.Surface()

        # define a grid of collimated input rays
        rays_input = optika.rays.RayVectorArray(
            position=na.Cartesian3dVectorArray(
                x=na.linspace(-25, 25, axis="pupil_x", num=5) * u.mm,
                y=na.linspace(-25, 25, axis="pupil_y", num=5) * u.mm,
                z=0 * u.mm,
            ),
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # propagate the rays through the mirror and detector surfaces
        rays_mirror = mirror.propagate_rays(rays_input)
        rays_detector = detector.propagate_rays(rays_mirror)

        # stack the 3 sets of rays into a single object
        # for easier plotting
        rays = [
            rays_input,
            rays_mirror,
            rays_detector,
        ]
        rays = na.stack(rays, axis="surface")

        # plot the rays and surface
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
            components_plot = ("z", "y")
            na.plt.plot(rays.position, axis="surface", components=components_plot, color="tab:blue");
            mirror.plot(ax=ax, components=components_plot, color="black");
    """

    name: None | str = None
    """The human-readable name of the surface."""

    sag: SagT = None
    """The sag profile of this surface."""

    material: MaterialT = None
    """The optical material type of this surface."""

    aperture: ApertureT = None
    """The region of this surface which allows light to propagate."""

    aperture_mechanical: ApertureMechanicalT = None
    """The shape of the physical substrate containing this optical surface."""

    rulings: RulingsT = None
    """The optional ruling profile of this surface."""

    is_field_stop: bool = False
    """Whether this surface is the field stop of an optical system."""

    is_pupil_stop: bool = False
    """Whether this surface is the pupil stop of an optical system."""

    transformation: None | na.transformations.AbstractTransformation = None
    """The transformation between system coordinates and this surface."""

    kwargs_plot: None | dict = None
    """Additional keyword arguments to pass to the :meth:`plot` function."""

    def __post_init__(self):
        if self.sag is None:
            self.sag = optika.sags.NoSag()
        if self.material is None:
            self.material = optika.materials.Vacuum()

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.name),
            optika.shape(self.sag),
            optika.shape(self.material),
            optika.shape(self.aperture),
            optika.shape(self.rulings),
            optika.shape(self.transformation),
        )
