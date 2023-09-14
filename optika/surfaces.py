from typing import TypeVar, Generic
import abc
import dataclasses
import numpy.typing as npt
import matplotlib.axes
import named_arrays as na
import optika
import optika.mixins
import optika.plotting
import optika.propagators

__all__ = [
    "AbstractSurface",
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
    optika.plotting.Plottable,
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.propagators.AbstractRayPropagator,
    Generic[SagT, MaterialT, ApertureT, ApertureMechanicalT, RulingsT],
):
    """
    Interface for a single optical surface.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        human-readable name of the surface
        """

    @property
    @abc.abstractmethod
    def sag(self) -> SagT:
        """
        the sag profile of the surface
        """

    @property
    @abc.abstractmethod
    def material(self) -> MaterialT:
        """
        optical material type of the surface
        """

    @property
    @abc.abstractmethod
    def aperture(self) -> ApertureT:
        """
        region of the surface which allows light to propagate
        """

    @property
    @abc.abstractmethod
    def aperture_mechanical(self) -> ApertureMechanicalT:
        """
        shape of the physical substrate containing the optical surface
        """

    @property
    @abc.abstractmethod
    def rulings(self) -> RulingsT:
        """
        the ruling density of the surface
        """

    @property
    @abc.abstractmethod
    def is_field_stop(self) -> bool:
        """
        flag controlling whether this surface should act as the field stop
        for the system
        """

    @property
    @abc.abstractmethod
    def is_pupil_stop(self) -> bool:
        """
        flag controlling whether this surface should act as the pupil stop
        for the system
        """

    @property
    @abc.abstractmethod
    def is_spectral_stop(self) -> bool:
        """
        flag controlling whether this surface should act as the surface
        determining the wavelength range of the system
        """

    def propagate_rays(
        self,
        rays: optika.rays.RayVectorArray,
        # material: None | optika.materials.AbstractMaterial = None,
    ) -> optika.rays.RayVectorArray:
        sag = self.sag
        material = self.material
        aperture = self.aperture
        rulings = self.rulings
        transformation = self.transformation

        if transformation is not None:
            rays = transformation.inverse(rays)

        rays = material.refract_rays(
            rays=rays,
            sag=sag,
            rulings=rulings,
        )

        if aperture is not None:
            rays = aperture.clip_rays(rays)

        if transformation is not None:
            rays = transformation(rays)

        return rays

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
                transformation = transformation_self @ transformation
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
    sag: SagT = None
    material: MaterialT = None
    aperture: ApertureT = None
    aperture_mechanical: ApertureMechanicalT = None
    rulings: RulingsT = None
    is_field_stop: bool = False
    is_pupil_stop: bool = False
    is_spectral_stop: bool = False
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

    def __post_init__(self):
        if self.sag is None:
            self.sag = optika.sags.NoSag()
        if self.material is None:
            self.material = optika.materials.Vacuum()
