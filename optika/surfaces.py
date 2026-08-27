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

    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        unit: None | u.UnitBase = None,
        **kwargs,
    ) -> dict[str, na.AbstractScalar]:
        """
        Draw this surface onto the given axes.

        Parameters
        ----------
        ax
            The matplotlib axes to draw onto.
            If :obj:`None`, the current axes is used.
        transformation
            Any extra transformation to apply before drawing.
        components
            Which components of the surface to draw, in the order of the axes
            of the plot.
        unit
            The unit to express every drawn length in.
        kwargs
            Additional keyword arguments passed along to the apertures and the
            substrate.

        Returns
        -------
        A dictionary of the artists drawn, under the keys

        ``aperture``
            the region which admits light, if this surface has one.

        ``aperture_mechanical``
            the outline of the substrate, if this surface has one.

        ``substrate``
            the thickness of the substrate, if the material of this surface
            has one. Itself a dictionary, holding the ``back`` face and either
            the ``wall`` joining it to the front on a 3D axes, or the
            ``edges`` of the substrate on a 2D one.
        """
        sag = self.sag
        material = self.material
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
                unit=unit,
                **kwargs,
            )

        if aperture_mechanical is not None:
            result["aperture_mechanical"] = aperture_mechanical.plot(
                ax=ax,
                transformation=transformation,
                components=components,
                sag=sag,
                unit=unit,
                **kwargs,
            )

        substrate = getattr(material, "substrate", None)
        if substrate is not None:
            result["substrate"] = self._plot_substrate(
                thickness=substrate.thickness,
                aperture=(
                    aperture_mechanical if aperture_mechanical is not None else aperture
                ),
                sag=sag,
                ax=ax,
                transformation=transformation,
                components=components,
                unit=unit,
                **kwargs,
            )

        return result

    def _plot_substrate(
        self,
        thickness: u.Quantity | na.AbstractScalar,
        aperture: None | optika.apertures.AbstractAperture,
        sag: optika.sags.AbstractSag,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        unit: None | u.UnitBase = None,
        **kwargs,
    ) -> dict[str, na.AbstractScalar]:
        """
        Draw the substrate holding this surface, giving it a visible thickness.

        An optic drawn as a bare outline reads as a plane rather than a piece
        of glass. The back of the substrate is drawn flat, a thickness behind
        the vertex, and joined to the front at each vertex of the aperture, so
        the optic reads as a solid seen edge on.

        On a 3D axes the faces are filled, so that the optic is opaque and
        hides whatever lies behind it. Seen flat there is nothing to hide, so
        the substrate is drawn as outlines, which is how a layout drawing
        shows one.
        """
        result = dict()

        if aperture is None:
            return result

        solid = optika.plot.is_3d(ax)

        # The substrate lies behind the optic, on the far side from the face
        # the light meets, which is the side the surface normal points away
        # from. Deriving it means a surface facing either direction is drawn
        # correctly; a fixed sign would be right for half of them.
        normal = sag.normal(na.Cartesian3dVectorArray() * u.mm)
        depth = -np.sign(normal.z) * thickness

        def draw(
            a: na.AbstractCartesian3dVectorArray,
            fill: bool = False,
        ) -> na.AbstractScalar:
            if unit is not None:
                a = a.to(unit)

            if fill and solid:
                # An opaque face is what hides whatever is behind the optic,
                # which is the whole reason for filling it. The face is left
                # blank and the colour asked for is used for the edges, so the
                # solid reads as a drawing rather than a silhouette.
                kwargs_fill = optika.plot.kwargs_filled(kwargs)
                return na.plt.fill(
                    a,
                    ax=ax,
                    axis="wire",
                    transformation=transformation,
                    components=components,
                    **kwargs_fill,
                )

            return na.plt.plot(
                a,
                ax=ax,
                axis="wire",
                transformation=transformation,
                components=components,
                **kwargs,
            )

        # the back of the substrate. The front is left open, so what is seen
        # through it is the inside of the back face, and the apertures drawn on
        # the surface itself stay on top of it.
        back = aperture.wire().explicit
        back.z = na.broadcast_to(depth, na.shape(back.z))
        result["back"] = draw(back, fill=True)

        axis = "vertex"

        vertices = getattr(aperture, "vertices", None)
        if vertices is not None:
            # An aperture can be placed within its surface, and `wire` returns
            # points which already account for that while `vertices` returns
            # them as the shape was defined. Without this the wall is built
            # from corners which are not where the faces are.
            if aperture.transformation is not None:
                vertices = aperture.transformation(vertices)
        else:
            # An aperture with no corners, a circle for instance, is walled
            # from the samples along its edge instead, which `wire` gives
            # already placed.
            vertices = aperture.wire().explicit.combine_axes(
                axes=("wire",),
                axis_new=axis,
            )

        front = vertices.copy_shallow()
        front.z = sag(front)
        behind = vertices.copy_shallow()
        behind.z = na.broadcast_to(depth, na.shape(front.z))

        if solid:
            # the wall between the faces, a panel at a time so that each is
            # sorted by depth on its own
            rolled = {
                axis: na.ScalarArray(
                    ndarray=np.roll(np.arange(front.shape[axis]), -1),
                    axes=(axis,),
                )
            }
            panel = np.stack(
                arrays=[front, front[rolled], behind[rolled], behind],
                axis="wire",
            )
            result["wall"] = draw(panel, fill=True)
        else:
            # seen flat there is no wall to speak of, only the edge of the
            # substrate showing at each vertex
            result["edges"] = draw(np.stack([front, behind], axis="wire"))

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
