from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import named_arrays as na
import optika

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
    optika.mixins.Plottable,
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.propagators.AbstractLightPropagator,
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
    def is_stop(self) -> bool:
        """
        If this surface is pupil stop or the field stop, return :obj:`True`.
        """
        return self.is_field_stop or self.is_pupil_stop

    def propagate_rays(
        self,
        rays: optika.rays.RayVectorArray,
        # material: None | optika.materials.AbstractMaterial = None,
    ) -> optika.rays.RayVectorArray:
        r"""
        Refract, reflect, and/or diffract the given rays off of this surface

        Parameters
        ----------
        rays
            a set of input rays that will interact with this surface

        Notes
        -----

        Our goal is to derive a 3D version of Snell's law that can model the
        diffraction from a periodic ruling pattern (diffraction grating).

        To start, consider an incident wave of the form:

        .. math::
            :label: incident-wave

            E_1(\mathbf{r}) = A_1 e^{i \mathbf{k}_1 \cdot \mathbf{r}},

        where :math:`E_1` is the magnitude of the incident electric field,
        :math:`A_1` is the amplitude of the incident wave, and :math:`\mathbf{k}_1` is
        the incident wavevector.

        Now define an interface at :math:`z = 0`, where the index of
        refraction changes, and/or there is a periodic ruling pattern inscribed.
        When the incident wave interacts with this interface, it will create an
        output wave of the form:

        .. math::
            :label: transmitted-wave

            E_2(\mathbf{r}) = A_2 e^{i \mathbf{k}_2 \cdot \mathbf{r}},

        where :math:`E_2` is the magnitude of the output electric field,
        :math:`A_2` is the amplitude of the output wave, and :math:`\mathbf{k}_2` is
        the output wavevector.

        Note in this case we care about only the reflected `or` the transmitted
        wave, not both, since we're only concerned with sequential optics.

        If the interface at :math:`z = 0` `doesn't` have a periodic ruling pattern,
        :math:`E_1` and :math:`E_2` satisfy homogenous Dirichlet boundary conditions.

        .. math::
            :label: boundary-condition

            A_1 \exp\left[i \mathbf{k}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
            = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]

        To include the ruling pattern, we model it as a phase shift of the wave at the interface,

        .. math::
            :label: phase-shift

            \phi(x, y) = i \boldsymbol{\kappa} \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})

        where the vector

        .. math::

            \boldsymbol{\kappa} = -\frac{2 \pi m}{d} \hat{\boldsymbol{\kappa}},

        :math:`m` is the diffraction order,
        :math:`d` is the groove spacing,
        and :math:`\hat{\boldsymbol{\kappa}}` is a unit vector normal to the
        planes of the rulings.

        With the inclusion of Equation :eq:`phase-shift`, Equation :eq:`boundary-condition` becomes:

        .. math::
            :label: boundary-condition-shifted

             A_1 \exp\left[i (\mathbf{k}_1 + \boldsymbol{\kappa}) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
            = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]

        For Equation :eq:`boundary-condition-shifted` to be true everywhere in
        the :math:`x`-:math:`y` plane, the exponents must be equal:

        .. math::
            :label: boundary-condition-exponents

            (\mathbf{k}_1 + \boldsymbol{\kappa}) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
            = \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

        If we take :math:`\mathbf{k}_i = k_i \hat{\mathbf{k}}_i = n_i k_0 \hat{\mathbf{k}}_i` for :math:`i=(1,2)`,
        where :math:`k_i` is the incident/output wavenumber,
        :math:`n_i` is the incident/output index of refraction,
        :math:`k_0` is the wavenumber in vacuum,
        and :math:`\hat{\mathbf{k}}_i` is the incident/output propagation direction,
        we get an expression in terms of the output direction, :math:`\hat{\mathbf{k}}_2`:

        .. math::
            :label: boundary-directions

            n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
            = n_2 \hat{\mathbf{k}}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

        Now, Equation :eq:`boundary-directions` can only be satisfied
        if the components are separately equal since if :math:`y=0`

        .. math::
            :label: k_x

            n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot \hat{\mathbf{x}}
            = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{x}},

        and if :math:`x=0`

        .. math::
            :label: k_y

            n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot \hat{\mathbf{y}}
            = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{y}}.

        So, if we define an effective incident propagation direction

        .. math::

            \mathbf{k}_\text{e} = \hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1,

        we can collect Equations :eq:`k_x` and :eq:`k_y` into a single vector
        equation

        .. math::
            :label: snells-law

            n_1 [ \mathbf{k}_\text{e} - ( \mathbf{k}_\text{e} \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ]
            = n_2 [ \hat{\mathbf{k}}_2 - ( \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ],

        where :math:`\hat{\mathbf{n}} = \hat{\mathbf{z}}` is the vector normal to
        the interface.

        Our goal now is to solve Equation :eq:`snells-law` for
        the output propagation direction, :math:`\hat{\mathbf{k}}_2`,
        and the only unknown is the component of the output propagation direction
        parallel to the surface normal vector, :math:`(\hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} )`.
        We can write this in terms of a cross product,

        .. math::
            :label: k2_dot_n


            \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} = \pm \sqrt{1 - |\hat{\mathbf{k}}_2 \times \hat{\mathbf{n}}|^2},

        where the :math:`\pm` represents a transmission or reflection, and the
        cross product :math:`\mathbf{k}_2 \times \hat{\mathbf{n}}` can be found by crossing
        Equation :eq:`snells-law` with :math:`\hat{\mathbf{n}}`,

        .. math::
            :label: k2_cross_n

            \hat{\mathbf{k}}_2 \times \hat{\mathbf{n}}
            = \frac{n_1}{n_2} \mathbf{k}_\text{e} \times \hat{\mathbf{n}},

        which leads to Equation :eq:`k2_dot_n` becoming:

        .. math::
            :label: k2_dot_n_expanded

            \mathbf{k}_2 \cdot \hat{\mathbf{n}}
            &= \pm \sqrt{1 - \left( \frac{n_1}{n_2} \right)^2 |\mathbf{k}_\text{e} \times \hat{\mathbf{n}}|^2} \\
            &= \pm \frac{n_1}{n_2} \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}})^2 - k_\text{e}^2 }

        By plugging Equation :eq:`k2_dot_n_expanded` into Equation :eq:`snells-law`,
        and solving for :math:`\hat{\mathbf{k}}_2`, we achieve our goal, an expression for the output ray
        in terms of the input ray and other known quantities.

        .. math::

            \hat{\mathbf{k}}_2
            = \frac{n_1}{n_2} \left[ \mathbf{k}_\text{e}
            + \left(
                \left( -\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}} \right)
                \pm \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}})^2 - k_\text{e}^2 }
            \right) \hat{\mathbf{n}} \right]
        """
        sag = self.sag
        material = self.material
        aperture = self.aperture
        rulings = self.rulings
        transformation = self.transformation

        if transformation is not None:
            rays = transformation.inverse(rays)

        rays_1 = sag.intercept(rays)

        wavelength_1 = rays_1.wavelength
        position_1 = rays_1.position
        a = rays_1.direction
        attenuation_1 = rays_1.attenuation
        displacement = position_1 - rays.position
        intensity_1 = rays_1.intensity * np.exp(-attenuation_1 * displacement.length)
        n1 = rays_1.index_refraction

        position_2 = position_1
        n2 = material.index_refraction(rays_1)
        r = n1 / n2

        wavelength_2 = wavelength_1 / r

        if rulings is not None:
            m = rulings.diffraction_order
            d = rulings.spacing(position_1)
            g = rulings.normal(position_1)
            a = a - (m * wavelength_2 * g) / (n1 * d)

        normal = sag.normal(position_1)

        c = -a @ normal

        mirror = np.sign(a.z) * (2 * material.is_mirror - 1)

        t = np.sqrt(np.square(1 / r) + np.square(c) - np.square(a.length))
        b = r * (a + (c + mirror * t) * normal)

        intensity_2 = intensity_1 * material.transmissivity(rays_1)
        attenuation_2 = material.attenuation(rays_1)

        rays_2 = optika.rays.RayVectorArray(
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
    transformation: None | na.transformations.AbstractTransformation = None
    kwargs_plot: None | dict = None

    def __post_init__(self):
        if self.sag is None:
            self.sag = optika.sags.NoSag()
        if self.material is None:
            self.material = optika.materials.Vacuum()
