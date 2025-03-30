import abc
import dataclasses
from dataclasses import MISSING
import numpy as np
import scipy.special
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractRulingSpacing

__all__ = [
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
    "SinusoidalRulings",
    "SquareRulings",
    "SawtoothRulings",
    "TriangularRulings",
    "RectangularRulings",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """
    Interface for the interaction of a ruled surface with incident light
    """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> int | na.AbstractScalar:
        """
        the diffraction order to simulate
        """

    @property
    @abc.abstractmethod
    def spacing(
        self,
    ) -> u.Quantity | na.AbstractScalar | AbstractRulingSpacing:
        """
        Spacing between adjacent rulings at the given position.
        """

    @property
    def spacing_(self) -> AbstractRulingSpacing:
        """
        A normalized version of :attr:`spacing` that is guaranteed to be
        an instance of :class:`optika.rulings.AbstractRulingSpacing`.
        """
        spacing = self.spacing
        if not isinstance(spacing, optika.rulings.AbstractRulingSpacing):
            spacing = optika.rulings.ConstantRulingSpacing(
                constant=spacing,
                normal=na.Cartesian3dVectorArray(1, 0, 0),
            )
        return spacing

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        """
        The fraction of light that is diffracted into a given order.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.
        """


@dataclasses.dataclass(eq=False, repr=False)
class Rulings(
    AbstractRulings,
):
    """
    An idealized set of rulings which have perfect efficiency in all diffraction
    orders.
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float:
        return 1


@dataclasses.dataclass(eq=False, repr=False)
class MeasuredRulings(
    AbstractRulings,
):
    """
    A set of rulings where the efficiency has been measured or calculated
    by an independent source.
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    efficiency_measured: na.FunctionArray[
        na.SpectralDirectionalVectorArray,
        na.AbstractScalar,
    ] = MISSING
    """The discrete measurements of the efficiency."""

    @property
    def shape(self) -> dict[str, int]:
        axis_wavelength = self.efficiency_measured.inputs.wavelength.axes
        shape = optika.shape(self.efficiency_measured.outputs)
        for ax in axis_wavelength:
            shape.pop(ax, None)
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.diffraction_order),
            shape,
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        measurement = self.efficiency_measured

        wavelength = measurement.inputs.wavelength
        direction = measurement.inputs.direction
        efficiency = measurement.outputs

        if direction.size != 1:  # pragma: nocover
            raise ValueError(
                "Interpolating over different incidence angles is not supported."
            )

        if wavelength.ndim != 1:  # pragma: nocover
            raise ValueError(
                f"wavelength must be one dimensional, got shape {wavelength.shape}"
            )

        return na.interp(
            x=rays.wavelength,
            xp=wavelength,
            fp=efficiency,
        )


@dataclasses.dataclass(eq=False, repr=False)
class SinusoidalRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a sinusoidal wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of sinusoidal rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SinusoidalRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), sinusoidal rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = J_i^2(2 \gamma)

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`J_i(x)` is a Bessel function of the first kind,
        :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """

        normal_rulings = self.spacing_(rays.position, normal).normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        wavelength = rays.wavelength
        cos_theta = -direction @ normal
        d = self.depth
        i = self.diffraction_order

        gamma = np.pi * d / (wavelength * cos_theta)

        result = scipy.special.jv(i, 2 * gamma)

        return result


@dataclasses.dataclass(eq=False, repr=False)
class SquareRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a square wave with a 50% duty cycle.

    Examples
    --------

    Compute the 1st-order groove efficiency of square rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SquareRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), square rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                \cos^2(\pi \gamma / 2) & i = 0 \\
                0 & i = \text{even} \\
                (2 / i \pi)^2 \sin^2 (\pi \gamma / 2) & i = \text{odd}, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """

        normal_rulings = self.spacing_(rays.position, normal).normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        wavelength = rays.wavelength
        cos_theta = -direction @ normal
        amplitude = np.pi / 4
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = np.pi * d / (wavelength * cos_theta)

        result = np.where(
            i % 2 == 0,
            x=0,
            y=np.square(2 * np.sin(np.pi * gamma / 2 * u.rad) / (i * np.pi)),
        )
        result = np.where(
            i == 0,
            x=np.square(np.cos(np.pi * gamma / 2 * u.rad)),
            y=result,
        )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class SawtoothRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a sawtooth wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of sawtooth rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.SawtoothRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=-1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), sawtooth rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = [\pi (\gamma + i)]^{-2} \sin^2(\pi \gamma)

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """
        i = self.diffraction_order
        d = self.depth

        spacing = self.spacing_(rays.position, normal)

        L = spacing.length
        normal_rulings = spacing.normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        amplitude = np.pi / 2
        wavelength = rays.wavelength
        cos_alpha = -direction @ normal
        # alpha = np.arccos(cos_alpha)
        # print(f"{alpha.to(u.deg)=}")
        # sin_theta = np.sin(alpha)
        # sin_beta = i * wavelength / L - sin_theta
        # beta = np.arcsin(sin_beta)
        # print(f"{beta.to(u.deg)=}")
        # cos_beta = np.cos(beta)
        # n1 = (1 + cos_beta) / (np.pi)
        n1 = 1 / amplitude

        cos_theta = optika.materials.snells_law_scalar(
            cos_incidence=cos_alpha,
            index_refraction=rays.index_refraction,
            index_refraction_new=rays.index_refraction + n1,
        )
        # cos_theta = cos_alpha

        print(f"{cos_theta=}")

        gamma = np.pi * d * n1 / (wavelength * cos_theta)
        print(f"{gamma=}")

        result = np.square(np.sin(np.pi * gamma * u.rad) / (np.pi * (gamma + i)))

        return result


@dataclasses.dataclass(eq=False, repr=False)
class TriangularRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a triangle wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of triangular rulings with a groove
    density of 2500 grooves/mm and a groove depth of 15 nm.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.TriangularRulings(
            spacing=1 / density,
            depth=depth,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), triangular rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                \{\gamma / [(\pi \gamma / 2)^2 - i^2]\}^2 \sin^2 (\pi^2 \gamma / 4) & i = \text{even} \\
                \{\gamma / [(\pi \gamma / 2)^2 - i^2]\}^2 \cos^2 (\pi^2 \gamma / 4) & i = \text{odd}, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order
        :math:`i`, :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """

        normal_rulings = self.spacing_(rays.position, normal).normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        wavelength = rays.wavelength
        cos_theta = -direction @ normal
        amplitude = np.square(np.pi) / 8
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = np.pi * d / (wavelength * cos_theta)

        a = gamma / (np.square(np.pi * gamma / 2) + np.square(i))

        result = np.where(
            i % 2 == 0,
            x=np.square(a * np.sin(np.square(np.pi) * gamma / 4 * u.rad)),
            y=np.square(a * np.cos(np.square(np.pi) * gamma / 4 * u.rad)),
        )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class RectangularRulings(
    AbstractRulings,
):
    r"""
    A ruling profile described by a rectangular wave.

    Examples
    --------

    Compute the 1st-order groove efficiency of rectangular rulings with a groove
    density of 2500 grooves/mm, a groove depth of 15 nm, and a duty cycle of
    30 percent.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the groove density
        density = 2500 / u.mm

        # Define the groove depth
        depth = 15 * u.nm

        # Define ruling model
        rulings = optika.rulings.RectangularRulings(
            spacing=1 / density,
            depth=depth,
            ratio_duty=0.3,
            diffraction_order=1,
        )

        # Define the wavelengths at which to sample the groove efficiency
        wavelength = na.geomspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the incidence angles at which to sample the groove efficiency
        angle = na.linspace(0, 30, num=3, axis="angle") * u.deg

        # Define the light rays incident on the grooves
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle),
                y=0,
                z=np.cos(angle),
            ),
        )

        # Compute the efficiency of the grooves for the given wavelength
        efficiency = rulings.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the groove efficiency as a function of wavelength
        fig, ax = plt.subplots()
        angle_str = angle.value.astype(str).astype(object)
        na.plt.plot(
            wavelength,
            efficiency,
            ax=ax,
            axis="wavelength",
            label=r"$\theta$ = " + angle_str + f"{angle.unit:latex_inline}",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel(f"efficiency");
        ax.legend();
    """

    spacing: u.Quantity | na.AbstractScalar | AbstractRulingSpacing = MISSING
    """Spacing between adjacent rulings at the given position."""

    depth: u.Quantity | na.AbstractScalar = MISSING
    """Depth of the ruling pattern."""

    ratio_duty: u.Quantity | na.AbstractScalar = MISSING
    """The duty cycle of the ruling pattern."""

    diffraction_order: int | na.AbstractScalar = MISSING
    """The diffraction order to simulate."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.spacing),
            optika.shape(self.depth),
            optika.shape(self.ratio_duty),
            optika.shape(self.diffraction_order),
        )

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        r"""
        The fraction of light diffracted into a given order.

        Calculated using the expression given in Table 1 of :cite:t:`Magnusson1978`.

        Parameters
        ----------
        rays
            The light rays incident on the rulings
        normal
            The vector normal to the surface on which the rulings are placed.

        Notes
        -----

        The theoretical efficiency of thin (wavelength much smaller than
        the groove spacing), rectangular rulings is given by Table 1 of
        :cite:t:`Magnusson1978`,

        .. math::

            \eta_i = \begin{cases}
                1 - [(2 a / \pi) - (a / \pi)^2] \sin^2 \left\{ \pi \gamma / [2 (1 - \cos a)]^{1/2} \right\} & i = 0 \\
                [2 / (i \pi)^2](1 - \cos i a) \sin^2 \left\{ \pi \gamma / [2 (1 - \cos a)]^{1/2} \right\} & i \ne 0, \\
            \end{cases}

        where :math:`\eta_i` is the groove efficiency for diffraction order :math:`i`,
        :math:`a` :math:`(0 < a < 2 \pi)` is the duty cycle of the rectangular wave,
        :math:`\gamma = \pi d n_1 / \lambda \cos \theta` is the
        normalized amplitude of the fundamental grating, :math:`d` is the
        thickness of the grating, :math:`n_1` is the amplitude of the fundamental
        grating, :math:`\lambda` is the free-space wavelength of the incident
        light, and :math:`\theta` is the angle of incidence inside the medium.
        """

        normal_rulings = self.spacing_(rays.position, normal).normalized

        parallel_rulings = normal.cross(normal_rulings).normalized

        direction = rays.direction
        direction = direction - direction @ parallel_rulings

        wavelength = rays.wavelength
        cos_theta = -direction @ normal
        a = 2 * np.pi * self.ratio_duty
        amplitude = np.pi / (2 * np.sqrt(2 * (1 - np.cos(a))))
        d = self.depth / amplitude
        i = self.diffraction_order

        gamma = np.pi * d / (wavelength * cos_theta)

        b = np.sin(np.pi * gamma / np.sqrt(2 * (1 - np.cos(a * u.rad))) * u.rad)
        b = np.square(b)

        result = np.where(
            i == 0,
            x=1 - ((2 * a / np.pi) - np.square(a / np.pi)) * b,
            y=(2 / np.square(i * np.pi)) * (1 - np.cos(i * a * u.rad)) * b,
        )

        return result
