import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractRulingSpacing",
    "ConstantRulingSpacing",
    "Polynomial1dRulingSpacing",
    "HolographicRulingSpacing",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulingSpacing(
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    """
    An interface describing the instantaneous ruling spacing on the surface
    of a diffraction grating.

    This is useful if you want to define a grating with variable line spacing.
    """

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The local ruling vector at the given position

        Parameters
        ----------
        position
            The location on the grating at which to evaluate the ruling spacing.
        normal
            The unit vector perpendicular to the surface at the given position.
        """


@dataclasses.dataclass(eq=False, repr=False)
class ConstantRulingSpacing(
    AbstractRulingSpacing,
):
    """
    The simplest type of ruling spacing, a constant distance between each ruling.
    """

    constant: u.Quantity | na.AbstractScalar
    """The constant describing the ruling spacing."""

    normal: na.AbstractCartesian3dVectorArray
    """The unit vector normal to the planes of the rulings."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.constant),
            optika.shape(self.normal),
        )

    @property
    def transformation(self) -> None:
        return None

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return self.constant * self.normal


@dataclasses.dataclass(eq=False, repr=False)
class Polynomial1dRulingSpacing(
    AbstractRulingSpacing,
):
    """
    Ruling spacing specified by a 1-dimensional polynomial.
    """

    coefficients: dict[int, u.Quantity | na.AbstractScalar]
    """
    The coefficients of the polynomial represented as a dictionary where
    the values are the coefficients and the keys are the power associated
    with each coefficient.
    """

    normal: na.AbstractCartesian3dVectorArray
    """The unit vector normal to the planes of the rulings."""

    transformation: None | na.transformations.AbstractTransformation = None
    """
    An arbitrary coordinate system transformation applied to the argument
    of the polynomial.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.coefficients),
            optika.shape(self.normal),
            optika.shape(self.transformation),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        coefficients = self.coefficients
        normal_rulings = self.normal
        transformation = self.transformation

        if transformation is not None:
            position = transformation(position)

        x = position @ normal_rulings

        result = 0 * u.mm
        for power, coefficient in coefficients.items():
            result = result + coefficient * (x**power)

        return result * normal_rulings


@dataclasses.dataclass(eq=False, repr=False)
class HolographicRulingSpacing(
    AbstractRulingSpacing,
):
    r"""
    Rulings created by interfering two beams.

    Examples
    --------

    Create some holographic rulings from two source points,
    launch rays from the first source point and confirm they are refocused
    onto the second source point.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Define the origins of the two recording beams
        x1 = na.Cartesian3dVectorArray(5, 0, -10) * u.mm
        x2 = na.Cartesian3dVectorArray(10, 0, -10) * u.mm

        # Define the wavelength of the recording beams
        wavelength = 500 * u.nm

        # Define the surface normal
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        # Define input rays emanating from the origin
        # of the first recording beam
        position = na.Cartesian3dVectorArray(
            x=na.linspace(-5, +5, axis="x", num=5) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm,
        )
        direction_input = position - x1

        # Initialize the holographic ruling spacing
        # representation
        rulings = optika.rulings.HolographicRulingSpacing(
            x1=x1,
            x2=x2,
            wavelength=wavelength,
        )

        # Evaluate the ruling spacing where
        # the rays strike the surface
        d = rulings(position, normal)

        # Compute the new direction of some diffracted
        # rays
        direction_output = optika.materials.snells_law(
            wavelength=wavelength,
            direction=direction_input.normalized,
            index_refraction=1,
            index_refraction_new=1,
            normal=normal,
            is_mirror=True,
            diffraction_order=1,
            spacing_rulings=d.length,
            normal_rulings=d.normalized,
        )
        direction_output = direction_output * 20 * u.mm

        # Plot the results
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                na.stack([x1, position], axis="t"),
                components=("z", "x"),
                axis="t",
                color="tab:blue",
            )
            na.plt.plot(
                na.stack([position, position + direction_output], axis="t"),
                components=("z", "x"),
                axis="t",
                color="tab:orange",
            )
            ax.axvline(0)
            ax.scatter(x1.z, x1.x)
            ax.scatter(x2.z, x2.x)

    Notes
    -----

    From :cite:t:`Welford1975`, the ruling spacing is given by

    .. math::

        \mathbf{d} = \frac{\lambda}{a} \mathbf{q} \times \mathbf{n}

    where :math:`\lambda` is the wavelength of the recording beams,
    :math:`\mathbf{n}` is a unit vector perpendicular to the surface,

    .. math::

        a \mathbf{q} = \mathbf{n} \times (\pm \mathbf{r}_1 \mp \mathbf{r}_2),

    :math:`\mathbf{r}_1` is a unit vector in the direction of the first
    recording beam,
    and :math:`\mathbf{r}_2` is a unit vector in the direction of the second
    recording beam.
    If rays are diverging from the origin of the recording beams,
    the top branch is used, otherwise the bottom branch is used.
    """

    x1: na.AbstractCartesian3dVectorArray
    """
    The origin of the first recording beam in local coordinates.
    """

    x2: na.AbstractCartesian3dVectorArray
    """
    The origin of the second recording beam in local coordinates.
    """

    wavelength: u.Quantity | na.AbstractScalar
    """
    The wavelength of the recording beams.
    """

    is_diverging_1: bool | na.AbstractScalar = True
    """
    A boolean flag indicating if rays are diverging from the origin of the
    first beam.
    """

    is_diverging_2: bool | na.AbstractScalar = False
    """
    A boolean flag indicating if rays are diverging from the origin of the
    second beam.
    """

    transformation: None | na.transformations.AbstractTransformation = None
    """
    A transformation from surface coordinates to ruling coordinates.
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.x1),
            optika.shape(self.x2),
            optika.shape(self.is_diverging_1),
            optika.shape(self.is_diverging_2),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:

        x1 = self.x1
        x2 = self.x2
        wavelength = self.wavelength
        d1 = self.is_diverging_1
        d2 = self.is_diverging_2
        n = normal

        d1 = 2 * d1 - 1
        d2 = 2 * d2 - 1

        r1 = position - x1
        r2 = position - x2

        r1 = d1 * r1.normalized
        r2 = d2 * r2.normalized

        dr = r1 - r2

        aq = n.cross(dr)

        a = aq.length
        q = aq / a

        spacing = wavelength / a

        result = spacing * q.cross(n)

        return result
