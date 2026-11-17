import dataclasses
import numpy as np
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
    optika.mixins.Replaceable,
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

    def count(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        r"""
        The number of rulings between the origin and the given position.

        This is the function :math:`N(\mathbf{r})` whose gradient along the
        surface is the local ruling density, :math:`\mathbf{d} / d^2`, where
        :math:`\mathbf{d}` is the ruling vector returned by :meth:`__call__`.
        It is zero at the origin and increases in the direction of
        :math:`\mathbf{d}`.

        Parameters
        ----------
        position
            The location on the grating at which to count the rulings.
        normal
            The unit vector perpendicular to the surface at the given position.

        Notes
        -----
        This implementation integrates the ruling density along the chord
        from the origin to each position,

        .. math::

            N(\mathbf{r}) = \int_0^1 \mathbf{r} \cdot
                \frac{\mathbf{d}(t \mathbf{r})}{d^2(t \mathbf{r})} \, dt,

        with 32-point Gauss-Legendre quadrature, evaluating the ruling vector
        with the normal at the end of the chord.
        It is exact for a ruling density which is a polynomial of degree 63
        or less in the position, and subclasses with a closed form override
        it.
        """
        nodes, weights = np.polynomial.legendre.leggauss(32)
        nodes = (nodes + 1) / 2
        weights = weights / 2

        result = 0
        for node, weight in zip(nodes, weights):
            d = self(node * position, normal)
            result = result + weight * (position @ d) / np.square(d.length)

        return result.to(u.dimensionless_unscaled)


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

    def count(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        return (position @ self.normal / self.constant).to(u.dimensionless_unscaled)


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

    def _coordinate(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        The argument of the polynomial at the given position.

        Parameters
        ----------
        position
            The location on the grating at which to evaluate the argument.
        """
        transformation = self.transformation
        if transformation is not None:
            position = transformation(position)
        return position @ self.normal

    def _spacing(
        self,
        x: na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The value of the polynomial, the distance between the rulings.

        Parameters
        ----------
        x
            The argument of the polynomial.
        """
        result = 0 * u.mm
        for power, coefficient in self.coefficients.items():
            result = result + coefficient * (x**power)
        return result

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian3dVectorArray:
        return self._spacing(self._coordinate(position)) * self.normal

    def count(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        The number of rulings between the origin and the given position.

        This is the integral of
        :meth:`optika.rulings.AbstractRulingSpacing.count`, computed with the
        same quadrature, but only on the argument of the polynomial, which
        changes linearly along the chord from the origin since
        :attr:`transformation` is affine.

        Parameters
        ----------
        position
            The location on the grating at which to count the rulings.
        normal
            The unit vector perpendicular to the surface at the given position.
            Ignored, since the spacing does not depend on it.
        """
        nodes, weights = np.polynomial.legendre.leggauss(32)
        nodes = (nodes + 1) / 2
        weights = weights / 2

        x_0 = self._coordinate(na.Cartesian3dVectorArray() * u.mm)
        x_1 = self._coordinate(position)

        result = 0
        for node, weight in zip(nodes, weights):
            result = result + weight / self._spacing(x_0 + node * (x_1 - x_0))

        result = result * (position @ self.normal)

        return result.to(u.dimensionless_unscaled)


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

        # Define input rays emanating from the origin of the first recording beam.
        position = na.Cartesian3dVectorArray(
            x=na.linspace(-5, +5, axis="x", num=5) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm,
        )
        direction_input = position - x1

        # Initialize the holographic ruling spacing representation.
        rulings = optika.rulings.HolographicRulingSpacing(
            x1=x1,
            x2=x2,
            wavelength=wavelength,
        )

        # Evaluate the ruling spacing where the rays strike the surface.
        d = rulings(position, normal)

        # Compute the effective propagation direction of the input rays
        direction_input = optika.rulings.incident_effective(
            wavelength=wavelength,
            direction=direction_input.normalized,
            index_refraction=1,
            normal=normal,
            diffraction_order=1,
            spacing_rulings=d.length,
            normal_rulings=d.normalized,
        )

        # Compute the output direction of the diffracted rays.
        direction_output = optika.materials.snells_law(
            direction=direction_input,
            index_refraction=1,
            index_refraction_new=1,
            normal=normal,
            is_mirror=True,
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

    where :math:`\lambda` is the wavelength of the recording beam,
    :math:`a` is an arbitrary scalar to be solved for,
    :math:`\mathbf{n}` is a unit vector perpendicular to the surface,
    and :math:`\mathbf{q}` is a unit vector parallel to the rulings.
    The vector :math:`a \mathbf{q}` is given by

    .. math::

        a \mathbf{q} = \mathbf{n} \times (\pm \mathbf{r}_1 \mp \mathbf{r}_2),

    where :math:`\mathbf{r}_1` is a unit vector in the direction of the first
    recording beam,
    and :math:`\mathbf{r}_2` is a unit vector in the direction of the second
    recording beam.
    If rays are diverging from the origin of the recording beams,
    the top branch is used, otherwise the bottom branch is used.
    Since :math:`\mathbf{q}` is a unit vector,

    .. math::

        a = | a \mathbf{q} |.

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

    def count(
        self,
        position: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        r"""
        The number of rulings between the origin and the given position,
        the number of fringes between them in the interference pattern of
        the two recording beams,

        .. math::

            N(\mathbf{r}) = \frac{\pm |\mathbf{r} - \mathbf{x}_1|
                \mp |\mathbf{r} - \mathbf{x}_2|}{\lambda} - N(0),

        with the signs of :meth:`__call__`.

        Parameters
        ----------
        position
            The location on the grating at which to count the rulings.
        normal
            The unit vector perpendicular to the surface at the given position.
            Ignored, since the interference pattern does not depend on it.
        """
        x1 = self.x1
        x2 = self.x2
        d1 = 2 * self.is_diverging_1 - 1
        d2 = 2 * self.is_diverging_2 - 1

        def phase(r: na.AbstractCartesian3dVectorArray) -> na.AbstractScalar:
            return d1 * (r - x1).length - d2 * (r - x2).length

        result = (phase(position) - phase(0 * position)) / self.wavelength

        return result.to(u.dimensionless_unscaled)
