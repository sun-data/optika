import abc
import dataclasses
import functools
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractFieldStopModel",
    "ApertureFieldStopModel",
    "PolynomialFieldStopModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFieldStopModel(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    """
    An interface describing the field of view of an optical system: which
    points of the scene it passes light from, at each wavelength.

    Unlike an :class:`~optika.apertures.AbstractAperture`, which is a shape in
    the frame of a single surface, a field-stop model is a function of scene
    coordinates, wavelength included.  The field of view of a system whose
    field stop sits behind a dispersive element moves across the scene with
    wavelength, and a model evaluated at each point of the scene at that
    point's own wavelength can follow it.
    """

    @abc.abstractmethod
    def __call__(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        """
        Whether each point of the scene lies inside the field of view.

        Parameters
        ----------
        coordinates
            The wavelength and position of each point in the scene.
        """

    @abc.abstractmethod
    def wire(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        num: None | int = None,
    ) -> na.AbstractCartesian2dVectorArray:
        """
        The outline of the field of view at the given wavelengths, as a
        sequence of points in field coordinates along the logical axis
        ``wire``.

        Parameters
        ----------
        wavelength
            The wavelengths at which to outline the field of view.
        num
            The number of points along each edge of the outline, see
            :meth:`optika.apertures.AbstractAperture.wire`.
        """


@dataclasses.dataclass(eq=False, repr=False)
class ApertureFieldStopModel(
    AbstractFieldStopModel,
):
    """
    A field of view which is the same at every wavelength, bounded by an
    aperture in field coordinates.

    This is the field of view of a system whose field stop sits ahead of
    every dispersive element, and the form an
    :class:`~optika.apertures.AbstractAperture` given as the
    :attr:`~optika.systems.LinearSystem.field_stop` of a linear system takes.
    """

    aperture: optika.apertures.AbstractAperture = dataclasses.MISSING
    """The outline of the field of view, in field coordinates."""

    @property
    def shape(self) -> dict[str, int]:
        return optika.shape(self.aperture)

    def __call__(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        position = coordinates.position
        return self.aperture(
            position=na.Cartesian3dVectorArray(x=position.x, y=position.y),
        )

    def wire(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        num: None | int = None,
    ) -> na.AbstractCartesian2dVectorArray:
        wire = self.aperture.wire(num=num)
        return na.Cartesian2dVectorArray(x=wire.x, y=wire.y)


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialFieldStopModel(
    AbstractFieldStopModel,
):
    """
    A polygonal field of view whose vertices move with wavelength, each fit
    by a polynomial in wavelength through its position at known wavelengths.

    A vertex is the same point on the edge of the field stop at every
    wavelength, and dispersion moves it smoothly, so each can be fit on its
    own.  Vertices which do not vary with wavelength are used as they are,
    and the field of view is then the same at every wavelength.

    Examples
    --------

    Outline a square field of view which drifts across the scene with
    wavelength, known at three wavelengths, at five others.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        wavelength = na.linspace(500, 600, axis="wavelength", num=3) * u.nm
        corners = na.linspace(45, 405, axis="vertex", num=5) * u.deg
        drift = (wavelength - 550 * u.nm) * (0.01 * u.deg / u.nm)

        model = optika.radiometry.PolynomialFieldStopModel(
            wavelength=wavelength,
            vertices=na.Cartesian2dVectorArray(
                x=0.5 * u.deg * np.cos(corners) + drift,
                y=0.5 * u.deg * np.sin(corners),
            ),
            axis_wavelength="wavelength",
        )

        wire = model.wire(na.linspace(480, 620, axis="wavelength", num=5) * u.nm)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(wire.x, wire.y, axis="wire", ax=ax)
            ax.set_aspect("equal")
    """

    wavelength: na.AbstractScalar = dataclasses.MISSING
    """The wavelengths at which the outline of the field of view is known."""

    vertices: na.AbstractCartesian2dVectorArray = dataclasses.MISSING
    """
    The vertices of the outline at each of :attr:`wavelength`, in field
    coordinates and in order along the logical axis ``vertex``.
    """

    axis_wavelength: str = dataclasses.MISSING
    """The logical axis corresponding to changing wavelength."""

    degree: int = 2
    """
    The degree of the polynomial fit to each vertex.

    Held one below the number of wavelengths, the most that many samples can
    determine.
    """

    @property
    def shape(self) -> dict[str, int]:
        shape = na.broadcast_shapes(
            optika.shape(self.wavelength),
            optika.shape(self.vertices),
        )
        return {
            ax: n
            for ax, n in shape.items()
            if ax not in (self.axis_wavelength, "vertex")
        }

    @functools.cached_property
    def fit(self) -> na.PolynomialFitFunctionArray:
        """The polynomial fit mapping wavelength to each vertex of the outline."""
        wavelength = self.wavelength
        num = na.broadcast_shapes(
            optika.shape(wavelength),
            optika.shape(self.vertices),
        )[self.axis_wavelength]
        return na.PolynomialFitFunctionArray.from_degree(
            inputs=wavelength,
            outputs=self.vertices,
            degree=min(self.degree, num - 1),
            center=wavelength.mean(self.axis_wavelength),
            axis_polynomial=self.axis_wavelength,
        )

    def polygon(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> optika.apertures.PolygonalAperture:
        """
        The outline of the field of view at the given wavelengths, as a
        polygonal aperture in field coordinates.

        Parameters
        ----------
        wavelength
            The wavelengths at which to outline the field of view.
        """
        vertices = self.vertices
        if self.axis_wavelength in na.shape(vertices):
            vertices = self.fit(wavelength).outputs
        return optika.apertures.PolygonalAperture(
            vertices=na.Cartesian3dVectorArray(
                x=vertices.x,
                y=vertices.y,
                z=0 * vertices.x,
            ),
        )

    def __call__(
        self,
        coordinates: na.AbstractSpectralPositionalVectorArray,
    ) -> na.AbstractScalar:
        position = coordinates.position
        return self.polygon(coordinates.wavelength)(
            position=na.Cartesian3dVectorArray(x=position.x, y=position.y),
        )

    def wire(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        num: None | int = None,
    ) -> na.AbstractCartesian2dVectorArray:
        wire = self.polygon(wavelength).wire(num=num)
        return na.Cartesian2dVectorArray(x=wire.x, y=wire.y)
