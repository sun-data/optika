import abc
import dataclasses
import named_arrays as na
import optika
from . import AbstractLayer, AbstractMultilayerFilm, meshes

__all__ = [
    "AbstractThinFilmFilter",
    "ThinFilmFilter",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractThinFilmFilter(
    AbstractMultilayerFilm,
):
    """
    An interface describing a thin-film filter.
    """

    @property
    @abc.abstractmethod
    def layer(self) -> AbstractLayer:
        """The main layer of bulk material comprising this filter."""

    @property
    @abc.abstractmethod
    def layer_oxide(self) -> AbstractLayer:
        """The oxide layer that lies on either side of the bulk layer."""

    @property
    def layers(self) -> list[AbstractLayer]:
        layer = self.layer
        layer_oxide = self.layer_oxide
        return [
            layer_oxide,
            layer,
            layer_oxide,
        ]

    @property
    @abc.abstractmethod
    def mesh(self) -> meshes.AbstractMesh:
        """The mesh backing supporting this thin-film filter."""

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> float | na.AbstractScalar:
        result = super().efficiency(
            rays=rays,
            normal=normal,
        )
        return result * self.mesh.efficiency


@dataclasses.dataclass(eq=False, repr=False)
class ThinFilmFilter(
    AbstractThinFilmFilter,
):
    """
    A model of a thin-film EUV filter, such as those manufactured by Luxel
    :cite:p:`Powell1990`.

    Examples
    --------
    Plot the transmissivity of an aluminum thin-film filter, with a thickness
    of 100 nm and an oxide layer of 2 nm.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a model of the thin-film filter
        film = optika.materials.ThinFilmFilter(
            layer=optika.materials.Layer(
                chemical="Al",
                thickness=100 * u.nm,
            ),
            layer_oxide=optika.materials.Layer(
                chemical="Al2O3",
                thickness=2 * u.nm,
            ),
            mesh=optika.materials.meshes.Mesh(
                chemical="Ni",
                efficiency=0.8,
                pitch=70 / u.mm,
            ),
        )

        # Define the wavelength of the incident light
        wavelength = na.linspace(100, 1000, axis="wavelength", num=1001) * u.AA

        # Define the rays incident on the filter
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Compute the transmissivity of the filter
        transmissivity = film.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the transmissivity as a function of wavelength
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            transmissivity,
            ax=ax,
            axis="wavelength",
        );
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("transmissivity");
    """

    layer: AbstractLayer = dataclasses.MISSING
    """The main layer of bulk material comprising this filter."""

    layer_oxide: AbstractLayer = dataclasses.MISSING
    """The oxide layer that lies on either side of the bulk layer."""

    mesh: meshes.AbstractMesh = dataclasses.MISSING
    """The mesh backing supporting this thin-film filter."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.layer),
            optika.shape(self.layer_oxide),
            optika.shape(self.mesh),
        )
