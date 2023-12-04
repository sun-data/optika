import abc
import dataclasses
import functools
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractMaterial

__all__ = [
    "AbstractWindtMaterial",
    "Silicon",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractWindtMaterial(
    AbstractMaterial,
):
    @property
    def transformation(self) -> None:
        return None

    @property
    @abc.abstractmethod
    def file_nk(self) -> pathlib.Path:
        """path to the ``.nk`` file storing the index of refraction"""

    @functools.cached_property
    def wavelength_n_k(self) -> tuple[na.ScalarArray, na.ScalarArray, na.ScalarArray]:
        """
        Return the wavelength, :math:`n`, and :math:`k` from the ``.nk`` file.
        """
        wavelength, n, k = np.genfromtxt(
            fname=self.file_nk,
            skip_header=8,
            unpack=True,
        )
        wavelength = wavelength << u.AA
        wavelength = na.ScalarArray(wavelength, axes="wavelength")
        n = na.ScalarArray(n, axes="wavelength")
        k = na.ScalarArray(k, axes="wavelength")
        return wavelength, n, k

    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        wavelength, n, k = self.wavelength_n_k
        return na.interp(
            rays.wavelength,
            xp=wavelength,
            fp=n,
        )

    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        wavelength, n, k = self.wavelength_n_k
        attenuation = 4 * np.pi * k / wavelength
        return na.interp(rays.wavelength, xp=wavelength, fp=attenuation)

    def transmissivity(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1


@dataclasses.dataclass(eq=False, repr=False)
class Silicon(AbstractWindtMaterial):
    is_mirror: bool = False

    @property
    def file_nk(self) -> pathlib.Path:
        return pathlib.Path(__file__).parent / "nk/Si.nk"

    @functools.cached_property
    def wavelength_n_k(self) -> tuple[na.ScalarArray, na.ScalarArray, na.ScalarArray]:
        """
        Return the wavelength, :math:`n`, and :math:`k` from the ``.nk`` file.

        Examples
        --------

        Plot :math:`n` and :math:`k` as a function of wavelength

        .. jupyter-execute::

            import named_arrays as na
            import optika
            import astropy.visualization

            si = optika.materials.Silicon()

            wavelength, n, k = si.wavelength_n_k

            with astropy.visualization.quantity_support():
                fig, axs = na.plt.subplots(nrows=2, sharex=True, squeeze=True)
                ax_n = axs[dict(subplots_row=0)].ndarray
                ax_k = axs[dict(subplots_row=1)].ndarray
                na.plt.plot(wavelength, n, ax=ax_n, label="$n$");
                na.plt.plot(wavelength, k, ax=ax_k, color="tab:orange", label="$k$");
                ax_n.set_xscale("log");
                ax_k.set_yscale("log");
                ax_n.legend();
                ax_k.legend();
        """
        return super().wavelength_n_k
