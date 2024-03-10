from __future__ import annotations
import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from ._layers import Layer

__all__ = [
    "AbstractMaterial",
    "Vacuum",
    "AbstractMirror",
    "Mirror",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMaterial(
    optika.mixins.Printable,
    optika.mixins.Transformable,
):
    @abc.abstractmethod
    def index_refraction(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the index of refraction of this material for the given input rays

        Parameters
        ----------
        rays
            input rays used to evaluate the index of refraction
        """

    @abc.abstractmethod
    def attenuation(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        the attenuation coefficient of the given rays

        Parameters
        ----------
        rays
            input rays to calculate the attenuation coefficient for
        """

    @abc.abstractmethod
    def efficiency(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        The fraction of light that passes through the interface.

        Parameters
        ----------
        rays
            the input rays to calculate the efficiency for
        normal
            the vector perpendicular to the optical surface
        """

    @property
    @abc.abstractmethod
    def is_mirror(self) -> bool:
        """
        flag controlling whether this material reflects or transmits light
        """


@dataclasses.dataclass(eq=False, repr=False)
class Vacuum(
    AbstractMaterial,
):
    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 0 / u.mm

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMirror(
    AbstractMaterial,
):

    @property
    @abc.abstractmethod
    def substrate(self) -> None | Layer:
        """
        A layer representing the substrate supporting the reflective surface.
        """

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.index_refraction

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.attenuation

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return True


@dataclasses.dataclass(eq=False, repr=False)
class Mirror(
    AbstractMirror,
):
    substrate: None | Layer = None
    """A layer representing the substrate supporting the reflective surface."""
