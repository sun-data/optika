from __future__ import annotations
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractMaterial",
    "Vacuum",
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
    def transmissivity(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> na.ScalarLike:
        """
        The fraction of light that passes through the interface.

        Parameters
        ----------
        rays
            the input rays to calculate the transmissivity for
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

    def transmissivity(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return False


@dataclasses.dataclass(eq=False, repr=False)
class Mirror(
    AbstractMaterial,
):
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

    def transmissivity(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return 1

    @property
    def is_mirror(self) -> bool:
        return True
