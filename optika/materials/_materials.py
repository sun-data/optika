from __future__ import annotations
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika.mixins

__all__ = [
    "AbstractMaterial",
    "Vacuum",
    "Mirror",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMaterial(
    optika.mixins.Printable,
    optika.transforms.Transformable,
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

    def refract_rays(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        sag: optika.sags.AbstractSag,
        rulings: None | optika.rulings.AbstractRulings,
    ) -> optika.rays.RayVectorArray:
        rays_input = rays

        rays = sag.intercept(rays)

        distance = (rays.position - rays_input.position).length
        depth = rays.attenuation * distance
        rays.intensity = rays.intensity * depth * self.transmissivity(rays)

        n2 = self.index_refraction(rays)
        k2 = self.attenuation(rays)

        if rulings is not None:
            rays = rulings.rays_apparent(rays, index_refraction=n2)

        a = rays.direction
        n1 = rays.index_refraction
        normal = sag.normal(rays.position)

        r = n1 / n2
        c = -a @ normal
        b = r * a + (r * c - np.sqrt(1 - np.square(r) * (1 - np.square(c)))) * normal

        rays.direction = b / b.length
        rays.index_refraction = n2
        rays.attenuation = k2

        return rays


@dataclasses.dataclass(eq=False, repr=False)
class Vacuum(
    AbstractMaterial,
):
    @property
    def transform(self) -> None:
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


@dataclasses.dataclass(eq=False, repr=False)
class Mirror(
    AbstractMaterial,
):
    @property
    def transform(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return -rays.index_refraction

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
