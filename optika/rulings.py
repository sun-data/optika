import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import optika.mixins

__all__ = [
    "AbstractRulings",
    "AbstractConstantDensityRulings",
    "ConstantDensityRulings",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRulings(
    optika.transforms.Transformable,
):
    """
    Interface for the interaction of a ruled surface with incident light
    """

    @property
    @abc.abstractmethod
    def diffraction_order(self) -> na.ScalarLike:
        """
        the diffraction order to simulate
        """

    @abc.abstractmethod
    def ruling_normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        vector normal to the plane of the rulings

        Parameters
        ----------
        position
            location to evaluate the normal vector
        """

    @abc.abstractmethod
    def rays_apparent(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        index_refraction: float,
    ) -> optika.rays.RayVectorArray:
        """
        the apparent input rays from the given actual input rays

        Parameters
        ----------
        rays
            actual input rays
        index_refraction
            index of refraction of the output rays
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractConstantDensityRulings(
    AbstractRulings,
):
    @property
    @abc.abstractmethod
    def ruling_density(self) -> na.ScalarLike:
        """
        the frequency of the ruling pattern
        """

    @property
    def ruling_spacing(self) -> na.ScalarLike:
        """
        the distance between successive rulings
        """
        return 1 / self.ruling_density

    def ruling_normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        unit vector normal to the planes of the rulings

        Parameters
        ----------
        position
            the location to evalulate the ruling normal
        Returns
        -------

        """
        return na.Cartesian3dVectorArray(1, 0, 0)
        # ruling_density = self.ruling_density
        # return na.Cartesian3dVectorArray(
        #     x=ruling_density,
        #     y=0 * ruling_density.unit,
        #     z=0 * ruling_density.unit,
        # )

    def rays_apparent(
        self,
        rays: optika.rays.RayVectorArray,
        index_refraction: float,
    ) -> optika.rays.RayVectorArray:
        """
        apparent direction of incoming rays if there was no diffraction

        Parameters
        ----------
        rays
            incoming rays that will be diffracted by the rulings
        index_refraction
            the index of refraction after the rulings
        """
        a = rays.index_refraction * rays.direction
        diffraction_order = self.diffraction_order
        ruling_density = self.ruling_density
        ruling_normal = ruling_density * self.ruling_normal(rays.position)
        a = a + index_refraction * diffraction_order * rays.wavelength * ruling_normal
        length_a = a.length
        return optika.rays.RayVectorArray(
            wavelength=rays.wavelength,
            position=rays.position,
            direction=a / length_a,
            intensity=rays.intensity,
            index_refraction=length_a,
        )


@dataclasses.dataclass(eq=False, repr=False)
class ConstantDensityRulings(
    AbstractConstantDensityRulings,
):
    ruling_density: na.ScalarLike = 0 / u.mm
    diffraction_order: na.ScalarLike = 1
    transform: None | optika.transforms.AbstractTransform = None
