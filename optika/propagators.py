import abc
import dataclasses
import optika

__all__ = [
    "AbstractPropagator",
    "AbstractRayPropagator",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPropagator(
    abc.ABC,
):
    pass


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRayPropagator(
    AbstractPropagator,
):
    @abc.abstractmethod
    def propagate_rays(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:
        """
        for the given input rays, calculate new rays based off of their
        interation with this object

        Parameters
        ----------
        rays
            a set of input rays that will interact with this object
        """
