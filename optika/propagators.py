from __future__ import annotations
from typing import Sequence
import abc
import dataclasses
import named_arrays as na
import optika

__all__ = [
    "AbstractPropagator",
    "AbstractRayPropagator",
]


def propagate_rays(
    propagators: AbstractRayPropagator | Sequence[AbstractRayPropagator],
    rays: optika.rays.RayVectorArray,
) -> optika.rays.RayVectorArray:
    """
    Iterate through a sequence of ray propagators, calling
    :meth:`optika.propagators.AbstractRayPropagator.propagate_rays` on the given
    set of input rays.

    Parameters
    ----------
    propagators
        a sequence of ray propagators to interact with ``rays``
    rays
        the input rays to propagate through the sequence
    """
    if isinstance(propagators, AbstractRayPropagator):
        propagators = [propagators]

    for propagator in propagators:
        rays = propagator.propagate_rays(rays)

    return rays


def accumulate_rays(
    propagators: AbstractRayPropagator | Sequence[AbstractRayPropagator],
    rays: optika.rays.RayVectorArray,
    axis: str,
) -> optika.rays.RayVectorArray:
    """
    Iterate through a sequence of ray propagators, calling
    :meth:`optika.propagators.AbstractRayPropagator.propagate_rays` on the given
    set of input rays, and store the resulting the rays at every propagator.

    Parameters
    ----------
    propagators
        a sequence of ray propagators to interact with ``rays``
    rays
        the input rays to propagate through the sequence
    axis
        the new axis representing the sequence of propagators
    """
    if isinstance(propagators, AbstractRayPropagator):
        propagators = [propagators]

    result = []
    for propagator in propagators:
        rays = propagator.propagate_rays(rays)
        result.append(rays)

    result = na.stack(result, axis=axis)

    return result


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


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLightPropagator(
    AbstractRayPropagator,
):
    pass
