"""Simulate light propagating through an optical system."""

from __future__ import annotations
from typing import Sequence
import abc
import dataclasses
import named_arrays as na
import optika

__all__ = [
    "propagate_rays",
    "accumulate_rays",
    "AbstractPropagator",
    "AbstractRayPropagator",
    "AbstractLightPropagator",
]


def propagate_rays(
    propagators: AbstractRayPropagator | Sequence[AbstractRayPropagator],
    rays: optika.rays.RayVectorArray,
    efficiency: bool = True,
) -> optika.rays.RayVectorArray:
    """
    Iterate through a sequence of ray propagators, calling
    :meth:`~optika.propagators.AbstractRayPropagator.propagate_rays` on the given
    set of input rays.

    Parameters
    ----------
    propagators
        A sequence of ray propagators to interact with `rays`.
    rays
        The input rays to propagate through the sequence.
    efficiency
        A boolean flag indicating whether to accumulate the efficiency of
        each propagator into
        :attr:`~optika.rays.AbstractRayVectorArray.intensity`.
        If :obj:`False`, the intensity of the result is meaningless, but the
        geometry is unchanged and much cheaper to compute.
    """
    if isinstance(propagators, AbstractRayPropagator):
        propagators = [propagators]

    for propagator in propagators:
        rays = propagator.propagate_rays(rays, efficiency=efficiency)

    return rays


def accumulate_rays(
    propagators: AbstractRayPropagator | Sequence[AbstractRayPropagator],
    rays: optika.rays.RayVectorArray,
    axis: str,
    efficiency: bool = True,
) -> optika.rays.RayVectorArray:
    """
    Iterate through a sequence of ray propagators, calling
    :meth:`~optika.propagators.AbstractRayPropagator.propagate_rays` on the given
    set of input rays, and store the resulting the rays at every propagator.

    Parameters
    ----------
    propagators
        A sequence of ray propagators to interact with `rays`.
    rays
        The input rays to propagate through the sequence.
    axis
        The new logical axis representing the sequence of propagators.
    efficiency
        A boolean flag indicating whether to accumulate the efficiency of
        each propagator into
        :attr:`~optika.rays.AbstractRayVectorArray.intensity`.
        If :obj:`False`, the intensity of the result is meaningless, but the
        geometry is unchanged and much cheaper to compute.
    """
    if isinstance(propagators, AbstractRayPropagator):
        propagators = [propagators]

    result = []
    for propagator in propagators:
        rays = propagator.propagate_rays(rays, efficiency=efficiency)
        result.append(rays)

    result = na.stack(result, axis=axis)

    return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPropagator(
    abc.ABC,
):
    """An interface for an object which can propagate information."""


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRayPropagator(
    AbstractPropagator,
):
    """An interface for an object that can interact with light rays."""

    @abc.abstractmethod
    def propagate_rays(
        self,
        rays: optika.rays.AbstractRayVectorArray,
        efficiency: bool = True,
    ) -> optika.rays.AbstractRayVectorArray:
        """
        For the given input rays, calculate new rays based off of their
        interation with this object.

        Parameters
        ----------
        rays
            A set of input rays that will interact with this object.
        efficiency
            A boolean flag indicating whether to accumulate the efficiency of
            this object into
            :attr:`~optika.rays.AbstractRayVectorArray.intensity`.
            If :obj:`False`, the intensity of the result is meaningless, but
            the geometry is unchanged and much cheaper to compute.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLightPropagator(
    AbstractRayPropagator,
):
    """An interface for an object which can interact with light."""
