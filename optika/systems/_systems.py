from __future__ import annotations
from typing import Sequence, Callable, Any, ClassVar
import abc
import dataclasses
import named_arrays as na
import optika

__all__ = [
    "AbstractSystem",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSystem(
    optika.mixins.DxfWritable,
    optika.mixins.Plottable,
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    """
    An interface describing an optical system.

    Could potentially be sequential or non-sequential.
    """

    @abc.abstractmethod
    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        **kwargs: Any,
    ) -> na.SpectralPositionalVectorArray:
        """
        Forward model of the optical system.
        Maps the given spectral radiance of a scene to detector counts.

        Parameters
        ----------
        scene
            The spectral radiance of the scene as a function of wavelength
            and field position.
        kwargs
            Additional keyword arguments used by subclass implementations
            of this method.
        """
