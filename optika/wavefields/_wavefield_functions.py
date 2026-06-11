from __future__ import annotations
import dataclasses
import named_arrays as na
import optika
from ._wavefield_vectors import WavefieldVectorArray

__all__ = [
    "AbstractWavefieldFunctionArray",
    "WavefieldFunctionArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractWavefieldFunctionArray(
    na.AbstractFunctionArray,
):
    """
    An interface describing a function which maps input wavelength and field
    coordinates to an output wavefield.
    """

    @property
    def type_explicit(self) -> type[WavefieldFunctionArray]:
        return WavefieldFunctionArray


@dataclasses.dataclass(eq=False, repr=False)
class WavefieldFunctionArray(
    AbstractWavefieldFunctionArray,
    na.FunctionArray[
        optika.vectors.ObjectVectorArray,
        WavefieldVectorArray,
    ],
):
    """
    A discrete function which maps input wavelength and field coordinates to
    an output wavefield.
    """
