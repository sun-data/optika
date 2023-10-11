from __future__ import annotations
import dataclasses
import named_arrays as na
import optika
from . import RayVectorArray

__all__ = [
    "AbstractRayFunctionArray",
    "RayFunctionArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRayFunctionArray(
    na.AbstractFunctionArray,
):
    @property
    def type_explicit(self) -> type[RayFunctionArray]:
        return RayFunctionArray


@dataclasses.dataclass(eq=False, repr=False)
class RayFunctionArray(
    AbstractRayFunctionArray,
    na.FunctionArray[
        optika.vectors.ObjectVectorArray,
        RayVectorArray,
    ],
):
    pass
