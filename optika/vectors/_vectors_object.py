from __future__ import annotations
from typing import TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na
from . import AbstractPupilVectorArray, PupilVectorArray
from . import AbstractSceneVectorArray, SceneVectorArray

__all__ = [
    "AbstractObjectVectorArray",
    "ObjectVectorArray",
]


WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
FieldT = TypeVar("FieldT", bound=na.AbstractCartesian2dVectorArray)
PupilT = TypeVar("PupilT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractObjectVectorArray(
    AbstractPupilVectorArray,
    AbstractSceneVectorArray,
):
    @property
    def type_abstract(self) -> type[AbstractObjectVectorArray]:
        return AbstractObjectVectorArray

    @property
    def type_explicit(self) -> type[ObjectVectorArray]:
        return ObjectVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class ObjectVectorArray(
    AbstractObjectVectorArray,
    PupilVectorArray,
    SceneVectorArray,
):
    @classmethod
    def from_scalar(
        cls,
        scalar: na.AbstractScalar,
        like: None | Self = None,
    ) -> Self:
        if like is not None:
            return type(like)(
                wavelength=scalar,
                field=scalar,
                pupil=scalar,
            )
        else:
            return cls(
                wavelength=scalar,
                field=scalar,
                pupil=scalar,
            )
