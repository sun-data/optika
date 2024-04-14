from __future__ import annotations
from typing import TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na
from . import AbstractFieldVectorArray, FieldVectorArray

__all__ = [
    "AbstractSceneVectorArray",
    "SceneVectorArray",
]


WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
FieldT = TypeVar("FieldT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSceneVectorArray(
    AbstractFieldVectorArray,
    na.AbstractSpectralVectorArray,
):
    @property
    def type_abstract(self) -> type[AbstractSceneVectorArray]:
        return AbstractSceneVectorArray

    @property
    def type_explicit(self) -> type[SceneVectorArray]:
        return SceneVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class SceneVectorArray(
    AbstractSceneVectorArray,
    FieldVectorArray[FieldT],
    na.SpectralVectorArray[WavelengthT],
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
                # field=na.Cartesian2dVectorArray.from_scalar(scalar, like=like.field)
            )
        else:
            return cls(
                wavelength=scalar,
                field=scalar,
                # field=na.Cartesian2dVectorArray.from_scalar(scalar),
            )
