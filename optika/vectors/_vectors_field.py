from __future__ import annotations
from typing import TypeVar, Generic
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractFieldVectorArray",
    "FieldVectorArray",
]

FieldT = TypeVar("FieldT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFieldVectorArray(
    na.AbstractCartesianVectorArray,
):
    @property
    @abc.abstractmethod
    def field(self) -> na.AbstractCartesian2dVectorArray:
        """
        the position of a point in the field of view
        """

    @property
    def type_abstract(self) -> type[AbstractFieldVectorArray]:
        return AbstractFieldVectorArray

    @property
    def type_explicit(self) -> type[FieldVectorArray]:
        return FieldVectorArray

    @property
    def type_matrix(self) -> type[na.PositionalMatrixArray]:
        raise NotImplementedError

    @property
    def explicit(self) -> FieldVectorArray:
        return super().explicit


@dataclasses.dataclass(eq=False, repr=False)
class FieldVectorArray(
    AbstractFieldVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[FieldT],
):
    field: FieldT = 0

    @classmethod
    def from_scalar(
        cls,
        scalar: na.ScalarLike,
        like: None | FieldVectorArray = None,
    ) -> FieldVectorArray:
        if like is None:
            type_like = cls
        else:
            type_like = type(like)
        return type_like(
            field=scalar,
        )
