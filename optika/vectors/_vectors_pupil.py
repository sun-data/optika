from __future__ import annotations
from typing import TypeVar, Generic
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractPupilVectorArray",
    "PupilVectorArray",
]

PupilT = TypeVar("PupilT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPupilVectorArray(
    na.AbstractCartesianVectorArray,
):
    """An interface describing a pupil position."""

    @property
    @abc.abstractmethod
    def pupil(self) -> na.AbstractCartesian2dVectorArray:
        """
        A point on the pupil.
        """

    @property
    def type_abstract(self) -> type[AbstractPupilVectorArray]:
        return AbstractPupilVectorArray

    @property
    def type_explicit(self) -> type[PupilVectorArray]:
        return PupilVectorArray

    @property
    def type_matrix(self) -> type[na.PositionalMatrixArray]:
        raise NotImplementedError

    @property
    def explicit(self) -> PupilVectorArray:
        return super().explicit


@dataclasses.dataclass(eq=False, repr=False)
class PupilVectorArray(
    AbstractPupilVectorArray,
    na.AbstractExplicitCartesianVectorArray,
    Generic[PupilT],
):
    """A vector describing a pupil position."""

    pupil: PupilT = 0
    """A point on the pupil."""

    @classmethod
    def from_scalar(
        cls,
        scalar: na.ScalarLike,
        like: None | PupilVectorArray = None,
    ) -> PupilVectorArray:
        if like is None:
            type_like = cls
        else:
            type_like = type(like)
        return type_like(
            pupil=scalar,
        )
