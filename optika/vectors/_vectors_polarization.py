from __future__ import annotations
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractPolarizationVectorArray",
    "PolarizationVectorArray",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolarizationVectorArray(
    na.AbstractCartesianVectorArray,
):
    """
    An interface describing a vector in the :math:`s` and :math:`p` coordinate
    system.
    """

    @property
    @abc.abstractmethod
    def s(self) -> float | na.AbstractScalar:
        """
        The component of the electric field perpendicular to the plane of incidence.
        """

    @property
    @abc.abstractmethod
    def p(self) -> float | na.AbstractScalar:
        """
        The component of the electric field parallel to the plane of incidence.
        """

    @property
    def average(self) -> float | na.AbstractScalar:
        return (self.s + self.p) / 2

    @property
    def type_abstract(self) -> type[AbstractPolarizationVectorArray]:
        return AbstractPolarizationVectorArray

    @property
    def type_explicit(self) -> type[PolarizationVectorArray]:
        return PolarizationVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError

    @property
    def explicit(self) -> PolarizationVectorArray:
        return super().explicit


@dataclasses.dataclass(eq=False, repr=False)
class PolarizationVectorArray(
    AbstractPolarizationVectorArray,
    na.AbstractExplicitCartesianVectorArray,
):
    """
    A vector described in terms of components parallel and perpendicular
    to the plane of incidence.
    """

    s: float | na.AbstractScalar = 0
    """The component of the electric field perpendicular to the plane of incidence."""

    p: float | na.AbstractScalar = 0
    """The component of the electric field parallel to the plane of incidence."""

    @classmethod
    def from_scalar(
        cls,
        scalar: na.ScalarLike,
        like: None | PolarizationVectorArray = None,
    ) -> PolarizationVectorArray:
        if like is None:
            type_like = cls
        else:
            type_like = type(like)
        return type_like(
            s=scalar,
            p=scalar,
        )
