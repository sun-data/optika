from __future__ import annotations
from typing import Iterator
from typing_extensions import Self
import abc
import dataclasses
import copy
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika.mixins

__all__ = [
    "AbstractTransform",
    "Translation",
    "AbstractRotation",
    "RotationX",
    "RotationY",
    "RotationZ",
    "TransformList",
    "Transformable",
]


class AbstractTransform(
    abc.ABC,
):
    @property
    def matrix(self) -> na.AbstractCartesian3dMatrixArray:
        return na.Cartesian3dIdentityMatrixArray()

    @property
    def vector(self) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray() * u.mm

    def __call__(
        self,
        value: na.AbstractCartesian3dVectorArray,
        rotate: bool = True,
        translate: bool = True,
    ) -> na.AbstractCartesian3dVectorArray:
        if rotate:
            value = self.matrix @ value
        if translate:
            value = value + self.vector
        return value

    @abc.abstractmethod
    def __invert__(self: Self) -> Self:
        pass

    @property
    def inverse(self: Self) -> Self:
        return self.__invert__()


@dataclasses.dataclass
class Translation(AbstractTransform):
    displacement: na.Cartesian3dVectorArray = dataclasses.MISSING

    @property
    def vector(self) -> na.Cartesian3dVectorArray:
        return self.displacement

    def __invert__(self: Self) -> Self:
        return type(self)(displacement=-self.displacement)


@dataclasses.dataclass
class AbstractRotation(AbstractTransform):
    angle: na.ScalarLike

    def __invert__(self: Self) -> Self:
        return type(self)(angle=-self.angle)


@dataclasses.dataclass
class RotationX(AbstractRotation):
    @property
    def matrix(self) -> na.Cartesian3dXRotationMatrixArray:
        return na.Cartesian3dXRotationMatrixArray(self.angle)


@dataclasses.dataclass
class RotationY(AbstractRotation):
    @property
    def matrix(self) -> na.Cartesian3dYRotationMatrixArray:
        return na.Cartesian3dYRotationMatrixArray(self.angle)


@dataclasses.dataclass
class RotationZ(AbstractRotation):
    @property
    def matrix(self) -> na.Cartesian3dZRotationMatrixArray:
        return na.Cartesian3dZRotationMatrixArray(self.angle)


@dataclasses.dataclass
class TransformList(
    AbstractTransform,
    optika.mixins.DataclassList,
):

    intrinsic: bool = True

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    @property
    def transforms(self) -> Iterator[AbstractTransform]:
        if self.intrinsic:
            return reversed(list(self))
        else:
            return iter(self)

    @property
    def matrix(self) -> na.Cartesian3dMatrixArray:
        rotation = na.Cartesian3dIdentityMatrixArray()

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                rotation = rotation @ transform.matrix

        return rotation

    @property
    def vector(self) -> na.Cartesian3dVectorArray:
        rotation = na.Cartesian3dIdentityMatrixArray()
        translation = 0

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                rotation = rotation @ transform.matrix
                translation = rotation @ transform.vector + translation

        return translation

    def __invert__(self: Self) -> Self:
        other = copy.copy(self)
        other.data = []
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other


@dataclasses.dataclass
class Transformable:
    transform: TransformList = dataclasses.field(default_factory=TransformList)
