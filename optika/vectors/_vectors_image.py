from typing import TypeVar
from typing_extensions import Self
import dataclasses
import named_arrays as na
from . import AbstractSceneVectorArray, SceneVectorArray

__all__ = [
    "AbstractImageVectorArray",
    "ImageVectorArray",
]


PositionT = TypeVar("PositionT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImageVectorArray(
    na.AbstractPositionalVectorArray,
    AbstractSceneVectorArray,
):
    """
    An interface describing a field position and wavelength, and a position
    in the image of that point of the scene, such as a position on the sensor.

    This is the image-side counterpart of
    :class:`optika.vectors.AbstractObjectVectorArray`, which describes where
    a ray starts, and the inputs of a function of the image of each point of
    the scene, such as the point-spread function of
    :meth:`optika.systems.SequentialSystem.psf`.
    """

    @property
    def type_abstract(self) -> type["AbstractImageVectorArray"]:
        return AbstractImageVectorArray

    @property
    def type_explicit(self) -> type["ImageVectorArray"]:
        return ImageVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError


@dataclasses.dataclass(eq=False, repr=False)
class ImageVectorArray(
    na.PositionalVectorArray[PositionT],
    SceneVectorArray,
    AbstractImageVectorArray,
):
    """
    A vector describing a field position and wavelength, and a position in
    the image of that point of the scene, such as a position on the sensor.
    """

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
                position=scalar,
            )
        else:
            return cls(
                wavelength=scalar,
                field=scalar,
                position=scalar,
            )
