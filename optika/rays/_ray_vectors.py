from __future__ import annotations
from typing import TypeVar, Generic
import abc
import dataclasses
import named_arrays as na

__all__ = [
    "AbstractRayVectorArray",
    "RayVectorArray",
    "AbstractImplicitRayVectorArray",
]

IntensityT = TypeVar("IntensityT", bound=na.ScalarLike)
AttenuationT = TypeVar("AttenuationT", bound=na.ScalarLike)
PositionT = TypeVar("PositionT", bound=na.Cartesian3dVectorArray)
DirectionT = TypeVar("DirectionT", bound=na.Cartesian3dVectorArray)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
IndexRefractionT = TypeVar("IndexRefractionT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRayVectorArray(
    na.AbstractSpectralPositionalVectorArray,
):
    @property
    @abc.abstractmethod
    def direction(self) -> na.AbstractCartesian3dVectorArray:
        """
        the propagation direction of the ray expressed in
        `direction cosines <https://en.wikipedia.org/wiki/Direction_cosine>`_
        """

    @property
    @abc.abstractmethod
    def intensity(self) -> na.AbstractScalar:
        """
        the radiometric contribution of the ray
        """

    @property
    @abc.abstractmethod
    def attenuation(self) -> na.AbstractScalar:
        """
        the current
        `attenuation coefficient <https://en.wikipedia.org/wiki/Absorption_coefficient>`_
        of the medium that the ray is propagating in
        """

    @property
    @abc.abstractmethod
    def index_refraction(self):
        """
        the current index of refraction of the medium that the ray is
        traveling in
        """

    @property
    def type_abstract(self) -> type[na.AbstractArray]:
        return AbstractRayVectorArray

    @property
    def type_explicit(self) -> type[na.AbstractExplicitArray]:
        return RayVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError

    @property
    def explicit(self) -> RayVectorArray:
        return super().explicit


@dataclasses.dataclass(eq=False, repr=False)
class RayVectorArray(
    AbstractRayVectorArray,
    na.SpectralPositionalVectorArray[WavelengthT, PositionT],
    Generic[IntensityT, PositionT, DirectionT, WavelengthT, IndexRefractionT],
):
    direction: DirectionT = 0
    intensity: IntensityT = 0
    attenuation: AttenuationT = 0
    index_refraction: IndexRefractionT = 1

    @classmethod
    def from_scalar(
        cls,
        scalar: na.AbstractScalar,
        like: None | na.AbstractExplicitVectorArray = None,
    ) -> RayVectorArray:
        return cls(
            wavelength=scalar,
            position=scalar,
            direction=scalar,
            intensity=scalar,
            attenuation=scalar,
            index_refraction=scalar,
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImplicitRayVectorArray(
    AbstractRayVectorArray,
    na.AbstractImplicitCartesianVectorArray,
):
    @property
    def direction(self) -> na.AbstractCartesian3dVectorArray:
        return self.explicit.direction

    @property
    def intensity(self) -> na.ScalarLike:
        return self.explicit.intensity

    @property
    def attenuation(self) -> na.ScalarLike:
        return self.explicit.attenuation

    @property
    def index_refraction(self) -> na.ScalarLike:
        return self.explicit.index_refraction
