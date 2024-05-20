from __future__ import annotations
from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractRayVectorArray",
    "RayVectorArray",
]

IntensityT = TypeVar("IntensityT", bound=na.ScalarLike)
AttenuationT = TypeVar("AttenuationT", bound=na.ScalarLike)
PositionT = TypeVar("PositionT", bound=na.Cartesian3dVectorArray)
DirectionT = TypeVar("DirectionT", bound=na.Cartesian3dVectorArray)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
IndexRefractionT = TypeVar("IndexRefractionT", bound=na.ScalarLike)
UnvignettedT = TypeVar("UnvignettedT", bound=na.ScalarLike)


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
    @abc.abstractmethod
    def unvignetted(self) -> na.AbstractScalar:
        """
        A boolean mask where :obj:`True` indicates the ray makes it all the
        way through the optical system without being blocked by an aperture,
        and :obj:`False` indicates the ray has been blocked by an aperture.
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

    def __array_matmul__(
        self,
        x1: na.ArrayLike,
        x2: na.ArrayLike,
        out: tuple[None | na.AbstractExplicitArray] = (None,),
        **kwargs,
    ) -> na.AbstractExplicitArray | RayVectorArray:
        result = super().__array_matmul__(
            x1=x1,
            x2=x2,
            out=out,
            **kwargs,
        )
        if result is not NotImplemented:
            return result

        if isinstance(x1, AbstractRayVectorArray):
            if isinstance(x2, na.AbstractCartesian3dMatrixArray):
                return dataclasses.replace(
                    x1,
                    position=x1.position @ x2,
                    direction=x1.direction @ x2,
                )
            else:
                return NotImplemented
        elif isinstance(x2, AbstractRayVectorArray):
            if isinstance(x1, na.AbstractCartesian3dMatrixArray):
                return dataclasses.replace(
                    x2,
                    position=x1 @ x2.position,
                    direction=x1 @ x2.direction,
                )
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array_add__(
        self,
        x1: na.ArrayLike,
        x2: na.ArrayLike,
        out: tuple[None | na.AbstractExplicitArray] = (None,),
        **kwargs,
    ) -> na.AbstractExplicitArray:
        if isinstance(x1, AbstractRayVectorArray):
            if isinstance(x2, na.AbstractCartesian3dVectorArray):
                return dataclasses.replace(
                    x1,
                    position=x1.position + x2,
                )
            else:
                return NotImplemented
        elif isinstance(x2, AbstractRayVectorArray):
            if isinstance(x1, na.AbstractCartesian3dVectorArray):
                return dataclasses.replace(
                    x2,
                    position=x1 + x2.position,
                )
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array_ufunc__(
        self,
        function: np.ufunc,
        method: str,
        *inputs,
        **kwargs,
    ) -> None | RayVectorArray | tuple[RayVectorArray, ...]:
        result = super().__array_ufunc__(
            function,
            method,
            *inputs,
            **kwargs,
        )
        if result is not NotImplemented:
            return result

        if function is np.add:
            if method == "__call__":
                return self.__array_add__(*inputs, **kwargs)


@dataclasses.dataclass(eq=False, repr=False)
class RayVectorArray(
    AbstractRayVectorArray,
    na.SpectralPositionalVectorArray[WavelengthT, PositionT],
    Generic[
        IntensityT,
        PositionT,
        DirectionT,
        WavelengthT,
        IndexRefractionT,
        UnvignettedT,
    ],
):
    direction: DirectionT = 0
    intensity: IntensityT = 1
    attenuation: AttenuationT = 0 / u.mm
    index_refraction: IndexRefractionT = 1

    unvignetted: na.AbstractScalar = True
    """
    A boolean mask where :obj:`True` indicates the ray makes it all the
    way through the optical system without being blocked by an aperture,
    and :obj:`False` indicates the ray has been blocked by an aperture.
    """

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
            unvignetted=scalar,
        )
