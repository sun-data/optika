from __future__ import annotations
from typing import TypeVar, Generic
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "AbstractWavefieldVectorArray",
    "WavefieldVectorArray",
]

AmplitudeT = TypeVar("AmplitudeT", bound=na.ScalarLike)
PositionT = TypeVar("PositionT", bound=na.Cartesian3dVectorArray)
NormalT = TypeVar("NormalT", bound=na.Cartesian3dVectorArray)
AreaT = TypeVar("AreaT", bound=na.ScalarLike)
WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
IndexRefractionT = TypeVar("IndexRefractionT", bound=na.ScalarLike)
UnvignettedT = TypeVar("UnvignettedT", bound=na.ScalarLike)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractWavefieldVectorArray(
    na.AbstractSpectralPositionalVectorArray,
):
    """
    An interface describing a complex scalar wavefield sampled at a discrete
    set of points on a surface.
    """

    @property
    @abc.abstractmethod
    def amplitude(self) -> na.AbstractScalar:
        """
        The dimensionless, complex scalar amplitude of the wavefield at each
        sample point.
        """

    @property
    @abc.abstractmethod
    def normal(self) -> na.AbstractCartesian3dVectorArray:
        """The unit vector perpendicular to the surface at each sample point."""

    @property
    @abc.abstractmethod
    def area(self) -> na.AbstractScalar:
        """The area of the surface element associated with each sample point."""

    @property
    @abc.abstractmethod
    def index_refraction(self) -> na.AbstractScalar:
        """
        The index of refraction of the medium the wavefield is propagating
        into.

        Note that :attr:`wavelength` is always the vacuum wavelength,
        the medium enters the propagation only through this index.
        """

    @property
    @abc.abstractmethod
    def unvignetted(self) -> na.AbstractScalar:
        """
        A boolean mask where :obj:`False` indicates the sample point was
        blocked by an aperture (the amplitude is also zero there).
        """

    @property
    def type_abstract(self) -> type[na.AbstractArray]:
        return AbstractWavefieldVectorArray

    @property
    def type_explicit(self) -> type[na.AbstractExplicitArray]:
        return WavefieldVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError

    @property
    def explicit(self) -> WavefieldVectorArray:
        return super().explicit

    def __array_matmul__(
        self,
        x1: na.ArrayLike,
        x2: na.ArrayLike,
        out: tuple[None | na.AbstractExplicitArray] = (None,),
        **kwargs,
    ) -> na.AbstractExplicitArray | WavefieldVectorArray:
        result = super().__array_matmul__(
            x1=x1,
            x2=x2,
            out=out,
            **kwargs,
        )
        if result is not NotImplemented:
            return result

        if isinstance(x1, AbstractWavefieldVectorArray):
            if isinstance(x2, na.AbstractCartesian3dMatrixArray):
                return dataclasses.replace(
                    x1,
                    position=x1.position @ x2,
                    normal=x1.normal @ x2,
                )
            else:
                return NotImplemented
        elif isinstance(x2, AbstractWavefieldVectorArray):
            if isinstance(x1, na.AbstractCartesian3dMatrixArray):
                return dataclasses.replace(
                    x2,
                    position=x1 @ x2.position,
                    normal=x1 @ x2.normal,
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
        if isinstance(x1, AbstractWavefieldVectorArray):
            if isinstance(x2, na.AbstractCartesian3dVectorArray):
                return dataclasses.replace(
                    x1,
                    position=x1.position + x2,
                )
            else:
                return NotImplemented
        elif isinstance(x2, AbstractWavefieldVectorArray):
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
        function,
        method: str,
        *inputs,
        **kwargs,
    ) -> None | WavefieldVectorArray | tuple[WavefieldVectorArray, ...]:
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
class WavefieldVectorArray(
    AbstractWavefieldVectorArray,
    na.SpectralPositionalVectorArray[WavelengthT, PositionT],
    Generic[
        AmplitudeT,
        PositionT,
        NormalT,
        AreaT,
        WavelengthT,
        IndexRefractionT,
        UnvignettedT,
    ],
):
    """
    A complex scalar wavefield sampled at a discrete set of points on a
    surface.

    Note that the amplitude is dimensionless: the Rayleigh-Sommerfeld kernel
    factor :math:`\\cos \\theta \\, dA / (\\lambda r)` is dimensionless, so the
    amplitude stays dimensionless as the wavefield propagates through an
    optical system, and intensities are typically normalized at the end of
    the calculation (e.g. for a point-spread function).
    """

    amplitude: AmplitudeT = 1 + 0j
    """
    The dimensionless, complex scalar amplitude of the wavefield at each
    sample point.
    """

    normal: NormalT = 0
    """The unit vector perpendicular to the surface at each sample point."""

    area: AreaT = 0 * u.mm**2
    """The area of the surface element associated with each sample point."""

    index_refraction: IndexRefractionT = 1
    """The index of refraction of the medium the wavefield is propagating into."""

    unvignetted: UnvignettedT = True
    """
    A boolean mask where :obj:`False` indicates the sample point was blocked
    by an aperture.
    """

    @classmethod
    def from_scalar(
        cls,
        scalar: na.AbstractScalar,
        like: None | na.AbstractExplicitVectorArray = None,
    ) -> WavefieldVectorArray:
        return cls(
            wavelength=scalar,
            position=scalar,
            amplitude=scalar,
            normal=scalar,
            area=scalar,
            index_refraction=scalar,
            unvignetted=scalar,
        )
