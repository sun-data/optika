from __future__ import annotations
from typing import Generic, TypeVar, Iterator, Iterable, Any, overload
from typing_extensions import Self
import abc
import copy
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "Printable",
    "Pitchable",
]


@dataclasses.dataclass(repr=False)
class Printable:
    @classmethod
    def _val_to_string(
        cls,
        val: Any,
        pre: str,
        tab: str,
        field_str: str,
    ) -> str:
        if isinstance(val, Printable):
            val_str = val.to_string(prefix=f"{pre}{tab}")
        elif isinstance(val, na.AbstractArray):
            val_str = val.to_string(prefix=f"{pre}{tab}")
        elif isinstance(val, np.ndarray):
            val_str = np.array2string(
                a=val,
                separator=", ",
                prefix=field_str,
            )
            if isinstance(val, u.Quantity):
                val_str = f"{val_str} {val.unit}"
        elif isinstance(val, list):
            val_str = f"[\n"
            for v in val:
                val_str += f"{pre}{tab}{tab}"
                val_str += cls._val_to_string(
                    val=v,
                    pre=f"{pre}{tab}",
                    tab=tab,
                    field_str=f"{pre}{tab}",
                )
                val_str += ",\n"
            val_str += f"{pre}{tab}]"
        else:
            val_str = repr(val)

        return val_str

    def to_string(
        self,
        prefix: None | str = None,
    ) -> str:
        """
        Public-facing version of the ``__repr__`` method that allows for
        defining a prefix string, which can be used to calculate how much
        whitespace to add to the beginning of each line of the result.

        Parameters
        ----------
        prefix
            an optional string, the length of which is used to calculate how
            much whitespace to add to the result.
        """

        fields = dataclasses.fields(self)

        delim_field = "\n"
        pre = " " * len(prefix) if prefix is not None else ""
        tab = " " * 4

        result_fields = ""
        for i, f in enumerate(fields):
            field_str = f"{pre}{tab}{f.name}="
            val = getattr(self, f.name)
            val_str = self._val_to_string(
                val=val,
                pre=pre,
                tab=tab,
                field_str=field_str,
            )
            field_str += val_str
            field_str += f",{delim_field}"
            result_fields += field_str

        if result_fields:
            result_fields = f"\n{result_fields}{pre}"

        result = f"{self.__class__.__qualname__}({result_fields})"

        return result

    def __repr__(self):
        return self.to_string()


@dataclasses.dataclass(eq=False, repr=False)
class Transformable(abc.ABC):
    @property
    @abc.abstractmethod
    def transformation(self) -> None | na.transformations.AbstractTransformation:
        """
        the coordinate transformation between the global coordinate system
        and this object's local coordinate system
        """
        return na.transformations.IdentityTransformation()


@dataclasses.dataclass(eq=False, repr=False)
class Translatable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def translation(self) -> u.Quantity | na.AbstractScalar | na.AbstractVectorArray:
        """translate the coordinate system"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Translation(self.translation)


@dataclasses.dataclass(eq=False, repr=False)
class Pitchable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def pitch(self) -> u.Quantity | na.ScalarLike:
        """pitch angle of this object"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dRotationX(
            angle=self.pitch
        )


@dataclasses.dataclass(eq=False, repr=False)
class Rollable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def roll(self) -> u.Quantity | na.ScalarLike:
        """roll angle of this object"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dRotationZ(
            angle=self.roll
        )
