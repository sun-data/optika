from __future__ import annotations
from typing import Generic, TypeVar, Iterator, Iterable, Any, overload
from typing_extensions import Self
import copy
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "Printable",
    "DataclassList",
]

ItemT = TypeVar("ItemT")


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

        result = f"{self.__class__.__qualname__}(\n"

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
            result += field_str
        result += f"{pre})"
        return result

    def __repr__(self):
        return self.to_string()


@dataclasses.dataclass
class DataclassList(
    Generic[ItemT],
):
    data: list[ItemT] = dataclasses.field(default_factory=list)

    def __contains__(self, item: ItemT) -> bool:
        return self.data.__contains__(item)

    def __iter__(self) -> Iterator[ItemT]:
        return self.data.__iter__()

    def __reversed__(self) -> Iterator[ItemT]:
        return self.data.__reversed__()

    @overload
    def __getitem__(self, item: int) -> ItemT:
        ...

    @overload
    def __getitem__(self: Self, item: slice) -> Self:
        ...

    def __getitem__(self: Self, item: int | slice) -> ItemT | Self:
        if isinstance(item, slice):
            other = copy.copy(self)
            other.data = self.data.__getitem__(item)
            return other
        else:
            return self.data.__getitem__(item)

    @overload
    def __setitem__(self, key: int, value: ItemT):
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[ItemT]):
        ...

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __delitem__(self, key: int | slice):
        self.data.__delitem__(key)

    def __len__(self) -> int:
        return self.data.__len__()

    def __add__(self, other: DataclassList):
        new_self = copy.copy(self)
        new_self.data = self.data.__add__(other.data)
        return new_self

    def index(self, value: ItemT) -> int:
        return self.data.index(value)

    def count(self, value: ItemT) -> int:
        return self.data.count(value)

    def append(self, item: ItemT) -> None:
        self.data.append(item)

    def reverse(self):
        self.data.reverse()
