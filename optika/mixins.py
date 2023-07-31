from __future__ import annotations
from typing import Self, Generic, TypeVar, Iterator, Iterable, overload
import copy
import dataclasses

__all__ = [
    "DataclassList",
]

ItemT = TypeVar("ItemT")


@dataclasses.dataclass
class DataclassList(
    Generic[ItemT],
):
    data: list[ItemT] = dataclasses.MISSING  # type: ignore

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
