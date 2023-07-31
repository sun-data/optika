from __future__ import annotations
from typing import Generic, TypeVar, Iterator
import copy
import dataclasses

ItemT = TypeVar("ItemT")


@dataclasses.dataclass
class DataclassList(
    Generic[ItemT],
):
    data: list = dataclasses.MISSING

    def __contains__(self, item: ItemT) -> bool:
        return self.data.__contains__(item)

    def __iter__(self) -> Iterator[ItemT]:
        return self.data.__iter__()

    def __reversed__(self) -> Iterator[ItemT]:
        return self.data.__reversed__()

    def __getitem__(self, item: int | slice) -> ItemT:
        if isinstance(item, slice):
            other = copy.copy(self)
            other.data = self.data.__getitem__(item)
            return other
        else:
            return self.data.__getitem__(item)

    def __setitem__(self, key: int | slice, value: ItemT):
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
