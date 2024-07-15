from typing import Any
import named_arrays as na
import optika

__all__ = [
    "shape",
]


_shaped_type = (
    na.AbstractArray,
    na.transformations.AbstractTransformation,
    optika.mixins.Shaped,
)


def shape(a: Any) -> dict[str, int]:
    """
    Return the array shape of the given object.

    If the given object is an instance of :class:`named_arrays.AbstractArray`,
    :class:`named_arrays.transformations.AbstractTransformation`, or
    :class:`optika.mixins.Shaped`, ``a.shape`` will be returned.
    Otherwise an empty dictionary will be returned.

    Parameters
    ----------
    a
        The object to find the shape of.
    """
    if isinstance(a, _shaped_type):
        return a.shape
    else:
        return dict()
