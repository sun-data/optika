"""Helpers shared by the parts of :mod:`optika` which draw themselves."""

from typing import Any

import matplotlib.axes
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import numpy as np
import named_arrays as na

__all__ = [
    "is_3d",
    "kwargs_filled",
]


def is_3d(
    ax: None | matplotlib.axes.Axes | na.ScalarArray,
) -> bool:
    """
    Whether every axes given is a 3D one.

    Parameters
    ----------
    ax
        The axes to check. If :obj:`None`, the current axes is used.
    """
    if ax is None:
        ax = matplotlib.pyplot.gca()
    ax = np.atleast_1d(na.as_named_array(ax).ndarray)
    return all(isinstance(a, mpl_toolkits.mplot3d.Axes3D) for a in ax.flat)


def kwargs_filled(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Turn keywords meant for a line into keywords for a filled polygon.

    The colour asked for becomes the edge of the polygon and the face is left
    blank, so that a filled surface reads as a drawing of an optic rather than
    a silhouette of one, while still hiding whatever lies behind it.

    Parameters
    ----------
    kwargs
        The keyword arguments given for drawing a line.
    """
    result = kwargs | {
        "facecolors": kwargs.get("facecolor", "white"),
        "edgecolors": kwargs.get("color", "black"),
    }
    result.pop("color", None)
    result.pop("facecolor", None)
    return result
