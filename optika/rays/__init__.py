"""A representation of light rays propagating through an optical system."""

from ._ray_vectors import (
    AbstractRayVectorArray,
    RayVectorArray,
)
from ._ray_functions import (
    AbstractRayFunctionArray,
    RayFunctionArray,
)

__all__ = [
    "AbstractRayVectorArray",
    "RayVectorArray",
    "AbstractRayFunctionArray",
    "RayFunctionArray",
]
