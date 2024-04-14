"""
A collection of tools for simulating measuring and tolerancing of optical surfaces.
"""

from ._slope_error import AbstractSlopeErrorParameters, SlopeErrorParameters
from ._roughness import AbstractRoughnessParameters, RoughnessParameters

__all__ = [
    "AbstractSlopeErrorParameters",
    "SlopeErrorParameters",
    "AbstractRoughnessParameters",
    "RoughnessParameters",
]
