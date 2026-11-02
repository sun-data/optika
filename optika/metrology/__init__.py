"""
Simulate the measurement and tolerancing of optical surfaces.
"""

from ._slope_error import SlopeErrorParameters, slope_error
from ._roughness import RoughnessParameters

__all__ = [
    "SlopeErrorParameters",
    "slope_error",
    "RoughnessParameters",
]
