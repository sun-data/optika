"""
Simulate the measurement and tolerancing of optical surfaces.
"""

from ._slope_error import SlopeErrorParameters
from ._roughness import RoughnessParameters

__all__ = [
    "SlopeErrorParameters",
    "RoughnessParameters",
]
