"""
Optical systems consisting of multiple optical surfaces.
"""

from ._systems import AbstractSystem
from ._sequential import AbstractSequentialSystem, SequentialSystem
from ._linear import AbstractLinearSystem, LinearSystem

__all__ = [
    "AbstractSystem",
    "AbstractSequentialSystem",
    "SequentialSystem",
    "AbstractLinearSystem",
    "LinearSystem",
]
