"""
Optical systems consisting of multiple optical surfaces.
"""

from ._systems import AbstractSystem
from ._linear import AbstractLinearSystem, LinearSystem
from ._sequential import StopSolveError, AbstractSequentialSystem, SequentialSystem

__all__ = [
    "AbstractSystem",
    "AbstractLinearSystem",
    "LinearSystem",
    "StopSolveError",
    "AbstractSequentialSystem",
    "SequentialSystem",
]
