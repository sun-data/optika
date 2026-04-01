"""
Optical systems consisting of multiple optical surfaces.
"""

from ._systems import AbstractSystem
from ._sequential import AbstractSequentialSystem, SequentialSystem

__all__ = [
    "AbstractSystem",
    "AbstractSequentialSystem",
    "SequentialSystem",
]
