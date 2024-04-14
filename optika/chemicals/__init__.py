"""
A subpackage for retrieving optical constants of various chemicals.
"""

from ._chemicals import AbstractChemical, Chemical

__all__ = [
    "AbstractChemical",
    "Chemical",
]
