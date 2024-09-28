"""
Models of the depletion region of a semiconducting imaging sensor.
"""

from ._depletion import (
    AbstractDepletionModel,
    AbstractJanesickDepletionModel,
)
from ._e2v_ccd64_thick import E2VCCD64ThickDepletionModel

__all__ = [
    "AbstractDepletionModel",
    "AbstractJanesickDepletionModel",
    "E2VCCD64ThickDepletionModel",
]
