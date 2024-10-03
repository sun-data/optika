"""
Models of the depletion region of a semiconducting imaging sensor.
"""

from ._depletion import (
    AbstractDepletionModel,
    AbstractJanesickDepletionModel,
)
from ._e2v_ccd64_thick import E2VCCD64ThickDepletionModel
from ._e2v_ccd64_thin import E2VCCD64ThinDepletionModel

__all__ = [
    "AbstractDepletionModel",
    "AbstractJanesickDepletionModel",
    "E2VCCD64ThickDepletionModel",
    "E2VCCD64ThinDepletionModel",
]
