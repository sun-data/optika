"""
Model the depletion region of a semiconducting imaging sensor.
"""

from ._depletion import (
    AbstractDepletionModel,
    JanesickDepletionModel,
)
from ._e2v_ccd64_thick import e2v_ccd64_thick
from ._e2v_ccd64_thin import e2v_ccd64_thin

__all__ = [
    "AbstractDepletionModel",
    "JanesickDepletionModel",
    "e2v_ccd64_thick",
    "e2v_ccd64_thin",
]
