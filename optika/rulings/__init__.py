"""
Periodic rulings which can be added to an optical surface to model a
diffraction grating.
"""

from ._spacing import (
    AbstractRulingSpacing,
    ConstantRulingSpacing,
    Polynomial1dRulingSpacing,
    HolographicRulingSpacing,
)
from ._rulings import (
    AbstractRulings,
    Rulings,
    MeasuredRulings,
    SinusoidalRulings,
    SquareRulings,
    SawtoothRulings,
    TriangularRulings,
    RectangularRulings,
)

__all__ = [
    "AbstractRulingSpacing",
    "ConstantRulingSpacing",
    "Polynomial1dRulingSpacing",
    "HolographicRulingSpacing",
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
    "SinusoidalRulings",
    "SquareRulings",
    "SawtoothRulings",
    "TriangularRulings",
    "RectangularRulings",
]
