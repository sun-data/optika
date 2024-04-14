"""
Periodic rulings which can be added to an optical surface to model a
diffraction grating.
"""

from ._spacing import (
    AbstractRulingSpacing,
    ConstantRulingSpacing,
    Polynomial1dRulingSpacing,
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
    "AbstractRulings",
    "Rulings",
    "MeasuredRulings",
    "SinusoidalRulings",
    "SquareRulings",
    "SawtoothRulings",
    "TriangularRulings",
    "RectangularRulings",
]
