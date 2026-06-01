"""Model the radiometry of an optical system."""

from ._effective_area import (
    AbstractEffectiveAreaModel,
    InterpolatedEffectiveAreaModel,
)
from ._vignetting import (
    AbstractVignettingModel,
    AbstractInterpolatedVignettingModel,
    PolynomialVignettingModel,
)

__all__ = [
    "AbstractEffectiveAreaModel",
    "InterpolatedEffectiveAreaModel",
    "AbstractVignettingModel",
    "AbstractInterpolatedVignettingModel",
    "PolynomialVignettingModel",
]
