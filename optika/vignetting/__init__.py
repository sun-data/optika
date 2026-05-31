"""Model the vignetting of a scene observed by an optical system."""

from ._vignetting import (
    AbstractVignettingModel,
    AbstractInterpolatedVignettingModel,
    PolynomialVignettingModel,
)

__all__ = [
    "AbstractVignettingModel",
    "AbstractInterpolatedVignettingModel",
    "PolynomialVignettingModel",
]
