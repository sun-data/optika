"""Model the radiometry of an optical system."""

from ._effective_area import (
    AbstractEffectiveAreaModel,
    InterpolatedEffectiveAreaModel,
)

__all__ = [
    "AbstractEffectiveAreaModel",
    "InterpolatedEffectiveAreaModel",
]
