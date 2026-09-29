"""
Models of the lateral diffusion of charge in a back-illuminated sensor.

The charge liberated by a photon spreads out as it travels from where the
photon was absorbed to the gates, so it is shared among neighboring pixels.
How far it spreads depends on the depth at which it was created, so each
model gives the width of the charge cloud at every depth, and the quantities
derived from it: the average width for photons of a given absorption
coefficient, the probability that two electrons land in the same pixel,
the mean charge capture, and the pixel-integrated kernel.

The thickness of the light-sensitive substrate is a property of the sensor
rather than of the model, so every method takes it as an argument.
"""

from ._models import (
    AbstractDiffusionModel,
    JanesickDiffusionModel,
)

__all__ = [
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
]
