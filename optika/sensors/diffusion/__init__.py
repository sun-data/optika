"""
Models of the lateral diffusion of charge in a back-illuminated sensor.

The charge liberated by a photon spreads out as it travels from where the
photon was absorbed to the gates, so it is shared among neighboring pixels.
How far it spreads depends on the depth at which it was created, so each
model gives the charge cloud at every depth, and the quantities derived from
it: its width, its profile along one axis, the probability that two electrons
land in the same pixel, and, averaged over the depths at which photons of a
given absorption coefficient are absorbed, its width, the pixel-integrated
kernel, and the mean charge capture.

The thickness of the light-sensitive substrate and the width of a pixel are
properties of the sensor rather than of the model, so the methods take them as
arguments.
"""

from ._measurements import (
    MeanChargeCaptureFunctionArray,
)
from ._models import (
    AbstractDiffusionModel,
    JanesickDiffusionModel,
)
from ._stern2004 import (
    mcc_stern2004,
    e2v_ccd64_thick,
    e2v_ccd64_thin,
)

__all__ = [
    "MeanChargeCaptureFunctionArray",
    "AbstractDiffusionModel",
    "JanesickDiffusionModel",
    "mcc_stern2004",
    "e2v_ccd64_thick",
    "e2v_ccd64_thin",
]
