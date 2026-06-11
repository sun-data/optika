"""
Physical-optics propagation of scalar wavefields through an optical system.
"""

from ._wavefield_vectors import AbstractWavefieldVectorArray, WavefieldVectorArray
from ._wavefield_functions import (
    AbstractWavefieldFunctionArray,
    WavefieldFunctionArray,
)
from ._propagation import rayleigh_sommerfeld
from ._metrics import encircled_energy_radius, ensquared_energy, fwhm

__all__ = [
    "AbstractWavefieldVectorArray",
    "WavefieldVectorArray",
    "AbstractWavefieldFunctionArray",
    "WavefieldFunctionArray",
    "rayleigh_sommerfeld",
    "encircled_energy_radius",
    "ensquared_energy",
    "fwhm",
]
