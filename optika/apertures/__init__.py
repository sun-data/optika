"""
Apertures that can be used by optical surfaces to block a portion of the beam.
"""

from ._apertures import (
    AbstractAperture,
    CircularAperture,
    CircularSectorAperture,
    AnnularAperture,
    EllipticalAperture,
    AbstractPolygonalAperture,
    PolygonalAperture,
    RectangularAperture,
    AbstractRegularPolygonalAperture,
    RegularPolygonalAperture,
    AbstractOctagonalAperture,
    OctagonalAperture,
    AbstractIsoscelesTrapezoidalAperture,
    IsoscelesTrapezoidalAperture,
)

__all__ = [
    "AbstractAperture",
    "CircularAperture",
    "CircularSectorAperture",
    "AnnularAperture",
    "EllipticalAperture",
    "AbstractPolygonalAperture",
    "PolygonalAperture",
    "RectangularAperture",
    "AbstractRegularPolygonalAperture",
    "RegularPolygonalAperture",
    "AbstractOctagonalAperture",
    "OctagonalAperture",
    "AbstractIsoscelesTrapezoidalAperture",
    "IsoscelesTrapezoidalAperture",
]
