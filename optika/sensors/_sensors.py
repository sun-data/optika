"""
Models of light sensors that can be used in optical systems.
"""

from typing import TypeVar
import abc
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractImagingSensor",
    "AbstractCCD",
]


MaterialT = TypeVar("MaterialT", bound=optika.materials.AbstractMaterial)


class AbstractImagingSensor(
    optika.surfaces.AbstractSurface[
        None,
        MaterialT,
        optika.apertures.RectangularAperture,
        optika.apertures.RectangularAperture,
        None,
    ],
):
    @property
    def sag(self) -> None:
        return None

    @property
    def rulings(self) -> None:
        return None


class AbstractCCD(
    AbstractImagingSensor[MaterialT],
):
    pass
