"""
Models of light sensors that can be used in optical systems.
"""

from typing import TypeVar
import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractImagingSensor",
    "IdealImagingSensor",
    "AbstractCCD",
]


MaterialT = TypeVar("MaterialT", bound=optika.materials.AbstractMaterial)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractImagingSensor(
    optika.surfaces.AbstractSurface[
        None,
        MaterialT,
        optika.apertures.RectangularAperture,
        optika.apertures.RectangularAperture,
        None,
    ],
):
    """
    An interface describing an imaging sensor that can be used as the last
    surface in an optical system.
    """

    @property
    def sag(self) -> optika.sags.AbstractSag:
        return optika.sags.NoSag()

    @property
    def rulings(self) -> None:
        return None

    @property
    @abc.abstractmethod
    def width_pixel(self) -> u.Quantity | na.AbstractCartesian2dVectorArray:
        """
        The physical size of each pixel on the sensor.
        """

    @property
    @abc.abstractmethod
    def num_pixel(self) -> na.Cartesian2dVectorArray[int, int]:
        """
        The number of pixels along each axis of the sensor.
        """

    @property
    def aperture(self):
        """
        The light-sensitive aperture of the sensor.
        """
        return optika.apertures.RectangularAperture(
            half_width=self.width_pixel * self.num_pixel / 2,
        )


@dataclasses.dataclass(eq=False, repr=False)
class IdealImagingSensor(
    AbstractImagingSensor,
):
    """
    An idealized imaging sensor with perfect efficiency and no noise sources.
    """

    name: None | str = None
    """The human-readable name of this sensor."""

    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.um
    """The physical size of each pixel on the sensor."""

    num_pixel: na.Cartesian2dVectorArray[int, int] = None
    """The number of pixels along each axis of the sensor."""

    aperture_mechanical: optika.apertures.RectangularAperture = None
    """The shape of the physical substrate supporting the sensor."""

    is_field_stop: bool = False
    """A flag controlling whether this sensor is the field stop for the system."""

    is_pupil_stop: bool = False
    """A flag controlling whether this sensor is the pupil stop for the system."""

    transformation: None | na.transformations.AbstractTransformation = None
    """The position and orientation of the sensor in the global coordinate system."""

    kwargs_plot: None | dict = None
    """Extra keyword arguments to pass to :meth:`plot`"""

    @property
    def material(self) -> optika.materials.AbstractMaterial:
        return optika.materials.Vacuum()


class AbstractCCD(
    AbstractImagingSensor[MaterialT],
):
    pass
