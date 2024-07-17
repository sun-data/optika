"""
Models of light sensors that can be used in optical systems.
"""

from typing import TypeVar, Sequence
import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractImagingSensorMaterial

__all__ = [
    "AbstractImagingSensor",
    "IdealImagingSensor",
    "AbstractCCD",
]


MaterialT = TypeVar("MaterialT", bound=AbstractImagingSensorMaterial)


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
    @abc.abstractmethod
    def timedelta_exposure(self) -> u.Quantity | na.AbstractScalar:
        """
        The exposure time of the sensor.
        """

    @property
    def aperture(self):
        """
        The light-sensitive aperture of the sensor.
        """
        return optika.apertures.RectangularAperture(
            half_width=self.width_pixel * self.num_pixel / 2,
        )

    def readout(
        self,
        rays: optika.rays.RayVectorArray,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        axis: None | str | Sequence[str] = None,
        where: bool | na.AbstractScalar = True,
    ) -> na.FunctionArray[
        na.Cartesian2dVectorArray,
        na.AbstractScalar,
    ]:
        """
        Given a set of rays incident on the sensor surface,
        where each ray represents an expected number of photons per unit time,
        simulate the number of electrons that would be measured by the sensor.

        This method is inherently stochastic since it applies both photon shot
        noise and electron recombination noise to arrive at the final number of
        electrons measured.

        Parameters
        ----------
        rays
            A set of incident rays in global coordinates to measure.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        axis
            The logical axes along which to collect photons.
        where
            A boolean mask used to indicate which photons should be considered
            when calculating the signal measured by the sensor.
        """
        if timedelta is None:
            timedelta = self.timedelta_exposure

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        where = where & rays.unvignetted

        rays = dataclasses.replace(
            rays,
            intensity=rays.intensity * timedelta,
        )

        electrons = self.material.electrons_measured(
            rays=rays,
            normal=self.sag.normal(rays.position),
        )

        hist = na.histogram2d(
            x=rays.position.x,
            y=rays.position.y,
            bins=dict(
                x=self.num_pixel.x,
                y=self.num_pixel.y,
            ),
            axis=axis,
            min=self.aperture.bound_lower,
            max=self.aperture.bound_upper,
            weights=electrons * where,
        )

        return hist


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

    timedelta_exposure: u.Quantity | na.AbstractScalar = 0 * u.s
    """The exposure time of the sensor."""

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
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.name),
            optika.shape(self.width_pixel),
            optika.shape(self.num_pixel),
            optika.shape(self.timedelta_exposure),
            optika.shape(self.transformation),
        )

    @property
    def material(self) -> optika.materials.AbstractMaterial:
        return optika.sensors.IdealImagingSensorMaterial()


class AbstractCCD(
    AbstractImagingSensor[MaterialT],
):
    pass
