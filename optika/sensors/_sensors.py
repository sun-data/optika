"""
Models of light sensors that can be used in optical systems.
"""

from typing import TypeVar, Sequence
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from .materials import AbstractSensorMaterial, IdealSensorMaterial

__all__ = [
    "AbstractImagingSensor",
    "ImagingSensor",
]


MaterialT = TypeVar("MaterialT", bound=AbstractSensorMaterial)


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
    def axis_pixel(self) -> na.Cartesian2dVectorArray[str, str]:
        """
        The names of the logical axes corresponding to the rows and
        columns of the pixel grid.
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
        wavelength: na.AbstractScalar,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        axis: None | str | Sequence[str] = None,
        where: bool | na.AbstractScalar = True,
        noise: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Given a set of rays incident on the sensor surface,
        where each ray represents an expected number of photons per unit time,
        simulate the number of electrons that would be measured by the sensor.

        Parameters
        ----------
        rays
            A set of incident rays in local coordinates to measure.
        wavelength
            The edges of the wavelength bins to sample.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        axis
            The logical axes along which to collect photons.
        where
            A boolean mask used to indicate which photons should be considered
            when calculating the signal measured by the sensor.
        noise
            Whether to add noise to the result
        """
        if timedelta is None:
            timedelta = self.timedelta_exposure

        where = where & rays.unvignetted

        sgn = np.sign(rays.intensity)

        rays = dataclasses.replace(
            rays,
            intensity=np.abs(rays.intensity) * timedelta,
        )

        normal = self.sag.normal(rays.position)

        rays = self.material.signal(
            rays=rays,
            normal=normal,
            noise=noise,
        )

        rays = dataclasses.replace(
            rays,
            intensity=sgn * rays.intensity,
        )

        rays = self.material.charge_diffusion(
            rays=rays,
            normal=normal,
        )

        bins = na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=na.Cartesian2dVectorLinearSpace(
                start=self.aperture.bound_lower.xy,
                stop=self.aperture.bound_upper.xy,
                axis=self.axis_pixel,
                num=self.num_pixel + 1,
            ),
        )

        return na.histogram(
            a=na.SpectralPositionalVectorArray(
                wavelength=rays.wavelength,
                position=rays.position.xy,
            ),
            bins=bins,
            axis=axis,
            weights=rays.intensity * where,
        )


@dataclasses.dataclass(eq=False, repr=False)
class ImagingSensor(
    AbstractImagingSensor,
):
    """
    An arbitrary imaging sensor described by a pixel grid and a light-sensitive
    material.
    """

    name: None | str = None
    """The human-readable name of this sensor."""

    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.um
    """The physical size of each pixel on the sensor."""

    axis_pixel: na.Cartesian2dVectorArray[str, str] = None
    """
    The names of the logical axes corresponding to the rows and 
    columns of the pixel grid.
    """

    num_pixel: na.Cartesian2dVectorArray[int, int] = None
    """The number of pixels along each axis of the sensor."""

    timedelta_exposure: u.Quantity | na.AbstractScalar = 0 * u.s
    """The exposure time of the sensor."""

    material: AbstractSensorMaterial = None
    """
    A model of the light-sensitive material composing this sensor.
    
    If :obj:`None` (the default), :class:`optika.sensors.IdealImagingSensor`
    will be used.
    """

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

    def __post_init__(self) -> None:
        if self.material is None:
            self.material = IdealSensorMaterial()

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.name),
            optika.shape(self.width_pixel),
            optika.shape(self.num_pixel),
            optika.shape(self.timedelta_exposure),
            optika.shape(self.transformation),
        )
