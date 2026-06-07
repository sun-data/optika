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

    def bin_rays(
        self,
        rays: optika.rays.RayVectorArray,
        wavelength: na.AbstractScalar,
        axis: None | str | Sequence[str] = None,
        where: bool | na.AbstractScalar = True,
    ) -> tuple[
        na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        na.AbstractScalar,
    ]:
        """
        Bin a cloud of rays onto the pixel grid.

        Returns the per-pixel photon image (the binned ray intensity) and the
        flux-weighted mean cosine of the incidence angle in each pixel: the two
        quantities :meth:`expose` needs. Performing the ray-to-cosine projection
        here is what lets :meth:`expose` be shared with systems that have no rays.

        Parameters
        ----------
        rays
            A set of incident rays in local coordinates to bin.
        wavelength
            The edges of the wavelength bins to sample.
        axis
            The logical axes along which to collect photons.
        where
            A boolean mask used to indicate which rays should be considered.
        """
        where = where & rays.unvignetted

        normal = self.sag.normal(rays.position)
        direction = -(rays.direction @ normal)

        flux = rays.intensity * where

        bins = na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=na.Cartesian2dVectorLinearSpace(
                start=self.aperture.bound_lower.xy,
                stop=self.aperture.bound_upper.xy,
                axis=self.axis_pixel,
                num=self.num_pixel + 1,
            ),
        )
        a = na.SpectralPositionalVectorArray(
            wavelength=rays.wavelength,
            position=rays.position.xy,
        )

        image = na.histogram(a, bins=bins, axis=axis, weights=flux)
        moment = na.histogram(a, bins=bins, axis=axis, weights=flux * direction)

        # flux-weighted mean cosine; empty pixels carry no flux, assume normal incidence
        nonempty = image.outputs > 0
        direction = np.where(nonempty, moment.outputs / image.outputs, 1)

        return image, direction

    def expose(
        self,
        image: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalar,
        ],
        direction: float | na.AbstractScalar = 1,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        noise: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Convert a per-pixel photon image into the electrons measured by the sensor.

        This is the detector-physics step shared by every optical system: it
        operates on a pixel grid (a photon image plus an incidence-cosine map),
        so it can be driven by :meth:`bin_rays` for a ray-traced system, or by
        any model that produces a per-pixel photon image directly.

        The photon flux is multiplied by the exposure time and converted to
        electrons using
        :meth:`~optika.sensors.materials.AbstractSensorMaterial.signal`, which
        applies the quantum efficiency, noise, and charge-diffusion models.

        Parameters
        ----------
        image
            The expected photon flux incident on each pixel, as a function of
            wavelength and pixel position.
        direction
            The cosine of the incidence angle in each pixel.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        noise
            Whether to add shot noise and intrinsic sensor noise to the result.
        """
        if timedelta is None:
            timedelta = self.timedelta_exposure

        photons = image.outputs * timedelta
        sgn = np.sign(photons)

        electrons = self.material.signal(
            photons=np.abs(photons),
            wavelength=image.inputs.wavelength,
            direction=direction,
            width_pixel=self.width_pixel,
            axis_xy=(self.axis_pixel.x, self.axis_pixel.y),
            noise=noise,
        )

        return dataclasses.replace(image, outputs=sgn * electrons)

    def readout(
        self,
        rays: optika.rays.RayVectorArray,
        wavelength: na.AbstractScalar,
        axis: None | str | Sequence[str] = None,
        where: bool | na.AbstractScalar = True,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        noise: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Bin a set of rays onto the pixel grid and convert them to the electrons
        measured by the sensor.

        This composes :meth:`bin_rays` (gather rays into the pixel grid) with
        :meth:`expose` (apply the detector physics).

        Parameters
        ----------
        rays
            A set of incident rays in local coordinates to measure.
        wavelength
            The edges of the wavelength bins to sample.
        axis
            The logical axes along which to collect photons.
        where
            A boolean mask used to indicate which rays should be considered.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        noise
            Whether to add shot noise and intrinsic sensor noise to the result.
        """
        image, direction = self.bin_rays(
            rays=rays,
            wavelength=wavelength,
            axis=axis,
            where=where,
        )
        return self.expose(
            image,
            direction=direction,
            timedelta=timedelta,
            noise=noise,
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
