from typing import Any
import abc
import dataclasses
import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
from . import AbstractSystem

__all__ = [
    "AbstractLinearSystem",
    "LinearSystem",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLinearSystem(
    AbstractSystem,
):
    """
    Approximate an exact optical system using a linear forward model.

    Subclasses must define a `distortion` method which maps coordinates on the
    object plane to positions on the detector.
    Subclasses must also define a `vignetting` method which controls how bright
    the scene appears.
    """

    @property
    @abc.abstractmethod
    def distortion(self) -> optika.distortion.AbstractDistortionModel:
        """
        A distortion model which maps coordinates on the object plane to
        positions on the detector plane.
        """

    @property
    @abc.abstractmethod
    def area_effective(self) -> optika.radiometry.AbstractEffectiveAreaModel:
        """A model of the effective area of the system's entrance aperture."""

    @property
    @abc.abstractmethod
    def vignetting(self) -> optika.radiometry.AbstractVignettingModel:
        """
        A vignetting model which only transmits a fraction of the light as a
        function of coordinates on the object plane.
        """

    @property
    @abc.abstractmethod
    def field_stop(self) -> optika.apertures.AbstractAperture:
        """
        A model of the field stop which blocks light on the object plane.
        """

    @property
    @abc.abstractmethod
    def sensor(self) -> optika.sensors.AbstractImagingSensor:
        """
        A model of the sensor which converts incident light intensity to an
        electrical signal.
        """

    @property
    @abc.abstractmethod
    def direction(self):
        """The cosine of the incidence angle on the sensor surface."""

    @property
    def coordinates_sensor(self) -> na.AbstractCartesian2dVectorArray:
        """
        The vertices of the sensor pixel grid onto which the scene is
        regridded, derived from :attr:`sensor`.

        This is the same grid of pixel edges that
        :meth:`~optika.sensors.AbstractImagingSensor.collect` bins onto.
        """
        sensor = self.sensor
        return na.Cartesian2dVectorLinearSpace(
            start=sensor.aperture.bound_lower.xy,
            stop=sensor.aperture.bound_upper.xy,
            axis=sensor.axis_pixel,
            num=sensor.num_pixel + 1,
        )

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.distortion),
            optika.shape(self.area_effective),
            optika.shape(self.vignetting),
            optika.shape(self.field_stop),
            optika.shape(self.sensor),
            optika.shape(self.direction),
        )

    @property
    def transformation(self) -> None:
        """
        A linear system has no geometric placement, so it has no
        transformation into the global coordinate system.
        """
        return None

    def weights(
        self,
        coordinates: na.SpectralPositionalVectorArray,
        axis_wavelength: str,
        axis_field: tuple[str, str],
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        Compute the weights which map the overlap of each pixel on the object
        plane to each pixel on the detector plane.

        Parameters
        ----------
        coordinates
            The vertices of each pixel on the object plane.
        axis_wavelength
            The logical axis corresponding to changing wavelength coordinate.
        axis_field
            The logical axes corresponding to changing field coordinate.
        """

        coordinates = coordinates.cell_centers(axis_wavelength)

        position_sensor = self.distortion.distort(coordinates).position

        # the conservative-regridding weights apply per input *cell*, but
        # `coordinates` are the cell vertices, so evaluate the radiometric
        # factors at the field cell centers (one fewer point along each axis).
        coordinates_cell = coordinates.cell_centers(axis_field)

        vignetting = self.vignetting
        if vignetting is not None:
            weights_vignetting = vignetting(coordinates_cell)
        else:
            weights_vignetting = 1

        field_stop = self.field_stop
        if field_stop is not None:
            weights_stop = field_stop(
                position=na.Cartesian3dVectorArray(
                    x=coordinates_cell.position.x,
                    y=coordinates_cell.position.y,
                ),
            )
        else:
            weights_stop = 1

        weights_area = self.area_effective(coordinates_cell.wavelength)

        weights_input = weights_vignetting * weights_stop * weights_area.value

        axis_pixel = self.sensor.axis_pixel

        result = na.regridding.weights(
            coordinates_input=position_sensor,
            coordinates_output=self.coordinates_sensor,
            axis_input=axis_field,
            axis_output=(axis_pixel.x, axis_pixel.y),
            weights_input=weights_input,
            method="conservative",
        )

        return result

    @property
    def weights_unit(self):
        """The units associated with :attr:`weights`."""
        return u.cm**2

    def image_from_weights(
        self,
        weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
        values_input: na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        Apply a precomputed set of regridding weights to the photon rate of a
        scene, mapping it onto the sensor plane.

        This is the cheap linear part of the forward model: :meth:`weights`
        builds the (expensive) conservative-regridding operator once, and this
        method reuses it for any scene.

        Parameters
        ----------
        weights
            The conservative-regridding weights computed by :meth:`weights`.
        values_input
            The photon rate within each pixel of the object plane.
        """

        values_output = na.regridding.regrid_from_weights(
            *weights,
            values_input=values_input,
        )

        # restore the unit stripped from `weights_input` inside `weights`
        # (the effective collecting area), turning the photon rate per unit
        # area into a photon rate per sensor pixel.
        values_output = values_output * self.weights_unit

        # photon counts cannot be negative
        values_output = np.maximum(values_output, 0)

        return values_output

    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        noise: bool = True,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Linear forward model of the optical system.

        Maps the spectral radiance of a scene to the electrons measured by the
        sensor.

        Parameters
        ----------
        scene
            The spectral radiance of the observed scene, sampled on the
            vertices of each pixel on the object plane.
        axis_wavelength
            The logical axis of `scene` corresponding to changing wavelength.
            If :obj:`None` (the default), the single axis of
            ``scene.inputs.wavelength`` that is not a field axis is used; a
            :class:`ValueError` is raised if there is more than one.
        axis_field
            The logical axes of `scene` corresponding to changing position on
            the object plane.
            If :obj:`None` (the default), the axes of ``scene.inputs.position``
            are used.
        noise
            Whether to include sensor noise in the result.
        kwargs
            Additional keyword arguments passed to the sensor's
            :meth:`~optika.sensors.AbstractImagingSensor.expose` method, such
            as `timedelta`.
        """

        scene = scene.explicit
        coordinates = scene.inputs

        if axis_field is None:
            axis_field = tuple(na.shape(coordinates.position))

        if axis_wavelength is None:
            axis = set(na.shape(coordinates.wavelength)) - set(axis_field)
            if len(axis) != 1:
                raise ValueError(
                    f"unable to infer `axis_wavelength`: expected exactly one "
                    f"axis of `scene.inputs.wavelength` "
                    f"({na.shape(coordinates.wavelength)}) that is not a field "
                    f"axis ({axis_field}), got {axis}."
                )
            (axis_wavelength,) = axis

        # volume of each voxel on the object plane: the spectral bin width
        # times the solid angle (or area) subtended by each field pixel.
        volume_wavelength = coordinates.wavelength.volume_cell(axis_wavelength)
        volume_field = coordinates.position.volume_cell(axis_field)
        volume_field = na.as_named_array(volume_field).cell_centers(axis_wavelength)
        volume = volume_wavelength * volume_field

        # energy carried by a single photon at each wavelength
        wavelength = coordinates.wavelength.cell_centers(axis_wavelength)
        energy_photon = astropy.constants.h * astropy.constants.c / wavelength / u.ph

        # integrate the spectral radiance over each voxel and convert it into
        # a photon rate per unit collecting area. `weights` folds in the
        # effective area, vignetting, and field stop during regridding.
        rate = scene.outputs * volume / energy_photon
        rate = rate.to(u.photon / u.s / self.weights_unit)

        weights = self.weights(
            coordinates=coordinates,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
        )

        rate_sensor = self.image_from_weights(weights, rate)

        image = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=coordinates.wavelength,
                position=self.coordinates_sensor,
            ),
            outputs=rate_sensor,
        )

        return self.sensor.expose(
            image,
            direction=self.direction,
            axis_wavelength=axis_wavelength,
            noise=noise,
            **kwargs,
        )


@dataclasses.dataclass(eq=False, repr=False)
class LinearSystem(
    AbstractLinearSystem,
):
    area_effective: optika.radiometry.AbstractEffectiveAreaModel = dataclasses.MISSING
    """A model of the effective area of the system's entrance aperture."""

    distortion: optika.distortion.AbstractDistortionModel = dataclasses.MISSING
    """
    A distortion model which maps coordinates on the object plane to
    positions on the detector plane.
    """

    sensor: optika.sensors.AbstractImagingSensor = dataclasses.MISSING
    """
    A model of the sensor which converts incident light intensity to an
    electrical signal.
    """

    direction: float | na.AbstractScalar = 1
    """
    The cosine of the incidence angle of the light striking the sensor surface.
    """

    vignetting: None | optika.radiometry.AbstractVignettingModel = None
    """
    A vignetting model which only transmits a fraction of the light as a
    function of coordinates on the object plane.
    If :obj:`None` (the default), the system will have no vignetting.
    """

    field_stop: None | optika.apertures.AbstractAperture = None
    """
    A model of the field stop which blocks light on the object plane.
    If :obj:`None` (the default), the sensor will be the field stop.
    """
