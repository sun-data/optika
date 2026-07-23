from typing import Any
import abc
import dataclasses
import numpy as np
import astropy.units as u
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
    An interface for a linear forward model of an optical system.

    A linear system approximates an exact
    :class:`~optika.systems.SequentialSystem` by characterizing it with a set
    of precomputed models instead of raytracing each scene: a
    :attr:`distortion` model mapping object-plane coordinates onto the detector,
    an :attr:`area_effective` model, and optional :attr:`vignetting` and
    :attr:`field_stop` models, together with a :attr:`sensor`.
    The forward model, :meth:`image`, conservatively regrids the scene through
    the distortion model, weighting each cell by the effective area, vignetting,
    and field stop, and converts the result into detector electrons.

    Concrete subclasses supply these models as attributes.
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
        """
        The broadcasted shape of the component models, which is the shape of
        any array-valued parametrization of this system.
        """
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
        plane to each pixel on the sensor plane.

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

    def weights_transposed(
        self,
        weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
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

        weights_area = self.area_effective(coordinates_cell.wavelength)

        weights_input = weights_vignetting * weights_area.value

        axis_pixel = self.sensor.axis_pixel

        result = na.regridding.transpose_weights_conservative(
            weights=weights,
            coordinates_input=position_sensor,
            coordinates_output=self.coordinates_sensor,
            axis_input=axis_field,
            axis_output=(axis_pixel.x, axis_pixel.y),
            weights_input=weights_input,
        )

        return result

    @property
    def weights_unit(self):
        """The units associated with :attr:`weights`."""
        return u.cm**2

    def image_from_weights(
        self,
        weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        noise: bool = True,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Apply a precomputed set of regridding weights to a scene, returning the
        electrons measured by the sensor.

        This reuses the (expensive) conservative-regridding operator built by
        :meth:`weights` for any scene: it integrates the spectral radiance over
        each object-plane voxel, regrids onto the sensor plane, and applies the
        sensor response
        (:meth:`~optika.sensors.AbstractImagingSensor.expose`).
        :meth:`image` is the special case that builds `weights` on the fly.

        Parameters
        ----------
        weights
            The conservative-regridding weights computed by :meth:`weights`.
        scene
            The spectral radiance of the observed scene, sampled on the
            vertices of each pixel on the object plane.
            The radiance may be given in either energy or photon units; the
            sensor converts energy to photons if necessary.
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
            if len(axis) != 1:  # pragma: nocover
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

        # integrate the spectral radiance over each voxel into a flux per unit
        # collecting area.
        rate = scene.outputs * volume

        rate_sensor = na.regridding.regrid_from_weights(
            *weights,
            values_input=rate,
        )

        # restore the unit stripped from `weights_input` inside `weights`
        # (the effective collecting area), turning the rate per unit area into
        # a rate per sensor pixel.
        rate_sensor = rate_sensor * self.weights_unit

        # the flux incident on the sensor cannot be negative
        rate_sensor = np.maximum(rate_sensor, 0)

        image = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=coordinates.wavelength,
                position=self.coordinates_sensor,
            ),
            outputs=rate_sensor,
        )

        return self.sensor.expose(
            image=image,
            direction=self.direction,
            axis_wavelength=axis_wavelength,
            noise=noise,
            **kwargs,
        )

    def backproject_from_weights(
        self,
        weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
        image: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        coordinates: na.SpectralPositionalVectorArray,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Apply a precomputed set of transposed regridding weights to a
        detector-plane image of electrons, returning the backprojected spectral
        radiance.

        This is the transpose of :meth:`image_from_weights`:
        :meth:`weights_transposed` builds the (expensive) transposed operator
        once, and this method reuses it to invert the sensor response
        (:meth:`~optika.sensors.AbstractImagingSensor.photons_absorbed`),
        spread the result back onto the object plane, and divide out the voxel
        volume. :meth:`backproject` is the special case that builds the weights
        on the fly.

        Parameters
        ----------
        weights
            The transposed regridding weights computed by
            :meth:`weights_transposed`.
        image
            The detector-plane image of electrons to project onto the object
            plane, as produced by :meth:`image`.
        coordinates
            The vertices of each pixel on the object plane to project onto.
        axis_wavelength
            The logical axis of `coordinates` corresponding to changing
            wavelength.
            If :obj:`None` (the default), the single axis of
            ``coordinates.wavelength`` that is not a field axis is used; a
            :class:`ValueError` is raised if there is more than one.
        axis_field
            The logical axes of `coordinates` corresponding to changing position
            on the object plane.
            If :obj:`None` (the default), the axes of ``coordinates.position``
            are used.
        """

        coordinates = coordinates.explicit

        if axis_field is None:
            axis_field = tuple(na.shape(coordinates.position))

        if axis_wavelength is None:
            axis = set(na.shape(coordinates.wavelength)) - set(axis_field)
            if len(axis) != 1:  # pragma: nocover
                raise ValueError(
                    f"unable to infer `axis_wavelength`: expected exactly one "
                    f"axis of `coordinates.wavelength` "
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

        # invert the detector response, mapping the measured electrons back into
        # the photon rate per pixel produced by `image_from_weights`.
        image = self.sensor.photons_absorbed(
            image,
            direction=self.direction,
            axis_wavelength=axis_wavelength,
        )

        radiance = na.regridding.regrid_from_weights(
            *weights,
            values_input=image.outputs,
        )

        # divide out the effective collecting area, the inverse of the
        # multiplication performed by `image_from_weights`, converting the
        # per-pixel rate back into a rate per unit collecting area.
        radiance = radiance / self.weights_unit

        # recover the spectral radiance by undoing the integration over each
        # object-plane voxel performed by `image`.
        radiance = radiance / volume

        return na.FunctionArray(
            inputs=coordinates,
            outputs=radiance,
        )

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
            The radiance may be given in either energy units (such as
            :math:`W / cm^2 / arcsec^2 / nm`) or photon units (such as
            :math:`photon / s / cm^2 / arcsec^2 / nm`); the sensor converts
            energy to photons if necessary.
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
            if len(axis) != 1:  # pragma: nocover
                raise ValueError(
                    f"unable to infer `axis_wavelength`: expected exactly one "
                    f"axis of `scene.inputs.wavelength` "
                    f"({na.shape(coordinates.wavelength)}) that is not a field "
                    f"axis ({axis_field}), got {axis}."
                )
            (axis_wavelength,) = axis

        # `weights` folds in the effective area, vignetting, and field stop and
        # is the expensive part of the forward model; `image_from_weights`
        # reuses it to integrate the radiance, regrid, and expose the sensor.
        weights = self.weights(
            coordinates=coordinates,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
        )

        return self.image_from_weights(
            weights,
            scene,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            noise=noise,
            **kwargs,
        )

    def backproject(
        self,
        image: (
            na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]
            | na.AbstractScalar
        ),
        coordinates: na.SpectralPositionalVectorArray,
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Transpose of the linear forward model, :meth:`image`.

        Inverts the detector response (:meth:`~optika.sensors.AbstractImagingSensor.expose`)
        and then applies the transpose of the optical regridding, projecting a
        detector-plane image of electrons back onto the object plane by
        spreading each pixel's value across every object-plane cell that could
        have contributed to it. This is the transpose of :meth:`image`, not its
        inverse (the geometric spreading is not undone, and the sensor noise is
        not recovered).

        Parameters
        ----------
        image
            The detector-plane image of electrons to project onto the object
            plane, as produced by :meth:`image`.
        coordinates
            The vertices of each pixel on the object plane to project onto.
        weights
            The forward regridding weights computed by :meth:`weights`.
            If :obj:`None` (the default), they are computed from `coordinates`.
            Supplying a precomputed value (for example one already built for
            :meth:`image`) avoids rebuilding the expensive regridding operator.
        axis_wavelength
            The logical axis of `coordinates` corresponding to changing
            wavelength.
            If :obj:`None` (the default), the single axis of
            ``coordinates.wavelength`` that is not a field axis is used; a
            :class:`ValueError` is raised if there is more than one.
        axis_field
            The logical axes of `coordinates` corresponding to changing position
            on the object plane.
            If :obj:`None` (the default), the axes of ``coordinates.position``
            are used.
        """

        coordinates = coordinates.explicit

        if axis_field is None:
            axis_field = tuple(na.shape(coordinates.position))

        if axis_wavelength is None:
            axis = set(na.shape(coordinates.wavelength)) - set(axis_field)
            if len(axis) != 1:  # pragma: nocover
                raise ValueError(
                    f"unable to infer `axis_wavelength`: expected exactly one "
                    f"axis of `coordinates.wavelength` "
                    f"({na.shape(coordinates.wavelength)}) that is not a field "
                    f"axis ({axis_field}), got {axis}."
                )
            (axis_wavelength,) = axis

        # `weights_transposed` is the expensive transposed operator;
        # `backproject_from_weights` reuses it to invert the sensor response,
        # regrid onto the object plane, and recover the spectral radiance.
        if weights is None:
            weights = self.weights(
                coordinates=coordinates,
                axis_wavelength=axis_wavelength,
                axis_field=axis_field,
            )
        weights_transposed = self.weights_transposed(
            weights=weights,
            coordinates=coordinates,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
        )

        return self.backproject_from_weights(
            weights_transposed,
            image,
            coordinates=coordinates,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
        )


@dataclasses.dataclass(eq=False, repr=False)
class LinearSystem(
    AbstractLinearSystem,
):
    """
    A linear forward model of an optical system, assembled from precomputed
    distortion, effective area, and (optionally) vignetting and field stop
    models.

    This is a fast approximation to
    :class:`~optika.systems.SequentialSystem`. Once a sequential system has been
    characterized (for example, its :meth:`~optika.systems.SequentialSystem.distortion`,
    :meth:`~optika.systems.SequentialSystem.vignetting`, and
    :meth:`~optika.systems.SequentialSystem.area_effective` models have been fit),
    those models can be reused here to image many scenes without raytracing each
    one.

    Examples
    --------

    Simulate an image of an airforce target using a simple spectrograph model.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # The distortion, effective area, and sensor models that
        # define the system.
        distortion = optika.distortion.SimpleDistortionModel(
            plate_scale=50 * u.arcsec / u.mm,
            dispersion=250 * u.nm / u.mm,
            angle=0 * u.deg,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(0, 0) * u.mm,
            ),
        )
        area_effective = optika.radiometry.InterpolatedEffectiveAreaModel(
            wavelength=na.linspace(500, 600, axis="wavelength", num=10) * u.nm,
            area=na.linspace(1, 2, axis="wavelength", num=10) * u.cm ** 2,
            axis_wavelength="wavelength",
        )
        sensor = optika.sensors.ImagingSensor(
            width_pixel=15 * u.um,
            axis_pixel=na.Cartesian2dVectorArray("detector_x", "detector_y"),
            timedelta_exposure=1 * u.s,
            num_pixel=na.Cartesian2dVectorArray(64, 64),
        )

        # Assemble the linear system from the component models.
        system = optika.systems.LinearSystem(
            area_effective=area_effective,
            distortion=distortion,
            sensor=sensor,
        )

        # Define the number of field points to sample.
        num_field = 2 * system.sensor.num_pixel

        # Define the scene as an airforce target. The coordinates (inputs)
        # are defined on cell vertices and the spectral radiance (outputs)
        # on cell centers.
        scene = na.FunctionArray(
            inputs=na.SpectralPositionalVectorArray(
                wavelength=na.linspace(549, 551, axis="wavelength", num=4) * u.nm,
                position=na.Cartesian2dVectorLinearSpace(
                    start=-15 * u.arcsec,
                    stop=+15 * u.arcsec,
                    axis=na.Cartesian2dVectorArray("field_x", "field_y"),
                    num=num_field + 1,
                ),
            ),
            outputs=optika.targets.airforce(
                axis_x="field_x",
                axis_y="field_y",
                num_x=num_field.x,
                num_y=num_field.y,
            ) * 1e-16 * u.W / u.cm ** 2 / u.arcsec ** 2 / u.nm,
        )

        # Simulate an image of the scene using the linear forward model.
        image = system.image(scene, noise=False)

        # Plot the original scene and the simulated image.
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                figsize=(8, 5),
                constrained_layout=True,
            )
            na.plt.pcolormesh(
                scene.inputs.position,
                C=scene.outputs.value,
                ax=ax[0],
            )
            na.plt.pcolormesh(
                image.inputs.position,
                C=image.outputs.value.sum("wavelength"),
                ax=ax[1],
            )
            ax[0].set_title("scene")
            ax[1].set_title("image")
    """

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
