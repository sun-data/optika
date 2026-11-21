from typing import Any
import abc
import dataclasses
import numpy as np
import astropy.units as u
import astropy.constants
import named_arrays as na
import optika
from optika.sensors._sensors import _collapse_wavelength
from . import AbstractSystem

__all__ = [
    "AbstractLinearSystem",
    "LinearSystem",
]


def _radiance_to_unit(
    radiance: na.AbstractScalar,
    wavelength: na.AbstractScalar,
    unit: None | u.UnitBase,
) -> na.AbstractScalar:
    """
    Express a photon spectral radiance in `unit`, scaling by the energy per
    photon (:math:`hc/\\lambda`) when `unit` is an energy rather than a photon
    unit. A `unit` compatible with neither raises a
    :class:`~astropy.units.UnitConversionError`.
    """
    if unit is None:
        return radiance

    if unit.is_equivalent(na.unit_normalized(radiance)):
        return radiance.to(unit)

    # the natural (photon) radiance is scaled by the energy per photon to
    # express it in energy units.
    energy_per_photon = (
        astropy.constants.h * astropy.constants.c / wavelength / u.photon
    )
    return (radiance * energy_per_photon).to(unit)


def _check_chunk_size(chunk_size: None | int) -> None:
    """
    Raise a :class:`ValueError` unless `chunk_size` is :obj:`None` or a
    positive integer.
    """
    if chunk_size is None:
        return
    is_integer = isinstance(chunk_size, (int, np.integer))
    if isinstance(chunk_size, bool) or not is_integer or chunk_size < 1:
        raise ValueError(
            f"`chunk_size` must be `None` or a positive integer, got {chunk_size!r}."
        )


def _chunks(num: int, chunk_size: None | int) -> list[tuple[int, int]]:
    """
    The first bin and one past the last bin of each chunk when `num` bins are
    split into chunks of `chunk_size` bins.

    If `chunk_size` is :obj:`None`, every bin is in one chunk.
    A grid with no bins gives one empty chunk, so that it goes through the
    same path as any other grid.
    """
    if chunk_size is None:
        chunk_size = max(num, 1)
    result = [(i, min(i + chunk_size, num)) for i in range(0, num, chunk_size)]
    return result or [(0, 0)]


def _check_cells(
    a: na.AbstractArray,
    shape: dict[str, int],
    axes: tuple[str, ...],
    name: str,
) -> None:
    """
    Raise a :class:`ValueError` unless `a` has one value for each cell of a
    grid along each of `axes`, or a single value which broadcasts against all
    of them.

    Each chunk takes its cells of `a` by index, so a value for each vertex
    would otherwise be cut to fit, dropping the last one and shifting the rest
    by half a cell.

    Parameters
    ----------
    a
        The quantity defined on the cells of the grid.
    shape
        The number of vertices of the grid along each axis.
    axes
        The logical axes along which `a` is checked.
    name
        What `a` is, for the error message.
    """
    shape_a = na.shape(a)
    for axis in axes:
        num_cells = shape[axis] - 1
        num = shape_a.get(axis, 1)
        if num not in (1, num_cells):
            raise ValueError(
                f"{name} must have a value for each of the {num_cells} cells "
                f"of the grid along {axis!r}, or a single value, got {num}."
            )


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
    def field_stop(
        self,
    ) -> (
        None
        | optika.apertures.AbstractAperture
        | optika.radiometry.AbstractFieldStopModel
    ):
        """
        A model of the field stop which blocks light on the object plane.

        An aperture bounds the same field of view at every wavelength.  A
        field-stop model can move it with wavelength, as the field of view of
        a system whose field stop sits behind a dispersive element moves.
        """

    @property
    def field_stop_(self) -> None | optika.radiometry.AbstractFieldStopModel:
        """
        :attr:`field_stop` as a field-stop model, so that an aperture and a
        model can be evaluated alike.

        An aperture is taken to bound the same field of view at every
        wavelength.
        """
        field_stop = self.field_stop
        if isinstance(field_stop, optika.apertures.AbstractAperture):
            return optika.radiometry.ApertureFieldStopModel(aperture=field_stop)
        return field_stop

    @property
    @abc.abstractmethod
    def sensor(self) -> optika.sensors.AbstractImagingSensor:
        """
        A model of the sensor which converts incident light intensity to an
        electrical signal.
        """

    def footprint(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        num: None | int = None,
    ) -> na.AbstractCartesian2dVectorArray:
        """
        The outline of the field stop on the sensor at the given wavelengths.

        The wire of :attr:`field_stop` at each of the given wavelengths,
        mapped through :attr:`distortion` at that same wavelength: where the
        edge of the field of view lands, which is the outline of the window
        each wavelength illuminates.  The field stop
        :meth:`~optika.systems.SequentialSystem.linearize` carries is the
        half-light outline of the field of view, so this is where an edge
        measured in an image sits.  A field stop made of several pieces, along
        the :attr:`~optika.apertures.AbstractAperture.axis_compound` of its
        aperture, has an outline for each piece, along that axis.

        Parameters
        ----------
        wavelength
            The wavelengths at which to map the outline.
        num
            The total number of points along the outline of the field stop,
            see :meth:`optika.apertures.AbstractAperture.wire`.

        Raises
        ------
        ValueError
            If this system carries no field stop.
        """
        field_stop = self.field_stop_
        if field_stop is None:
            raise ValueError("this system carries no field stop to outline")
        coordinates = na.SpectralPositionalVectorArray(
            wavelength=wavelength,
            position=field_stop.wire(wavelength, num=num),
        )
        return self.distortion.distort(coordinates).position

    @property
    @abc.abstractmethod
    def direction(self):
        """The cosine of the incidence angle on the sensor surface."""

    @property
    def coordinates_sensor(self) -> na.AbstractCartesian2dVectorArray:
        """
        The vertices of the sensor pixel grid onto which the scene is
        regridded, derived from :attr:`sensor`.

        These are the pixel edges in fractional pixel coordinates
        (:meth:`~optika.sensors.AbstractImagingSensor.pixels`), matching the
        pixel units of :attr:`distortion`.
        """
        sensor = self.sensor
        return na.Cartesian2dVectorLinearSpace(
            start=sensor.pixels(sensor.aperture.bound_lower.xy),
            stop=sensor.pixels(sensor.aperture.bound_upper.xy),
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

    def _distort(
        self,
        coordinates: na.SpectralPositionalVectorArray,
    ) -> na.AbstractCartesian2dVectorArray:
        """
        The position on the sensor of each point in `coordinates`, with every
        axis of `coordinates`.

        A term of the distortion which is zero at every point, such as the
        dispersion of a scene whose every wavelength is the reference
        wavelength of the model, is dropped by :mod:`named_arrays` along with
        the axes it would have been broadcast along. The regridding needs the
        positions to have every axis of the scene, so they are broadcast
        against `coordinates`.
        """
        position = self.distortion.distort(coordinates).position
        shape = na.broadcast_shapes(position.shape, coordinates.shape)
        return position.broadcast_to(shape)

    def weights(
        self,
        coordinates: na.SpectralPositionalVectorArray,
        axis_wavelength: str,
        axis_field: tuple[str, str],
        device: None | str = None,
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
        device
            The device on which to build the weights, passed through to
            :func:`named_arrays.regridding.weights`.
            If :obj:`None` (the default), the weights are built on the host.
        """

        coordinates = coordinates.spectral_positional

        coordinates = coordinates.cell_centers(axis_wavelength)

        position_sensor = self._distort(coordinates)

        # the conservative-regridding weights apply per input *cell*, but
        # `coordinates` are the cell vertices, so evaluate the radiometric
        # factors at the field cell centers (one fewer point along each axis).
        coordinates_cell = coordinates.cell_centers(axis_field)

        vignetting = self.vignetting
        if vignetting is not None:
            weights_vignetting = vignetting(coordinates_cell)
        else:
            weights_vignetting = 1

        # evaluated at each cell's own wavelength, so that a field of view
        # which moves with wavelength is followed across the scene
        field_stop = self.field_stop_
        if field_stop is not None:
            weights_stop = field_stop(coordinates_cell)
        else:
            weights_stop = 1

        weights_area = self.area_effective(coordinates_cell.wavelength)

        weights_input = (
            weights_vignetting * weights_stop * weights_area.to_value(self.weights_unit)
        )

        axis_pixel = self.sensor.axis_pixel

        result = na.regridding.weights(
            coordinates_input=position_sensor,
            coordinates_output=self.coordinates_sensor,
            axis_input=axis_field,
            axis_output=(axis_pixel.x, axis_pixel.y),
            weights_input=weights_input,
            method="conservative",
            device=device,
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

        coordinates = coordinates.spectral_positional

        coordinates = coordinates.cell_centers(axis_wavelength)

        position_sensor = self._distort(coordinates)

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

        weights_input = weights_vignetting * weights_area.to_value(self.weights_unit)

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
        *,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        noise: bool = True,
        uncertainty: bool = False,
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
        integrate
            Whether to integrate the electrons over wavelength into a single
            sensor readout, applying the read noise once
            (see :meth:`~optika.sensors.AbstractImagingSensor.expose`).
            Defaults to :obj:`True`.
        noise
            Whether to include sensor noise in the result.
        uncertainty
            Whether to attach the standard deviation of the measurement noise
            to the result, as a
            :class:`~named_arrays.NormalUncertainScalarArray`, using the
            sensor's :meth:`~optika.sensors.AbstractImagingSensor.uncertainty`.
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
        # `.spectral_positional` accepts a scene defined on a Doppler grid (as
        # produced by the ctis `IdealInstrument`).
        volume = coordinates.spectral_positional.volume_cell(
            (axis_wavelength, *axis_field)
        )

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
            integrate=integrate,
            uncertainty=uncertainty,
            **kwargs,
        )

    def backproject_from_weights(
        self,
        weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
        image: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        coordinates: na.SpectralPositionalVectorArray,
        *,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        unit: None | u.UnitBase = None,
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
        integrate
            Whether `image` is a single wavelength-integrated readout, to be
            spread back over the wavelength bins before the transpose.
            Defaults to :obj:`True`.
        unit
            The unit of the backprojected spectral radiance.
            :meth:`image_from_weights` accepts a scene in either photon or
            energy units, so the backprojection is expressed in whichever the
            caller requests, converting between photon and energy units using
            the energy per photon (:math:`hc/\\lambda`).
            If :obj:`None` (the default), the radiance is left in the natural
            (photon) units of the backprojection and is not converted.
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
        # `.spectral_positional` accepts a Doppler grid (as used by the ctis
        # `IdealInstrument`); the radiance is returned on the original grid.
        volume = coordinates.spectral_positional.volume_cell(
            (axis_wavelength, *axis_field)
        )

        # an integrated readout only carries the two band edges, but the sensor
        # inverse spreads it across the wavelength bins it is reconstructed
        # onto; give it the full target grid so it spreads over the right number
        # of bins (and evaluates the quantum efficiency per bin) instead of
        # treating the readout as a single bin.
        if integrate:
            image = image.replace(
                inputs=image.inputs.replace(
                    wavelength=coordinates.spectral_positional.wavelength,
                ),
            )

        # invert the detector response, mapping the measured electrons back into
        # the photon rate per pixel produced by `image_from_weights`.
        image = self.sensor.photons_absorbed(
            image,
            direction=self.direction,
            axis_wavelength=axis_wavelength,
            integrate=integrate,
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

        # express the (photon) radiance in the requested unit, converting to
        # energy units with the energy per photon if necessary.
        radiance = _radiance_to_unit(
            radiance=radiance,
            wavelength=coordinates.wavelength.cell_centers(axis_wavelength),
            unit=unit,
        )

        return na.FunctionArray(
            inputs=coordinates,
            outputs=radiance,
        )

    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        *,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        noise: bool = True,
        uncertainty: bool = False,
        device: None | str = None,
        chunk_size: None | int = None,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Linear forward model of the optical system.

        Maps the spectral radiance of a scene to the electrons measured by the
        sensor.

        The weights built by :meth:`weights` take memory in proportion to the
        number of cells in the scene, which can exceed the memory of the host
        for a large scene.
        Setting `chunk_size` images the scene a few wavelength bins at a time
        instead, so that only the weights of one chunk are held at once.

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
        integrate
            Whether to integrate the electrons over wavelength into a single
            sensor readout, applying the read noise once
            (see :meth:`~optika.sensors.AbstractImagingSensor.expose`).
            Defaults to :obj:`True`.
        noise
            Whether to include sensor noise in the result.
        uncertainty
            Whether to attach the standard deviation of the measurement noise
            to the result, as a
            :class:`~named_arrays.NormalUncertainScalarArray`, using the
            sensor's :meth:`~optika.sensors.AbstractImagingSensor.uncertainty`.
        device
            The device on which to build and apply the weights, see
            :meth:`weights`. If :obj:`None` (the default), the host is used.
        chunk_size
            The number of wavelength bins of `scene` to image at once.
            Each chunk is imaged on its own and the electrons are summed over
            the chunks (or joined along `axis_wavelength` if `integrate` is
            :obj:`False`).
            The shot, Fano, and charge-diffusion noise is independent from one
            wavelength to the next and the read noise is added once, to the
            sum of the chunks, so the result does not depend on `chunk_size`
            if `noise` is :obj:`False`, and has the same distribution if it is
            :obj:`True`.
            If :obj:`None` (the default), the whole scene is imaged at once.
        kwargs
            Additional keyword arguments passed to the sensor's
            :meth:`~optika.sensors.AbstractImagingSensor.expose` method, such
            as `timedelta`.
        """

        _check_chunk_size(chunk_size)

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

        return self._image_chunked(
            scene,
            axis_wavelength=axis_wavelength,
            axis_field=axis_field,
            integrate=integrate,
            noise=noise,
            uncertainty=uncertainty,
            device=device,
            chunk_size=chunk_size,
            **kwargs,
        )

    def _image_chunked(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        *,
        axis_wavelength: str,
        axis_field: tuple[str, str],
        integrate: bool,
        noise: bool,
        uncertainty: bool,
        device: None | str,
        chunk_size: None | int,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Image an explicit `scene` `chunk_size` wavelength bins at a time, or
        all at once if `chunk_size` is :obj:`None`.

        Every call to :meth:`image` comes through here, so that the chunks are
        read out the same way whether there is one of them or many.
        See :meth:`image` for the meaning of each parameter.
        """

        coordinates = scene.inputs
        radiance = scene.outputs

        _check_cells(
            a=radiance,
            shape=na.shape(coordinates),
            axes=(axis_wavelength,) + tuple(axis_field),
            name="The radiance of `scene`",
        )

        # the coordinates are the vertices of the wavelength bins
        num_wavelength = na.shape(coordinates.wavelength)[axis_wavelength] - 1

        # a radiance which does not vary with wavelength broadcasts against
        # every chunk, so only one with a value for each bin is cut up
        varies = na.shape(radiance).get(axis_wavelength, 1) != 1

        electrons_total = None
        variance_total = None
        electrons_chunks = []

        for start, stop in _chunks(num_wavelength, chunk_size):

            # a chunk of bins has one more vertex than it has bins
            item = {axis_wavelength: slice(start, stop)}
            item_vertices = {axis_wavelength: slice(start, stop + 1)}
            scene_chunk = na.FunctionArray(
                inputs=coordinates[item_vertices],
                outputs=radiance[item] if varies else radiance,
            )

            weights = self.weights(
                coordinates=scene_chunk.inputs,
                axis_wavelength=axis_wavelength,
                axis_field=axis_field,
                device=device,
            )

            # the read noise is left out here and added once below, to the sum
            # of every chunk, since the sensor is read out only once
            image = self.image_from_weights(
                weights,
                scene_chunk,
                axis_wavelength=axis_wavelength,
                axis_field=axis_field,
                integrate=False,
                noise=noise,
                uncertainty=uncertainty,
                **kwargs,
            )

            # the weights of this chunk are released before those of the next
            # one are built, so that only one chunk is held at a time
            del weights

            electrons = image.outputs

            if not integrate:
                electrons_chunks.append(electrons)
                continue

            if uncertainty:
                variance = np.square(electrons.width).sum(axis_wavelength)
                if variance_total is None:
                    variance_total = variance
                else:
                    variance_total = variance_total + variance
                electrons = electrons.nominal

            electrons = electrons.sum(axis_wavelength)
            if electrons_total is None:
                electrons_total = electrons
            else:
                electrons_total = electrons_total + electrons

        inputs = na.SpectralPositionalVectorArray(
            wavelength=coordinates.wavelength,
            position=self.coordinates_sensor,
        )

        sensor = self.sensor

        if integrate:
            inputs = _collapse_wavelength(inputs, axis_wavelength)
            if uncertainty:
                electrons_total = na.NormalUncertainScalarArray(
                    nominal=electrons_total,
                    width=np.sqrt(variance_total),
                )
            outputs = sensor.readout(electrons_total, noise)
        elif len(electrons_chunks) == 1:
            (outputs,) = electrons_chunks
        elif uncertainty:
            outputs = na.NormalUncertainScalarArray(
                nominal=na.concatenate(
                    arrays=[e.nominal for e in electrons_chunks],
                    axis=axis_wavelength,
                ),
                width=na.concatenate(
                    arrays=[e.width for e in electrons_chunks],
                    axis=axis_wavelength,
                ),
            )
        else:
            outputs = na.concatenate(
                arrays=electrons_chunks,
                axis=axis_wavelength,
            )

        return na.FunctionArray(
            inputs=inputs,
            outputs=outputs,
        )

    def backproject(
        self,
        image: (
            na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]
            | na.AbstractScalar
        ),
        coordinates: na.SpectralPositionalVectorArray,
        *,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        unit: None | u.UnitBase = None,
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
        chunk_size: None | int = None,
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

        As in :meth:`image`, setting `chunk_size` backprojects onto a few
        wavelength bins of `coordinates` at a time, so that only the weights of
        one chunk are held at once.

        Parameters
        ----------
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
        integrate
            Whether `image` is a single wavelength-integrated readout, to be
            spread back over the wavelength bins before the transpose.
            Defaults to :obj:`True`.
        unit
            The unit of the backprojected spectral radiance.
            :meth:`image` accepts a scene in either photon or energy units, so
            the backprojection is expressed in whichever the caller requests,
            converting between photon and energy units using the energy per
            photon (:math:`hc/\\lambda`).
            If :obj:`None` (the default), the radiance is left in the natural
            (photon) units of the backprojection and is not converted.
        weights
            The forward regridding weights computed by :meth:`weights`.
            If :obj:`None` (the default), they are computed from `coordinates`.
            Supplying a precomputed value (for example one already built for
            :meth:`image`) avoids rebuilding the expensive regridding operator.
            They are the weights of every wavelength bin, so they cannot be
            combined with `chunk_size`.
        chunk_size
            The number of wavelength bins of `coordinates` to backproject onto
            at once.
            The backprojection onto each chunk is computed on its own and the
            chunks are joined along `axis_wavelength`, so the result does not
            depend on `chunk_size`.
            If :obj:`None` (the default), the whole grid is backprojected onto
            at once.
        """

        _check_chunk_size(chunk_size)
        if weights is not None and chunk_size is not None:
            raise ValueError(
                "`weights` holds the weights of every wavelength bin, so it "
                "cannot be combined with `chunk_size`."
            )

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

        # the coordinates are the vertices of the wavelength bins
        num_wavelength = na.shape(coordinates.wavelength)[axis_wavelength] - 1

        # a readout integrated over wavelength is spread across the bins of the
        # whole grid once, here, so that each chunk can then be inverted on its
        # own as if its bins had been read out separately
        if integrate:
            image = image.replace(
                inputs=image.inputs.replace(
                    wavelength=coordinates.spectral_positional.wavelength,
                ),
                outputs=self.sensor.spread_readout(image.outputs, num_wavelength),
            )

        electrons = image.outputs

        _check_cells(
            a=electrons,
            shape=na.shape(coordinates.wavelength),
            axes=(axis_wavelength,),
            name="The electrons of `image`",
        )

        # electrons which do not vary with wavelength, such as a spread
        # readout, broadcast against every chunk
        varies = na.shape(electrons).get(axis_wavelength, 1) != 1

        radiance_chunks = []

        for start, stop in _chunks(num_wavelength, chunk_size):

            # a chunk of bins has one more vertex than it has bins
            item = {axis_wavelength: slice(start, stop)}
            item_vertices = {axis_wavelength: slice(start, stop + 1)}
            coordinates_chunk = coordinates[item_vertices]
            image_chunk = na.FunctionArray(
                inputs=image.inputs[item_vertices],
                outputs=electrons[item] if varies else electrons,
            )

            # `weights_transposed` is the expensive transposed operator;
            # `backproject_from_weights` reuses it to invert the sensor
            # response, regrid onto the object plane, and recover the spectral
            # radiance.
            if weights is None:
                weights_chunk = self.weights(
                    coordinates=coordinates_chunk,
                    axis_wavelength=axis_wavelength,
                    axis_field=axis_field,
                )
            else:
                weights_chunk = weights
            weights_transposed = self.weights_transposed(
                weights=weights_chunk,
                coordinates=coordinates_chunk,
                axis_wavelength=axis_wavelength,
                axis_field=axis_field,
            )

            # the weights of this chunk are released before those of the next
            # one are built, so that only one chunk is held at a time
            del weights_chunk

            radiance = self.backproject_from_weights(
                weights_transposed,
                image_chunk,
                coordinates=coordinates_chunk,
                axis_wavelength=axis_wavelength,
                axis_field=axis_field,
                integrate=False,
                unit=unit,
            )

            del weights_transposed

            radiance_chunks.append(radiance.outputs)

        if len(radiance_chunks) == 1:
            (outputs,) = radiance_chunks
        else:
            outputs = na.concatenate(
                arrays=radiance_chunks,
                axis=axis_wavelength,
            )

        return na.FunctionArray(
            inputs=coordinates,
            outputs=outputs,
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
            plate_scale=0.75 * u.arcsec / u.pix,
            dispersion=3.75 * u.nm / u.pix,
            angle=0 * u.deg,
            reference=na.SpectralPositionalVectorArray(
                wavelength=550 * u.nm,
                position=na.Cartesian2dVectorArray(32, 32) * u.pix,
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
                C=image.outputs.value,
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

    field_stop: (
        None
        | optika.apertures.AbstractAperture
        | optika.radiometry.AbstractFieldStopModel
    ) = None
    """
    A model of the field stop which blocks light on the object plane.

    An aperture bounds the same field of view at every wavelength, and a
    field-stop model can move it with wavelength.
    If :obj:`None` (the default), the sensor will be the field stop.
    """
