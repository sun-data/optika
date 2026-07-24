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
    @abc.abstractmethod
    def read_noise(self) -> u.Quantity | na.AbstractScalar:
        """
        The standard deviation of the Gaussian read noise added to each pixel
        during readout, in electrons.
        """

    @property
    def aperture(self):
        """
        The light-sensitive aperture of the sensor.
        """
        return optika.apertures.RectangularAperture(
            half_width=self.width_pixel * self.num_pixel / 2,
        )

    def pixels(
        self,
        position: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractCartesian2dVectorArray:
        """
        Convert an in-plane position on the sensor plane into fractional pixel
        coordinates.

        Pixel ``0`` is the lower edge of the light-sensitive area (the same grid
        that :meth:`collect` bins onto), so an integer value lands on a pixel
        boundary and a half-integer on a pixel center.

        Parameters
        ----------
        position
            The in-plane (``xy``) position on the sensor plane, in physical
            units.
        """
        lower = self.aperture.bound_lower.xy
        return (position - lower) / self.width_pixel * u.pix

    def collect(
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
        flux-weighted mean cosine of the *refracted* angle inside the
        light-sensitive region in each pixel: the two quantities :meth:`expose`
        needs. Refracting each ray here (with its own ambient index of
        refraction) and binning the result is what lets :meth:`expose` and the
        material's :meth:`~optika.sensors.materials.AbstractSensorMaterial.signal`
        model be shared with systems that have no rays,
        without threading a separate ambient-index argument through them.

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

        # Cosine of the refracted angle inside the light-sensitive region,
        # folding in each ray's ambient index of refraction. This is generally
        # complex, so its real and imaginary parts are binned separately below.
        direction = self.material.direction_refracted(
            wavelength=rays.wavelength,
            direction=rays.direction,
            n=rays.n,
            normal=normal,
        )

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
        moment_real = na.histogram(
            a, bins=bins, axis=axis, weights=flux * np.real(direction)
        )
        moment_imag = na.histogram(
            a, bins=bins, axis=axis, weights=flux * np.imag(direction)
        )

        # flux-weighted mean refracted cosine; empty pixels carry no flux, so
        # assume normal incidence (cosine of 1).
        nonempty = image.outputs > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            direction_real = np.where(nonempty, moment_real.outputs / image.outputs, 1)
            direction_imag = np.where(nonempty, moment_imag.outputs / image.outputs, 0)
        direction = direction_real + direction_imag * 1j

        return image, direction

    @staticmethod
    def _collapse_wavelength(
        inputs: na.SpectralPositionalVectorArray,
        axis_wavelength: str,
    ) -> na.SpectralPositionalVectorArray:
        """Collapse a wavelength axis to its two band edges."""
        wavelength = inputs.wavelength
        return inputs.replace(
            wavelength=na.stack(
                arrays=[
                    wavelength[{axis_wavelength: +0}],
                    wavelength[{axis_wavelength: ~0}],
                ],
                axis=axis_wavelength,
            )
        )

    def _integrate(
        self,
        image: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalar,
        ],
        axis_wavelength: str,
        noise: bool,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """Sum electrons over wavelength into one readout, adding read noise."""
        electrons = image.outputs.sum(axis_wavelength)
        if noise:
            # add zero-mean Gaussian read noise once per readout
            electrons = na.random.normal(loc=electrons, scale=self.read_noise)
        inputs = self._collapse_wavelength(image.inputs, axis_wavelength)
        return dataclasses.replace(image, inputs=inputs, outputs=electrons)

    def expose(
        self,
        image: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalar,
        ],
        direction: float | na.AbstractScalar = 1,
        axis_wavelength: None | str = None,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        noise: bool = True,
        integrate: bool = True,
        uncertainty: bool = False,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Convert a per-pixel photon image into the electrons measured by the sensor.

        This is the detector-physics step shared by every optical system: it
        operates on a pixel grid (a photon image plus a refracted-cosine map),
        so it can be driven by :meth:`collect` for a ray-traced system, or by
        any model that produces a per-pixel photon image directly.

        The photon flux is multiplied by the exposure time and converted to
        electrons using
        :meth:`~optika.sensors.materials.AbstractSensorMaterial.signal`, which
        applies the quantum efficiency and the shot, Fano, and charge-diffusion
        noise per wavelength. If `integrate` is :obj:`True`, the electrons are
        then summed over wavelength into a single readout and the sensor's
        :attr:`read_noise` is added once.

        Parameters
        ----------
        image
            The expected photon flux incident on each pixel, as a function of
            wavelength and pixel position.
            The wavelength inputs (``image.inputs.wavelength``) must be the
            bin *edges*, not the centers.
        direction
            The cosine of the refracted angle inside the light-sensitive region
            in each pixel, as produced by :meth:`collect`.
        axis_wavelength
            The logical axis of `image` corresponding to changing wavelength.
            If :obj:`None` (the default), ``image.inputs.wavelength`` must have
            only one logical axis.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        noise
            Whether to add shot, Fano, and charge-diffusion noise per wavelength
            and (if `integrate`) read noise once per readout.
        integrate
            Whether to integrate the electrons over wavelength into a single
            readout, applying :attr:`read_noise` once.
            A real imaging sensor cannot resolve the individual wavelengths, so
            this defaults to :obj:`True`; :obj:`False` keeps the wavelengths
            separate for demonstration.
        uncertainty
            Whether to attach the standard deviation of the measurement noise to
            the result as a
            :class:`~named_arrays.NormalUncertainScalarArray`, computed with
            :meth:`uncertainty`.
            The width uses the *expected* electrons, so the noiseless signal
            model is only re-evaluated when `noise` is also :obj:`True`.
        """
        if axis_wavelength is None:
            shape_wavelength = na.shape(image.inputs.wavelength)
            if len(shape_wavelength) != 1:  # pragma: nocover
                raise ValueError(
                    f"if `axis_wavelength` is `None`, `image.inputs.wavelength` "
                    f"must have exactly one logical axis, got {shape_wavelength}."
                )
            (axis_wavelength,) = shape_wavelength

        if timedelta is None:
            timedelta = self.timedelta_exposure

        photons = image.outputs * timedelta
        wavelength = image.inputs.wavelength.cell_centers(axis_wavelength)

        def signal(noise: bool) -> na.AbstractScalar:
            return self.material.signal(
                photons=photons,
                wavelength=wavelength,
                direction=direction,
                width_pixel=self.width_pixel,
                axis_xy=(self.axis_pixel.x, self.axis_pixel.y),
                noise=noise,
            )

        result = dataclasses.replace(image, outputs=signal(noise))

        if uncertainty:
            # the width uses the expected electrons, which are the same as the
            # nominal result unless `noise` added a realization
            expected = result if not noise else image.replace(outputs=signal(False))
            width = self.uncertainty(
                expected,
                direction=direction,
                axis_wavelength=axis_wavelength,
                integrate=integrate,
            )

        if integrate:
            # a real sensor reads out all wavelengths at once
            result = self._integrate(result, axis_wavelength, noise)

        if uncertainty:
            result = result.replace(
                outputs=na.NormalUncertainScalarArray(
                    nominal=result.outputs,
                    width=width.outputs,
                ),
            )

        return result

    def photons_absorbed(
        self,
        image: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalar,
        ],
        direction: float | na.AbstractScalar = 1,
        axis_wavelength: None | str = None,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        integrate: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Invert :meth:`expose`, mapping the electrons measured in each pixel back
        into a photon flux absorbed by the light-sensitive region.

        The absorbance is *not* restored, since :meth:`expose` runs the
        detector with an absorbance of one (the absorbance is usually accounted
        for elsewhere, such as in the effective area of an optical system), so
        this only divides out the quantum yield, the charge collection
        efficiency, and the exposure time. It is the deterministic inverse of
        :meth:`expose`; the sensor noise is not undone.

        Parameters
        ----------
        image
            The electrons measured in each pixel, as a function of wavelength
            and pixel position.
            The wavelength inputs (``image.inputs.wavelength``) must be the
            bin *edges*, not the centers.
        direction
            The cosine of the refracted angle inside the light-sensitive region,
            matching the value passed to :meth:`expose`.
        axis_wavelength
            The logical axis of `image` corresponding to changing wavelength.
            If :obj:`None` (the default), ``image.inputs.wavelength`` must have
            only one logical axis.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        integrate
            Whether `image` is a single wavelength-integrated readout (as
            produced by :meth:`expose` with ``integrate=True``).
            If :obj:`True` (the default), the readout is spread uniformly across
            the wavelength bins before the per-wavelength inverse, mirroring the
            integration performed by :meth:`expose`.
        """
        if axis_wavelength is None:
            shape_wavelength = na.shape(image.inputs.wavelength)
            if len(shape_wavelength) != 1:  # pragma: nocover
                raise ValueError(
                    f"if `axis_wavelength` is `None`, `image.inputs.wavelength` "
                    f"must have exactly one logical axis, got {shape_wavelength}."
                )
            (axis_wavelength,) = shape_wavelength

        if timedelta is None:
            timedelta = self.timedelta_exposure

        electrons = image.outputs

        if integrate:
            # spread the integrated readout uniformly across the wavelength bins
            num_wavelength = na.shape(image.inputs.wavelength)[axis_wavelength] - 1
            electrons = electrons / num_wavelength

        photons = self.material.photons_absorbed(
            electrons=electrons,
            wavelength=image.inputs.wavelength.cell_centers(axis_wavelength),
            direction=direction,
        )

        return dataclasses.replace(image, outputs=photons / timedelta)

    def uncertainty(
        self,
        image: na.FunctionArray[
            na.SpectralPositionalVectorArray,
            na.AbstractScalar,
        ],
        direction: float | na.AbstractScalar = 1,
        axis_wavelength: None | str = None,
        integrate: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Compute the standard deviation of the noise in an image of electrons
        measured by the sensor.

        This uses the material's analytic per-wavelength noise model
        (:meth:`~optika.sensors.materials.AbstractSensorMaterial.uncertainty`),
        which accounts for shot, Fano, and partial-charge-collection noise. If
        `integrate` is :obj:`True`, the per-wavelength variances are summed in
        quadrature and the sensor's :attr:`read_noise` is added once, giving the
        deterministic counterpart of the noise added by :meth:`expose`.

        Parameters
        ----------
        image
            The electrons measured in each pixel, as a function of wavelength
            and pixel position.
            The wavelength inputs (``image.inputs.wavelength``) must be the
            bin *edges*, not the centers.
        direction
            The cosine of the refracted angle inside the light-sensitive region,
            matching the value passed to :meth:`expose`.
        axis_wavelength
            The logical axis of `image` corresponding to changing wavelength.
            If :obj:`None` (the default), ``image.inputs.wavelength`` must have
            only one logical axis.
        integrate
            Whether to integrate the noise over wavelength into a single
            readout: the per-wavelength variances are summed in quadrature and
            :attr:`read_noise` is added once.
            Defaults to :obj:`True`, matching :meth:`expose`.
        """
        if axis_wavelength is None:
            shape_wavelength = na.shape(image.inputs.wavelength)
            if len(shape_wavelength) != 1:  # pragma: nocover
                raise ValueError(
                    f"if `axis_wavelength` is `None`, `image.inputs.wavelength` "
                    f"must have exactly one logical axis, got {shape_wavelength}."
                )
            (axis_wavelength,) = shape_wavelength

        uncertainty = self.material.uncertainty(
            electrons=image.outputs,
            wavelength=image.inputs.wavelength.cell_centers(axis_wavelength),
            direction=direction,
        )

        inputs = image.inputs

        if integrate:
            # sum the per-wavelength variances and add the read noise once
            variance = np.square(uncertainty).sum(axis_wavelength)
            variance = variance + np.square(self.read_noise)
            uncertainty = np.sqrt(variance)
            inputs = self._collapse_wavelength(inputs, axis_wavelength)

        return dataclasses.replace(image, inputs=inputs, outputs=uncertainty)

    def measure(
        self,
        rays: optika.rays.RayVectorArray,
        wavelength: na.AbstractScalar,
        axis: None | str | Sequence[str] = None,
        axis_wavelength: None | str = None,
        where: bool | na.AbstractScalar = True,
        timedelta: None | u.Quantity | na.AbstractScalar = None,
        noise: bool = True,
        integrate: bool = True,
    ) -> na.FunctionArray[
        na.SpectralPositionalVectorArray,
        na.AbstractScalar,
    ]:
        """
        Bin a set of rays onto the pixel grid and convert them to the electrons
        measured by the sensor.

        This composes :meth:`collect` (gather rays into the pixel grid) with
        :meth:`expose` (apply the detector physics).

        Parameters
        ----------
        rays
            A set of incident rays in local coordinates to measure.
        wavelength
            The edges of the wavelength bins to sample.
        axis
            The logical axes along which to collect photons.
        axis_wavelength
            The logical axis of `wavelength` corresponding to changing
            wavelength coordinate, forwarded to :meth:`expose`.
            If :obj:`None` (the default), `wavelength` must have only one
            logical axis.
        where
            A boolean mask used to indicate which rays should be considered.
        timedelta
            The exposure time of the measurement.
            If :obj:`None` (the default), the value in :attr:`timedelta_exposure`
            will be used.
        noise
            Whether to add shot noise and intrinsic sensor noise to the result.
        """
        image, direction = self.collect(
            rays=rays,
            wavelength=wavelength,
            axis=axis,
            where=where,
        )
        return self.expose(
            image,
            direction=direction,
            axis_wavelength=axis_wavelength,
            timedelta=timedelta,
            noise=noise,
            integrate=integrate,
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

    read_noise: u.Quantity | na.AbstractScalar = 0 * u.electron
    """
    The standard deviation of the Gaussian read noise added to each pixel
    during readout, in electrons.
    """

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
            optika.shape(self.read_noise),
            optika.shape(self.transformation),
        )
