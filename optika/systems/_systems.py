from __future__ import annotations
from typing import Any
import abc
import dataclasses
import named_arrays as na
import optika

__all__ = [
    "AbstractSystem",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSystem(
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
):
    """
    An interface describing an optical system.

    Could potentially be sequential or non-sequential.
    """

    @abc.abstractmethod
    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        noise: bool = True,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Forward model of the optical system.
        Maps the given spectral radiance of a scene to the electrons measured
        by the sensor.

        Parameters
        ----------
        scene
            The spectral radiance of the scene as a function of wavelength
            and field position.
        axis_wavelength
            The logical axis of `scene` corresponding to changing wavelength.
        axis_field
            The logical axes of `scene` corresponding to changing field position.
        integrate
            Whether to integrate the electrons over wavelength into a single
            sensor readout.
        noise
            Whether to include sensor noise in the result.
        kwargs
            Additional keyword arguments used by subclass implementations
            of this method.
        """

    @abc.abstractmethod
    def backproject(
        self,
        image: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        coordinates: na.SpectralPositionalVectorArray,
        axis_wavelength: None | str = None,
        axis_field: None | tuple[str, str] = None,
        integrate: bool = True,
        **kwargs: Any,
    ) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar]:
        """
        Transpose of the forward model, :meth:`image`.
        Maps an image of measured electrons back onto the object plane.

        Parameters
        ----------
        image
            The detector-plane image of electrons to project onto the object
            plane.
        coordinates
            The vertices of each pixel on the object plane to project onto.
        axis_wavelength
            The logical axis of `coordinates` corresponding to changing
            wavelength.
        axis_field
            The logical axes of `coordinates` corresponding to changing field
            position.
        integrate
            Whether `image` is a single wavelength-integrated readout.
        kwargs
            Additional keyword arguments used by subclass implementations
            of this method.
        """
