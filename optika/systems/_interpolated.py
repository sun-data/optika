from typing import Callable, Any
import abc
import dataclasses
import named_arrays as na
import optika
from . import AbstractSystem

__all__ = [
    "AbstractInterpolatedSystem",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractInterpolatedSystem(
    AbstractSystem,
):
    """
    Approximate an exact optical system using interpolation.
    """

    @property
    @abc.abstractmethod
    def distortion(self) -> Callable[
        [na.SpectralPositionalVectorArray],
        na.Cartesian2dVectorArray,
    ]:
        """
        A distortion model which maps positions on the object plane to
        positions on the detector.
        """

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
            The logical axis
        axis_field
            The logical axes corresponding to changing field coordinate.
        """

        position = coordinates.position

        position_new = self.distortion(coordinates)

        result = na.regridding.weights(
            coordinates_input=position,
            coordinates_output=position_new,
            axis_input=axis_field,
            axis_output=axis_field,
            method="conservative",
        )

        return result


    def image(
        self,
        scene: na.FunctionArray[na.SpectralPositionalVectorArray, na.AbstractScalar],
        **kwargs: Any,
    ) -> na.SpectralPositionalVectorArray:
        pass

