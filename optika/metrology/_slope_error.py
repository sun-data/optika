import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika.mixins

__all__ = [
    "AbstractSlopeErrorParameters",
    "SlopeErrorParameters",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSlopeErrorParameters(
    optika.mixins.Printable,
):
    """collection of parameters used to compute the slope error"""

    @property
    @abc.abstractmethod
    def kernel_size(self) -> na.ScalarLike:
        """
        size of the boxcar kernel that is convolved with the wavefront error
        before measuring the slope
        """

    @property
    @abc.abstractmethod
    def step_size(self) -> na.ScalarLike:
        """the horizontal distance to use when measuring the slope"""


@dataclasses.dataclass(eq=False, repr=False)
class SlopeErrorParameters(
    AbstractSlopeErrorParameters,
):
    kernel_size: na.ScalarLike = 0 * u.mm
    step_size: na.ScalarLike = 0 * u.mm