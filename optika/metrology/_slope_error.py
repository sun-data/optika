import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractSlopeErrorParameters",
    "SlopeErrorParameters",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSlopeErrorParameters(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """collection of parameters used to compute the slope error"""

    @property
    @abc.abstractmethod
    def step_size(self) -> na.ScalarLike:
        """the horizontal distance to use when measuring the slope"""

    @property
    @abc.abstractmethod
    def kernel_size(self) -> na.ScalarLike:
        """
        size of the boxcar kernel that is convolved with the wavefront error
        before measuring the slope
        """


@dataclasses.dataclass(eq=False, repr=False)
class SlopeErrorParameters(
    AbstractSlopeErrorParameters,
):
    step_size: na.ScalarLike = 0 * u.mm
    kernel_size: na.ScalarLike = 0 * u.mm

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.step_size),
            optika.shape(self.kernel_size),
        )
