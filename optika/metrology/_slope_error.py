import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "SlopeErrorParameters",
]


@dataclasses.dataclass(eq=False, repr=False)
class SlopeErrorParameters(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """The parameters needed to compute the slope error."""

    step_size: na.ScalarLike = 0 * u.mm
    """The horizontal distance to use when measuring the slope"""

    kernel_size: na.ScalarLike = 0 * u.mm
    """The size of the boxcar kernel that is convolved with the wavefront error."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.step_size),
            optika.shape(self.kernel_size),
        )
