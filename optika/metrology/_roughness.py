import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "RoughnessParameters",
]


@dataclasses.dataclass(eq=False, repr=False)
class RoughnessParameters(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """The parameters needed to compute the roughness of an optical surface."""

    period_min: na.ScalarLike = 0 * u.mm
    """The minimum period to consider when calculating roughness."""

    period_max: na.ScalarLike = 0 * u.mm
    """The maximum period to consider when calculating roughness."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.period_min),
            optika.shape(self.period_max),
        )
