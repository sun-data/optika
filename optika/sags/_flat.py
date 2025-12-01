import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag

__all__ = [
    "NoSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class NoSag(
    AbstractSag,
):
    """A flat sag profile."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        result = position.replace(z=0 * u.mm)

        if self.transformation is not None:
            result = self.transformation(result)

        return result.z

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        return na.Cartesian3dVectorArray(0, 0, -1)

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        if self.transformation is not None:
            rays = self.transformation.inverse(rays)

        d = -rays.position.z / rays.direction.z

        position = rays.position + rays.direction * d

        result = rays.replace(position=position)

        if self.transformation is not None:
            result = self.transformation(result)

        return result
