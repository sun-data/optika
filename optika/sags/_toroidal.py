import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from ._abc import AbstractSag

__all__ = [
    "ToroidalSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class ToroidalSag(
    AbstractSag,
):
    """
    A toroidal sag profile.
    """

    radius: u.Quantity | na.AbstractScalar = np.inf * u.mm
    """The minor radius of this toroidal surface."""

    radius_of_rotation: u.Quantity | na.AbstractScalar = 0 * u.mm
    """The major radius of this toroidal surface."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.radius),
            optika.shape(self.radius_of_rotation),
            optika.shape(self.transformation),
            optika.shape(self.parameters_slope_error),
            optika.shape(self.parameters_roughness),
            optika.shape(self.parameters_microroughness),
        )

    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        c = 1 / self.radius
        r = self.radius_of_rotation
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(position, c, r)
        position = na.broadcast_to(position, shape)
        c = na.broadcast_to(c, shape)
        r = na.broadcast_to(r, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        zy = c * y2 / (1 + np.sqrt(1 - np.square(c) * y2))
        z = r - np.sqrt(np.square(r - zy) - x2)
        return z

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        c = 1 / self.radius
        r = self.radius_of_rotation
        transformation = self.transformation
        if transformation is not None:
            position = transformation.inverse(position)

        shape = na.shape_broadcasted(position, c, r)
        position = na.broadcast_to(position, shape)
        c = na.broadcast_to(c, shape)
        r = na.broadcast_to(r, shape)

        x2 = np.square(position.x)
        y2 = np.square(position.y)
        c2 = np.square(c)
        g = np.sqrt(1 - c2 * y2)
        zy = c * y2 / (1 + g)
        f = np.sqrt(np.square(r - zy) - x2)
        dzdx = position.x / f
        dzydy = c * position.y / g
        dzdy = (r - zy) * dzydy / f
        result = na.Cartesian3dVectorArray(
            x=dzdx,
            y=dzdy,
            z=-1 * u.dimensionless_unscaled,
        )
        return result / result.length
