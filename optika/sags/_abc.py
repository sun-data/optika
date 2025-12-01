import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractSag(
    optika.mixins.Printable,
    optika.mixins.Transformable,
    optika.mixins.Shaped,
    optika.propagators.AbstractRayPropagator,
):
    """
    Base class for all types of sag profiles.
    """

    transformation: None | na.transformations.AbstractTransformation = (
        dataclasses.field(default=None, kw_only=True)
    )
    """
    The transformation between the surface coordinate system and the sag
    coordinate system.
    """

    parameters_slope_error: None | optika.metrology.SlopeErrorParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The slope error parameters for this sag profile."""

    parameters_roughness: None | optika.metrology.RoughnessParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The roughness parameters for this sag profile."""

    parameters_microroughness: None | optika.metrology.RoughnessParameters = (
        dataclasses.field(default=None, kw_only=True)
    )
    """The microroughness parameters for this sag profile."""

    @abc.abstractmethod
    def __call__(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractScalar:
        """
        Check if the given positions are inside the aperture.

        Parameters
        ----------
        position
            The points to check if inside the aperture.
            The :math:`z` coordinate is ignored.
        """

    @abc.abstractmethod
    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The vector perpendicular to the surface at the given position.

        Parameters
        ----------
        position
            The location on the surface to evaluate the normal vector
        """

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.RayVectorArray:
        """
        A set of new rays with the same direction as the input rays,
        but with the :attr:`optika.rays.RayVectorArray.position` updated to
        their interception point with this sag function.

        Parameters
        ----------
        rays
            input rays that will intercept this sag function
        """

        def line(t: na.AbstractScalar) -> na.Cartesian3dVectorArray:
            return rays.position + rays.direction * t

        def func(t: na.AbstractScalar) -> na.AbstractScalar:
            a = line(t)
            z = self(a)
            return a.z - z

        t_intercept = na.optimize.root_secant(
            function=func,
            guess=0 * u.mm,
            min_step_size=1e-6 * u.mm,
        )

        result = rays.copy_shallow()
        result.position = line(t_intercept)
        return result

    def propagate_rays(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:

        result = self.intercept(rays)

        displacement = result.position - rays.position

        f = np.exp(-result.attenuation * displacement.length)

        result.intensity = f * result.intensity

        return result
