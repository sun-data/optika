from typing import Sequence
import dataclasses
import named_arrays as na
import optika
from ._abc import AbstractSag
from ._flat import NoSag

__all__ = [
    "FigureErrorSag",
]


@dataclasses.dataclass(eq=False, repr=False)
class FigureErrorSag(
    AbstractSag,
):
    r"""
    A sag profile consisting of a base profile plus a map of its figure
    error, such as one measured with an interferometer.

    The figure error raises the surface above the base profile, which a
    physical-optics calculation that samples the surface sees, but it does
    not tilt the surface.
    The normal vector of this sag profile, and the intercepts of rays with
    it, are those of the base profile, so the figure error does not deflect
    rays.
    A raytrace is then the same as through the base profile, unless this
    sag profile is on a stop surface.
    A system launches the rays of its stop solve from the stop surface
    itself, including the figure error, so there the figure error moves the
    rays of a raytrace slightly, by about its height.
    The slope of a measured map between its pixels is mostly noise, and where
    the map shifts the phase of the wavefront by less than about a radian, it
    scatters a little of the light instead of deflecting all of it, which
    only a physical-optics calculation can describe.
    Terms which are large and smooth enough to deflect rays, like a fit of
    the low-order Zernike polynomials, belong in :attr:`base`, for example as
    a :class:`optika.sags.ZernikeSag`, with the rest of the map here.

    The map is evaluated with multilinear interpolation, using
    :meth:`named_arrays.FunctionArray.__call__`, so its inputs should be the
    positions at which it was sampled, such as the centers of the pixels of
    an interferogram.
    Outside the map, the figure error is extrapolated linearly from the
    cells at its edge, so the map should cover the surface wherever it is
    evaluated.
    A pixel which was not measured should be filled, since the map is NaN
    next to a pixel which is NaN.

    Examples
    --------

    Plot the height of a parabolic sag profile above its base profile, along
    a slice which is sampled more finely than its map of the figure error.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # a map of the figure error, sampled on a grid of pixels
        pixel = na.Cartesian2dVectorLinearSpace(
            start=-50 * u.mm,
            stop=50 * u.mm,
            axis=na.Cartesian2dVectorArray("pixel_x", "pixel_y"),
            num=41,
        ).explicit
        height = 10 * u.nm * np.sin(pixel.x / (4 * u.mm) * u.rad) * np.cos(pixel.y / (7 * u.mm) * u.rad)

        sag = optika.sags.FigureErrorSag(
            base=optika.sags.ParabolicSag(focal_length=1000 * u.mm),
            figure_error=na.FunctionArray(pixel, height),
            axis=("pixel_x", "pixel_y"),
        )

        position = na.Cartesian3dVectorArray(
            x=na.linspace(-50, 50, axis="x", num=1001) * u.mm,
            y=0 * u.mm,
            z=0 * u.mm,
        )

        z = sag(position) - sag.base(position)

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(position.x, z.to(u.nm), axis="x", ax=ax)
            ax.set_xlabel(f"$x$ ({ax.get_xlabel()})")
            ax.set_ylabel(f"height above the base ({ax.get_ylabel()})")
    """

    base: None | AbstractSag = None
    """
    The base sag profile to be perturbed.
    If :obj:`None` (the default), a flat profile, :class:`optika.sags.NoSag`,
    is used.
    """

    figure_error: None | na.AbstractFunctionArray = None
    """
    The height of the surface above :attr:`base`, as a function of position.

    Its inputs are the :math:`x` and :math:`y` coordinates at which the map
    is sampled, in the coordinate system of this sag profile, as a
    :class:`named_arrays.Cartesian2dVectorArray`, and its outputs are the
    heights there.
    If :obj:`None` (the default), there is no figure error.
    """

    axis: None | Sequence[str] = None
    """
    The two logical axes along which :attr:`figure_error` is sampled.

    Any other axis of :attr:`figure_error`, such as one along which there is
    a separate map for each channel, is broadcast against the positions at
    which this sag profile is evaluated.
    Required if there is a map, and ignored if there is not.
    """

    @property
    def base_(self) -> AbstractSag:
        """The base sag profile, with :obj:`None` resolved to a flat profile."""
        if self.base is None:
            return NoSag()
        return self.base

    @property
    def axis_(self) -> tuple[str, ...]:
        """
        The logical axes along which :attr:`figure_error` is sampled, checked
        against its inputs, or none if there is no map.

        A misnamed axis is rejected here, so that :attr:`shape` fails fast
        instead of reporting an axis of the map as if it were a batch axis.
        """
        figure_error = self.figure_error
        if figure_error is None:
            return ()

        axis = self.axis
        if axis is None:
            raise ValueError(
                "`axis` must name the two axes along which `figure_error` is "
                "sampled."
            )
        if isinstance(axis, str):
            raise TypeError(
                f"`axis` must be a sequence of two axis names, not a str, {axis!r}."
            )
        axis = tuple(axis)

        shape_inputs = na.shape(figure_error.inputs)
        distinct = len(set(axis)) == len(axis) == 2
        if not distinct or any(a not in shape_inputs for a in axis):
            raise ValueError(
                f"`axis`, {axis}, must name two axes along which the inputs of "
                f"`figure_error` vary, {tuple(shape_inputs)}."
            )

        return axis

    @property
    def shape(self) -> dict[str, int]:
        figure_error = self.figure_error
        if figure_error is None:
            shape_figure_error = dict()
        else:
            axis = self.axis_
            shape_figure_error = {
                a: num for a, num in na.shape(figure_error).items() if a not in axis
            }
        return na.broadcast_shapes(
            optika.shape(self.base_),
            shape_figure_error,
            optika.shape(self.transformation),
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

        result = self.base_(position)

        figure_error = self.figure_error
        if figure_error is not None:
            figure_error = figure_error(position.xy, axis=self.axis_)
            result = result + figure_error.outputs

        return result

    def intercept(
        self,
        rays: optika.rays.AbstractRayVectorArray,
    ) -> optika.rays.AbstractRayVectorArray:
        """
        A set of new rays with the same direction as the input rays, but with
        the :attr:`optika.rays.RayVectorArray.position` updated to their
        interception point with :attr:`base`.

        The figure error is left out, since it does not deflect rays, and it
        would move each interception point along its ray by its height
        divided by the cosine of the angle of incidence.

        Parameters
        ----------
        rays
            input rays that will intercept this sag function
        """

        transformation = self.transformation
        if transformation is not None:
            rays = transformation.inverse(rays)

        result = self.base_.intercept(rays)

        if transformation is not None:
            result = transformation(result)

        return result

    def normal(
        self,
        position: na.AbstractCartesian3dVectorArray,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        The vector perpendicular to :attr:`base` at the given position.

        The slope of the figure error is left out, so that it does not
        deflect rays.

        Parameters
        ----------
        position
            The location on the surface to evaluate the normal vector
        """

        if self.transformation is not None:
            position = self.transformation.inverse(position)

        return self.base_.normal(position)
