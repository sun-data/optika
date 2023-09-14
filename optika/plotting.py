import abc
import dataclasses
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import named_arrays as na

__all__ = [
    "Plottable",
]


@dataclasses.dataclass(eq=False, repr=False)
class Plottable(abc.ABC):
    @property
    @abc.abstractmethod
    def kwargs_plot(self) -> None | dict:
        """
        Extra keyword arguments that will be used in the call to
        :func:`named_arrays.plt.plot` within the :meth:`plot` method.
        """

    @abc.abstractmethod
    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
    ) -> na.AbstractScalar:
        """
        Plot the selected components onto the given axes.

        Parameters
        ----------
        ax
            The matplotlib axes to plot onto
        transformation
            Any extra transformations to apply to the coordinate system before
            plotting
        components
            Which 3d components to plot, helpful if plotting in 2d.
        kwargs
            Additional keyword arguments that will be passed along to
            :func:`named_arrays.plt.plot()`
        """
