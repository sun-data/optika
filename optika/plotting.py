import abc
import dataclasses
import numpy.typing as npt
import matplotlib.axes
import matplotlib.lines
import named_arrays as na
import optika.transforms

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
        transform: None | optika.transforms.AbstractTransform = None,
        component_map: None | dict[str, str] = None,
        **kwargs,
    ) -> na.AbstractScalar:
        """
        Plot the selected components onto the given axes.

        Parameters
        ----------
        ax
            The matplotlib axes to plot onto
        transform
            Any extra transformations to apply to the coordinate system before
            plotting
        component_map
            An optional mapping that relates the coordinate system of the
            :class:`Plottable` instance to the coordinate system of the
            :class:`matplotlib.axes.Axes` instance.
            If :obj:`None`, it is equivalent to ``dict(x="x", y="y", z="z")``.
        kwargs
            Additional keyword arguments that will be passed along to
            :func:`named_arrays.plt.plot()`
        """
