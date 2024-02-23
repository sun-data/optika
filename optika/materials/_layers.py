from __future__ import annotations
from typing import Literal, Sequence
import abc
import dataclasses
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
from . import matrices

__all__ = [
    "AbstractLayer",
    "Layer",
    "AbstractLayerSequence",
    "LayerSequence",
    "PeriodicLayerSequence",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLayer(
    optika.mixins.Printable,
):
    """
    An interface for representing a single homogeneous layer or a sequence
    of homogeneous layers.
    """

    @property
    @abc.abstractmethod
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of this layer."""

    @abc.abstractmethod
    def matrix_transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        polarization: Literal["s", "p"],
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian2dMatrixArray:
        """
        Compute the transfer matrix for this layer, which propagates the
        electric field from the left side of the layer to the right side.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The propagation direction of the incident light in vacuum.
        polarization
            Flag controlling whether the incident light is :math:`s` or :math:`p`
            polarized.
        normal
            The vector perpendicular to the surface of this layer.
        """

    def plot(
        self,
        z: u.Quantity = 0 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:
        """
        Plot this layer.

        Parameters
        ----------
        x
            The horizontal offset of the plotted layer
        z
            The vertical offset of the plotted layer.
        width
            The horizontal width of the plotted layer.
        ax
            The matplotlib axes instance on which to plot the layer.
        kwargs
            Additional keyword arguments to pass into.
        """


@dataclasses.dataclass(eq=False, repr=False)
class Layer(
    AbstractLayer,
):
    """
    An isotropic, homogenous layer of optical material.

    Examples
    --------

    Plot a 10-nm-thick layer of silicon.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import optika

        # Create a layer of silicon
        layer = optika.materials.Layer(
            material="Si",
            thickness=10 * u.nm,
            kwargs_plot=dict(
                color="tab:blue",
                alpha=0.5,
            ),
        )

        # Plot the layer
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            layer.plot(ax=ax)
            ax.set_axis_off()
            ax.autoscale_view()
    """

    material: None | str = None
    """The chemical formula of the layer material."""

    thickness: None | u.Quantity | na.AbstractScalar = None
    """The thickness of this layer"""

    interface: None | optika.materials.profiles.AbstractInterfaceProfile = None
    """
    The interface profile for the right side of this layer.

    While it might be more natural to want to specify the interface profile for
    the left side of the layer, specifying the right side was chosen so that
    there would not be any coupling between subsequent layers.
    """

    kwargs_plot: None | dict = None
    """Keyword arguments to be used in :meth:`plot` for styling of the layer."""

    x_label: None | u.Quantity = None
    """
    The horizontal coordinate of the label.
    If :obj:`None`, the label is plotted in the center of the layer
    """

    @property
    def chemical(self) -> optika.chemicals.Chemical:
        """
        The chemical representation of this layer.
        """
        return optika.chemicals.Chemical(self.material)

    def matrix_transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        polarization: Literal["s", "p"],
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian2dMatrixArray:
        return matrices.transfer(
            wavelength=wavelength,
            direction=direction,
            polarization=polarization,
            thickness=self.thickness,
            n=self.chemical.n(wavelength),
            normal=normal,
            interface=self.interface,
        )

    def plot(
        self,
        x: u.Quantity = 0 * u.nm,
        z: u.Quantity = 0 * u.nm,
        width: u.Quantity = 1 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:
        """
        Plot this layer using  a :class:`matplotlib.patches.Rectangle` patch.

        Parameters
        ----------
        x
            The horizontal offset of the plotted layer
        z
            The vertical offset of the plotted layer.
        width
            The horizontal width of the plotted layer.
        ax
            The matplotlib axes instance on which to plot the layer.
        kwargs
            Additional keyword arguments to pass into
            :class:`matplotlib.patches.Rectangle`.
        """
        if ax is None:
            ax = plt.gca()

        material = self.material
        thickness = self.thickness
        kwargs_plot = self.kwargs_plot
        x_label = self.x_label

        if kwargs_plot is not None:
            kwargs = kwargs_plot | kwargs

        if x_label is None:
            x_label = x + width / 2

        unit_x = na.unit(x)
        unit_z = na.unit(z)

        result = []
        if thickness is not None:

            patch = matplotlib.patches.Rectangle(
                xy=(x.to_value(unit_x), z.to_value(unit_z)),
                width=width.to_value(unit_x),
                height=thickness.to_value(unit_z),
                **kwargs,
            )

            result = ax.add_patch(patch)

            result = [result]

            if x_label <= x:
                ha = "right"
                x_arrow = 0 * unit_x
            elif x < x_label < x + width:
                ha = "center"
                x_arrow = x_label
            else:
                ha = "left"
                x_arrow = 1 * unit_x

            y_arrow = y_label = z + thickness / 2

            ax.annotate(
                text=rf"{material} (${thickness.value:0.0f}\,${thickness.unit:latex_inline})",
                xy=(x_arrow.to_value(unit_x), y_arrow.to_value(unit_z)),
                xytext=(x_label.to_value(unit_x), y_label.to_value(unit_z)),
                ha=ha,
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                ),
                transform=ax.get_yaxis_transform(),
            )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLayerSequence(
    AbstractLayer,
):
    """
    An interface describing a sequence of layers.
    """

    @property
    @abc.abstractmethod
    def layers(self) -> Sequence[AbstractLayer]:
        """A sequence of layers."""


@dataclasses.dataclass(eq=False, repr=False)
class LayerSequence(AbstractLayerSequence):
    """
    An explicit sequence of layers.

    Examples
    --------

    Plot a Si/Cr/Zr stack of layers

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import optika

        # Define the layer stack
        layers = optika.materials.LayerSequence([
            optika.materials.Layer(
                material="Si",
                thickness=10 * u.nm,
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
            optika.materials.Layer(
                material="Cr",
                thickness=5 * u.nm,
                kwargs_plot=dict(
                    color="tab:orange",
                    alpha=0.5,
                ),
            ),
            optika.materials.Layer(
                material="Zr",
                thickness=15 * u.nm,
                kwargs_plot=dict(
                    color="tab:green",
                    alpha=0.5,
                ),
            ),
        ])

        # Plot the layer stack
        with astropy.visualization.quantity_support() as qs:
            fig, ax = plt.subplots(constrained_layout=True)
            layers.plot(ax=ax)
            ax.set_axis_off()
            ax.autoscale_view()
    """

    layers: Sequence[AbstractLayer] = dataclasses.MISSING
    """A sequence of layers."""

    @property
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        result = 0 * u.nm
        for layer in self.layers:
            result = result + layer.thickness
        return result

    def matrix_transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        polarization: Literal["s", "p"],
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian2dMatrixArray:

        result = na.Cartesian2dIdentityMatrixArray()

        for layer in self.layers:
            matrix_transfer = layer.matrix_transfer(
                wavelength=wavelength,
                direction=direction,
                polarization=polarization,
                normal=normal,
            )
            result = result @ matrix_transfer

        return result

    def plot(
        self,
        x: u.Quantity = 0 * u.nm,
        z: u.Quantity = 0 * u.nm,
        width: u.Quantity = 10 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:

        z_current = z

        result = []
        for layer in reversed(self.layers):
            result += layer.plot(
                x=x,
                z=z_current,
                width=width,
                ax=ax,
                **kwargs,
            )
            z_current = z_current + layer.thickness

        return result


@dataclasses.dataclass(eq=False, repr=False)
class PeriodicLayerSequence(AbstractLayerSequence):
    """
    A sequence of layers repeated an arbitrary number of times.

    This class is potentially much more efficient than :class:`LayerSequence`
    for large numbers of repeats.

    Examples
    --------

    Plot the periodic layer stack

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import optika

        # Define the layer stack
        layers = optika.materials.PeriodicLayerSequence(
            layers=[
                optika.materials.Layer(
                    material="Si",
                    thickness=10 * u.nm,
                    kwargs_plot=dict(
                        color="tab:blue",
                        alpha=0.5,
                    ),
                ),
                optika.materials.Layer(
                    material="Cr",
                    thickness=5 * u.nm,
                    kwargs_plot=dict(
                        color="tab:orange",
                        alpha=0.5,
                    ),
                ),
            ],
            num_periods=10,
        )

        # Plot the layer stack
        with astropy.visualization.quantity_support() as qs:
            fig, ax = plt.subplots(constrained_layout=True)
            layers.plot(ax=ax)
            ax.set_axis_off()
            ax.autoscale_view()
    """

    layers: Sequence[AbstractLayer] = dataclasses.MISSING
    """A sequence of layers."""

    num_periods: int = dataclasses.MISSING
    """The number of times to repeat the layer sequence."""

    @property
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        return self.num_periods * LayerSequence(self.layers).thickness

    def _chebyshev_u(self, n: int, x: float | na.AbstractScalar) -> na.AbstractScalar:
        return np.sin((n + 1) * np.arccos(x)) / np.sqrt(1 - np.square(x))

    def matrix_transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        polarization: Literal["s", "p"],
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.Cartesian2dMatrixArray:

        result = LayerSequence(self.layers).matrix_transfer(
            wavelength=wavelength,
            direction=direction,
            polarization=polarization,
            normal=normal,
        )

        n = self.num_periods

        a = (result.x.x + result.y.y) / 2

        un1a = self._chebyshev_u(n - 1, a)
        un2a = self._chebyshev_u(n - 2, a)

        result.x.x = result.x.x * un1a - un2a
        result.x.y = result.x.y * un1a
        result.y.x = result.y.x * un1a
        result.y.y = result.y.y * un1a - un2a

        return result

    def plot(
        self,
        z: u.Quantity = 0 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:

        layers = LayerSequence(self.layers)

        result = layers.plot(
            z=z,
            ax=ax,
            **kwargs,
        )

        na.plt.brace_vertical(
            x=0,
            width=0.5 * u.nm,
            ymin=z,
            ymax=z + layers.thickness,
            ax=ax,
            label=rf"$\times {self.num_periods}$",
            kind="left",
            color="black",
        )

        return result
