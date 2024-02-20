from __future__ import annotations
from typing import Literal
import abc
import dataclasses
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
from . import matrices

__all__ = [
    "AbstractLayer",
    "Layer",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractLayer(
    optika.mixins.Printable,
):
    """
    An interface for representing a single homogeneous layer or a sequence
    of homogeneous layers.
    """

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
        Plot this layer sequence.

        Parameters
        ----------
        z
            The vertical offset of the plot.
        ax
            The matplotlib axes instance on which to plot the layers.
        kwargs
            Additional keyword arguments
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
            x_label=1.1,
        )

        # Plot the layer
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            layer.plot(ax=ax)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.set_ylabel(f"z ({layers.thickness.unit:latex_inline})")
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

    x_label: float = 0.5
    """The horizontal coordinate of the label, in axis units."""

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
        z: u.Quantity = 0 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:
        """
        Plot this layer using :meth:`matplotlib.axes.Axes.axhspan`

        Parameters
        ----------
        z
            The vertical offset of the plotted layer.
        ax
            The matplotlib axes instance on which to plot the layer.
        kwargs
            Additional keyword arguments to pass into
            :meth:`matplotlib.axes.Axes.axhspan`.
        """
        if ax is None:
            ax = plt.gca()

        material = self.material
        thickness = self.thickness
        kwargs_plot = self.kwargs_plot
        x_label = self.x_label

        if kwargs_plot is not None:
            kwargs = kwargs_plot | kwargs

        result = []
        if thickness is not None:

            result = ax.axhspan(
                ymin=z,
                ymax=z + thickness,
                **kwargs,
            )
            result = [result]

            if x_label <= 0:
                ha = "right"
                x = 0
            elif 0 < x_label < 1:
                ha = "center"
                x = x_label
            else:
                ha = "left"
                x = 1

            y = y_label = z + thickness / 2

            ax.annotate(
                text=rf"{material} (${thickness.value:0.0f}\,${thickness.unit:latex_inline})",
                xy=(x, y),
                xytext=(x_label, y_label),
                ha=ha,
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                ),
                transform=ax.get_yaxis_transform(),
            )

        return result
