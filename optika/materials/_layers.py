from __future__ import annotations
from typing import Sequence
import abc
import dataclasses
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
from . import matrices, snells_law_scalar

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
    optika.mixins.Shaped,
    abc.ABC,
):
    """
    An interface for representing a single homogeneous layer or a sequence
    of homogeneous layers.
    """

    @abc.abstractmethod
    def n(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The index of refraction on the left side of the layer.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum
        """

    @property
    @abc.abstractmethod
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of this layer."""

    @property
    @abc.abstractmethod
    def _thickness_plot(self) -> u.Quantity | na.AbstractScalar:
        """The thickness of the layer when plotted."""

    @property
    @abc.abstractmethod
    def interface(self) -> None | optika.materials.profiles.AbstractInterfaceProfile:
        """
        The interface profile on the left side of this layer.
        """

    @property
    @abc.abstractmethod
    def layer_sequence(self) -> LayerSequence:
        """
        This object interpreted as an instance of :class:`LayerSequence`.
        """

    @abc.abstractmethod
    def transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        polarized_s: bool | na.AbstractScalar,
        n: float | na.AbstractScalar,
        where: bool | na.AbstractScalar = True,
    ) -> tuple[
        na.AbstractScalar,
        na.AbstractScalar,
        na.Cartesian2dMatrixArray,
        na.AbstractScalar,
    ]:
        """
        Compute the index of refraction, internal propagation direction,
        transfer matrix, and where the transfer matrix is valid for this layer,
        which propagates the electric field from the left side of the layer to
        the right side.

        Parameters
        ----------
        wavelength
            The wavelength of the incident light in vacuum.
        direction
            The component of the incident light's propagation direction in the
            medium before this layer antiparallel to the surface normal.
        polarized_s
            If :obj:`True`, the incident light is :math:`s`-polarized.
            If :obj:`False`, the incident light is :math:`p`-polarized.
        n
            The complex index of refraction of the medium before this layer.
        where
            The boolean mask that should be applied to the transfer matrix.
            :obj:`True` where the transfer matrix is invertible.
        """

    def plot(
        self,
        x: u.Quantity = 0 * u.nm,
        z: u.Quantity = 0 * u.nm,
        width: u.Quantity = 10 * u.nm,
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
            chemical="Si",
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

    chemical: None | str | optika.chemicals.AbstractChemical = None
    """
    The chemical formula of the layer medium.
    If :obj:`None` (default), vacuum is assumed.
    """

    thickness: None | u.Quantity | na.AbstractScalar = None
    """The thickness of this layer"""

    interface: None | optika.materials.profiles.AbstractInterfaceProfile = None
    """
    The interface profile on the left side of this layer.
    """

    kwargs_plot: None | dict = None
    """Keyword arguments to be used in :meth:`plot` for styling of the layer."""

    x_label: None | u.Quantity = None
    """
    The horizontal coordinate of the label.
    If :obj:`None`, the label is plotted in the center of the layer
    """

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.chemical),
            optika.shape(self.thickness),
            optika.shape(self.interface),
        )

    @property
    def _thickness_plot(self) -> u.Quantity | na.AbstractScalar:
        return self.thickness

    @property
    def _chemical(self) -> optika.chemicals.AbstractChemical:
        result = self.chemical
        if not isinstance(result, optika.chemicals.AbstractChemical):
            result = optika.chemicals.Chemical(result)
        return result

    def n(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> float | na.AbstractScalar:
        """
        The complex index of refraction of this entire layer.
        """
        if self.chemical is None:
            return 1
        else:
            return self._chemical.n(wavelength)

    @property
    def layer_sequence(self) -> LayerSequence:
        return LayerSequence([self])

    def transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        polarized_s: bool | na.AbstractScalar,
        n: float | na.AbstractScalar,
        where: bool | na.AbstractScalar = True,
    ) -> tuple[
        na.AbstractScalar,
        na.AbstractScalar,
        na.Cartesian2dMatrixArray,
        na.AbstractScalar,
    ]:

        n_internal = self.n(wavelength)

        direction_internal = snells_law_scalar(
            cos_incidence=direction,
            index_refraction=n,
            index_refraction_new=n_internal,
        )

        refraction = matrices.refraction(
            wavelength=wavelength,
            direction_left=direction,
            direction_right=direction_internal,
            polarized_s=polarized_s,
            n_left=n,
            n_right=n_internal,
            interface=self.interface,
        )

        identity = na.Cartesian2dIdentityMatrixArray()
        refraction = np.where(where, refraction, identity)

        propagation = matrices.propagation(
            wavelength=wavelength,
            direction=direction_internal,
            thickness=self.thickness,
            n=n_internal,
        )

        where_propagation = np.abs(propagation.x.x) < 1e10
        where = where & where_propagation

        transfer = refraction @ propagation
        transfer = np.where(where, transfer, refraction)

        return n_internal, direction_internal, transfer, where

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

        chemical = self._chemical
        thickness = self.thickness
        kwargs_plot = self.kwargs_plot
        x_label = self.x_label

        if kwargs_plot is not None:
            kwargs = kwargs_plot | kwargs

        if x_label is None:
            x_label = x + width / 2
        else:
            x_label = x_label * width

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
                x_arrow = width

            y_arrow = y_label = z + thickness / 2

            formula = chemical.formula_latex

            ax.annotate(
                text=rf"{formula} (${thickness.value:0.0f}\,${thickness.unit:latex_inline})",
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
                chemical="Si",
                thickness=10 * u.nm,
                kwargs_plot=dict(
                    color="tab:blue",
                    alpha=0.5,
                ),
            ),
            optika.materials.Layer(
                chemical="Cr",
                thickness=5 * u.nm,
                kwargs_plot=dict(
                    color="tab:orange",
                    alpha=0.5,
                ),
            ),
            optika.materials.Layer(
                chemical="Zr",
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
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            *[optika.shape(layer) for layer in self.layers],
        )

    def n(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return self.layers[0].n(wavelength)

    @property
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        result = 0 * u.nm
        for layer in self.layers:
            result = result + layer.thickness
        return result

    @property
    def _thickness_plot(self) -> u.Quantity | na.AbstractScalar:
        return self.thickness

    @property
    def interface(self) -> None | optika.materials.profiles.AbstractInterfaceProfile:
        return self.layers[0].interface

    def __getitem__(self, item: int | slice) -> LayerSequence:
        if isinstance(item, int):
            return self.layers[item]
        else:
            return type(self)(self.layers[item])

    @property
    def layer_sequence(self) -> LayerSequence:
        return self

    def transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        polarized_s: bool | na.AbstractScalar,
        n: float | na.AbstractScalar,
        where: bool | na.AbstractScalar = True,
    ) -> tuple[
        na.AbstractScalar,
        na.AbstractScalar,
        na.Cartesian2dMatrixArray,
        na.AbstractScalar,
    ]:

        result = na.Cartesian2dIdentityMatrixArray()

        for layer in self.layers:
            n, direction, matrix_transfer, where = layer.transfer(
                wavelength=wavelength,
                direction=direction,
                polarized_s=polarized_s,
                n=n,
                where=where,
            )
            result = result @ matrix_transfer

        return n, direction, result, where

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
            z_current = z_current + layer._thickness_plot

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
                    chemical="Si",
                    thickness=10 * u.nm,
                    kwargs_plot=dict(
                        color="tab:blue",
                        alpha=0.5,
                    ),
                ),
                optika.materials.Layer(
                    chemical="Cr",
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
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            *[optika.shape(layer) for layer in self.layers],
        )

    def n(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        return self.layers[0].n(wavelength)

    @property
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        return self.num_periods * LayerSequence(self.layers).thickness

    @property
    def _thickness_plot(self) -> u.Quantity | na.AbstractScalar:
        return LayerSequence(self.layers).thickness

    @property
    def interface(self) -> None | optika.materials.profiles.AbstractInterfaceProfile:
        return self.layers[0].interface

    @property
    def layer_sequence(self) -> LayerSequence:
        return LayerSequence(self.layers * self.num_periods)

    def transfer(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        polarized_s: bool | na.AbstractScalar,
        n: float | na.AbstractScalar,
        where: bool | na.AbstractScalar = True,
    ) -> tuple[
        na.AbstractScalar,
        na.AbstractScalar,
        na.Cartesian2dMatrixArray,
        na.AbstractScalar,
    ]:

        period = LayerSequence(self.layers)

        n, direction, start, where = period.transfer(
            wavelength=wavelength,
            direction=direction,
            polarized_s=polarized_s,
            n=n,
            where=where,
        )

        n, direction, periodic, where = period.transfer(
            wavelength=wavelength,
            direction=direction,
            polarized_s=polarized_s,
            n=n,
            where=where,
        )

        periodic = periodic.power(self.num_periods - 1)

        return n, direction, start @ periodic, where

    def plot(
        self,
        x: u.Quantity = 0 * u.nm,
        z: u.Quantity = 0 * u.nm,
        width: u.Quantity = 10 * u.nm,
        ax: None | matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:

        layers = LayerSequence(self.layers)

        result = layers.plot(
            x=x,
            z=z,
            width=width,
            ax=ax,
            **kwargs,
        )

        na.plt.brace_vertical(
            x=x,
            width=0.05 * width,
            beta=1 / (0.02 * layers.thickness),
            ymin=z,
            ymax=z + layers.thickness,
            ax=ax,
            label=rf"$\times {self.num_periods}$",
            kind="left",
            color="black",
        )

        return result
