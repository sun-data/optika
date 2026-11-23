"""
Meshes designed to support thin-film filters, such as the ones manufactured by
Luxel :cite:p:`Powell1990`.
"""

import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractMesh",
    "Mesh",
]


def _coefficient(
    order: int | na.AbstractScalar,
    fraction: float | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The Fourier coefficient of a square wave which is one over the given
    fraction of each period and zero over the rest,
    :math:`\sin(\pi m a) / (\pi m)`, which is :math:`a` for :math:`m = 0`.

    Parameters
    ----------
    order
        The index :math:`m` of the coefficient.
    fraction
        The fraction :math:`a` of each period over which the wave is one.
    """
    result = fraction * optika.rulings._rulings._sinc(order * fraction)
    return result.to(u.dimensionless_unscaled).value


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMesh(
    optika.mixins.Printable,
    optika.mixins.Replaceable,
    optika.mixins.Shaped,
):
    """
    An interface describing the supporting mesh for a thin-film filter.

    The mesh is a square grid of thin, opaque wires, which run along the
    :math:`x` and :math:`y` axes of the surface it supports, so the
    transformation of the surface orients it.
    Besides blocking some of the light, it diffracts the rest like a crossed
    transmission grating, into the orders of :meth:`efficiency_order`.
    """

    @property
    @abc.abstractmethod
    def chemical(self) -> str | optika.chemicals.AbstractChemical:
        """The chemical makeup of the mesh material."""

    @property
    @abc.abstractmethod
    def efficiency(self) -> float | na.AbstractScalar:
        """The fraction of light that is not blocked by the mesh."""

    @property
    @abc.abstractmethod
    def pitch(self) -> u.Quantity | na.AbstractScalar:
        """The density of the mesh in lines per inch or equivalent."""

    def efficiency_order(
        self,
        order: na.AbstractCartesian2dVectorArray,
    ) -> na.AbstractScalar:
        r"""
        The fraction of the incident light which this mesh diffracts into the
        given order.

        Light passes through the fraction :math:`a = \sqrt{\eta}` of each
        period of the mesh along each of its axes, where :math:`\eta` is
        :attr:`efficiency`, the fraction of the area which is open.
        So the transmission of the mesh is the product of two square waves,
        one along each axis, whose Fourier coefficients are

        .. math::

            c_m = \frac{\sin(\pi m a)}{\pi m},

        with :math:`c_0 = a`.
        The order :math:`(m, n)` leaves the mesh deflected as if by a grating
        of period :math:`d` of order :math:`m` along the :math:`x` axis of the
        mesh and of order :math:`n` along its :math:`y` axis, where :math:`d`
        is one over :attr:`pitch`, and carries the fraction
        :math:`c_m^2 c_n^2` of the incident light.
        These fractions sum to :attr:`efficiency` over all the orders, and the
        zeroth order carries :math:`\eta^2`.

        Parameters
        ----------
        order
            The diffraction order along the :math:`x` and :math:`y` axes of
            the mesh.

        Examples
        --------
        Plot the fraction of the incident light in each order along one axis
        of a mesh which blocks 18% of the light.

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            import astropy.units as u
            import named_arrays as na
            import optika

            mesh = optika.materials.meshes.Mesh(
                chemical="Ni",
                efficiency=0.82,
                pitch=70 / u.imperial.inch,
            )

            order = na.arange(-20, 21, axis="order")
            efficiency = mesh.efficiency_order(na.Cartesian2dVectorArray(order, 0))

            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.plot(
                order,
                efficiency,
                ax=ax,
                axis="order",
                marker="o",
                linestyle="none",
            )
            ax.set_yscale("log")
            ax.set_xlabel("order")
            ax.set_ylabel("fraction of the incident light")
        """
        fraction = np.sqrt(self.efficiency)
        coefficient_x = _coefficient(order.x, fraction)
        coefficient_y = _coefficient(order.y, fraction)
        return np.square(coefficient_x * coefficient_y)


@dataclasses.dataclass(eq=False, repr=False)
class Mesh(
    AbstractMesh,
):
    """
    An explicit description of a mesh designed to support a thin-film filter,
    a square grid of wires along the :math:`x` and :math:`y` axes of the
    surface it supports.
    """

    chemical: str | optika.chemicals.AbstractChemical = dataclasses.MISSING
    """The chemical formula of the mesh material."""

    efficiency: float | na.AbstractScalar = dataclasses.MISSING
    """The fraction of light that is not blocked by the mesh."""

    pitch: u.Quantity | na.AbstractScalar = dataclasses.MISSING
    """The density of the mesh in lines per inch or equivalent."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.chemical),
            optika.shape(self.efficiency),
            optika.shape(self.pitch),
        )
