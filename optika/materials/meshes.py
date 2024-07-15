"""
Meshes designed to support thin-film filters, such as the ones manufactured by
Luxel :cite:p:`Powell1990`.
"""

import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractMesh",
    "Mesh",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMesh(
    optika.mixins.Printable,
    optika.mixins.Shaped,
):
    """
    An interface describing the supporting mesh for a thin-film filter.
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


@dataclasses.dataclass(eq=False, repr=False)
class Mesh(
    AbstractMesh,
):
    """
    An explicit description of a mesh designed to support a thin-film filter.
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
