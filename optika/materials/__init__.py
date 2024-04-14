"""
A subpackage for simulating the interaction of light with transparent
or reflective materials.
"""

from ._snells_law import snells_law
from . import profiles
from . import matrices
from ._layers import (
    AbstractLayer,
    Layer,
    AbstractLayerSequence,
    LayerSequence,
    PeriodicLayerSequence,
)
from ._materials import (
    AbstractMaterial,
    Vacuum,
    AbstractMirror,
    Mirror,
    MeasuredMirror,
)
from ._multilayers import (
    multilayer_efficiency,
    AbstractMultilayerMaterial,
    AbstractMultilayerFilm,
    MultilayerFilm,
    AbstractMultilayerMirror,
    MultilayerMirror,
)
from . import meshes
from ._thin_films import (
    AbstractThinFilmFilter,
    ThinFilmFilter,
)

__all__ = [
    "snells_law",
    "profiles",
    "matrices",
    "AbstractLayer",
    "Layer",
    "AbstractLayerSequence",
    "LayerSequence",
    "PeriodicLayerSequence",
    "AbstractMaterial",
    "Vacuum",
    "AbstractMirror",
    "Mirror",
    "MeasuredMirror",
    "multilayer_efficiency",
    "AbstractMultilayerMaterial",
    "AbstractMultilayerFilm",
    "MultilayerFilm",
    "AbstractMultilayerMirror",
    "MultilayerMirror",
    "meshes",
    "AbstractThinFilmFilter",
    "ThinFilmFilter",
]
