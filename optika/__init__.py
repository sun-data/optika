"""
A Python package for simulating and designing optical systems.
"""

from . import mixins
from ._util import shape, direction, angles
from . import vectors
from . import targets
from . import rays
from . import metrology
from . import sags
from . import chemicals
from . import materials
from . import apertures
from . import rulings
from . import propagators
from . import surfaces
from . import sensors
from . import systems

__all__ = [
    "mixins",
    "shape",
    "direction",
    "angles",
    "vectors",
    "targets",
    "rays",
    "metrology",
    "sags",
    "chemicals",
    "materials",
    "apertures",
    "rulings",
    "propagators",
    "surfaces",
    "sensors",
    "systems",
]
