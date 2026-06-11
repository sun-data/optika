"""
A Python package for simulating and designing optical systems.
"""

from ._caching import memory
from . import mixins
from ._util import shape, direction, angles
from . import zernikes
from . import vectors
from . import targets
from . import rays
from . import wavefields
from . import propagators
from . import metrology
from . import sags
from . import chemicals
from . import materials
from . import apertures
from . import rulings
from . import surfaces
from . import sensors
from . import distortion
from . import radiometry
from . import systems

__all__ = [
    "memory",
    "mixins",
    "shape",
    "direction",
    "angles",
    "zernikes",
    "vectors",
    "targets",
    "rays",
    "wavefields",
    "propagators",
    "metrology",
    "sags",
    "chemicals",
    "materials",
    "apertures",
    "rulings",
    "surfaces",
    "sensors",
    "distortion",
    "radiometry",
    "systems",
]
