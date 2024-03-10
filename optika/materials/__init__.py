"""
A subpackage for simulating the interaction of light with transparent
or reflective materials.
"""

from ._snells_law import *
from . import profiles
from . import matrices
from ._layers import *
from ._materials import *
from ._multilayers import *
