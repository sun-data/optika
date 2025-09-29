"""
Models of light-sensitive materials designed to be used by
:class:`~optika.sensors.ImagingSensor`.
"""

from . import depletion
from ._materials import (
    AbstractSensorMaterial,
    IdealSensorMaterial,
    AbstractSiliconSensorMaterial,
    AbstractBackIlluminatedSiliconSensorMaterial,
    BackIlluminatedSiliconSensorMaterial,
)
from ._tektronix_tk512cb import tektronix_tk512cb
from ._e2v_ccd97 import e2v_ccd97
from ._e2v_ccd203 import e2v_ccd203

__all__ = [
    "depletion",
    "AbstractSensorMaterial",
    "IdealSensorMaterial",
    "AbstractSiliconSensorMaterial",
    "AbstractBackIlluminatedSiliconSensorMaterial",
    "BackIlluminatedSiliconSensorMaterial",
    "tektronix_tk512cb",
    "e2v_ccd203",
    "e2v_ccd97",
]
