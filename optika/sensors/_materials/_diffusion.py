import astropy.units as u
import numpy as np

import named_arrays as na

__all__ = [
    "charge_diffusion",
]


def charge_diffusion(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    The standard deviation of the charge diffusion in a backilluminated CCD
    given by :cite:t:`Janesick2001`.

    Parameters
    ----------
    absorption
        The absorption coefficient of the light-sensitive layer for the
        incident photon.
    thickness_implant
        The thickness of the partial-charge collection region of the imaging
        sensor.
    thickness_substrate
        The thickness of the light-sensitive region of the imaging sensor.
    thickness_depletion
        The thickness of the depletion region of the imaging sensor.

    Notes
    -----

    The standard deviation of the charge diffusion is given by
    :cite:t:`Janesick2001` as

    .. math::

        \sigma_d = x_{ff} \left( 1 - \frac{L_A}{x_{ff}} \right)^{1/2}

    where :math:`L_A = \ln{2} / \alpha` is the distance from the back surface at
    which the photon interacts with the sensor,
    :math:`\alpha` is the absorption coefficient of the light-sensitive layer,
    and :math:`x_{ff}` is the thickness of the field-free region of the sensor,
    given by

    .. math::

        x_{ff} = x_{\text{sub}} - x_{\text{PCC}} - x_{\text{dep}},

    where x_{\text{sub}} is the total thickness of the light-sensitive region,
    x_{\text{PCC}] is the thickness of the partial-charge collection region,
    and :math:`x_{\text{dep}}` is the thickness of the depletion region.
    """
    x_ff = thickness_substrate - thickness_implant - thickness_depletion

    depth_penetration = np.log(2) / absorption

    result = x_ff * np.sqrt(1 - depth_penetration / x_ff)

    return result
