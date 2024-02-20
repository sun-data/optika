"""
A collection of transfer matrices for computing the efficiency of optical films and
multilayers.
"""

from __future__ import annotations
from typing import Literal
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import snells_law


__all__ = [
    "refraction",
    "propagation",
    "transfer",
]


def refraction(
    wavelength: u.Quantity | na.AbstractScalar,
    direction_left: na.AbstractCartesian3dVectorArray,
    direction_right: na.AbstractCartesian3dVectorArray,
    polarization: Literal["s", "p"],
    n_left: float | na.AbstractScalar,
    n_right: float | na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
    interface: None | optika.materials.profiles.AbstractInterfaceProfile = None,
) -> na.Cartesian2dMatrixArray:
    r"""
    Compute the refractive matrix, which represents the bending of light
    at an interface.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum
    direction_left
        The direction of the incident light on the left side of the interface.
    direction_right
        The direction of the incident light on the right side of the interface.
    polarization
        Flag controlling whether the incident light is :math:`s`- or
        :math:`p`-polarized.
    n_left
        The complex index of refraction on the left side of the interface.
    n_right
        The complex index of refraction on the right side of the interface.
    normal
        The vector perpendicular to the surface of this layer.
    interface
        The interface profile between the left side and the right side.

    Examples
    --------

    Compute the refractive matrix for :math:`s`-polarized light normally
    incident on the interface between vacuum and silicon dioxide.

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the wavelength of the incident light
        wavelength = 100 * u.AA

        # Initialize a representation of silicon dioxide
        sio2 = optika.chemicals.Chemical("SiO2")

        # Compute the refractive matrix
        optika.materials.matrices.refraction(
            wavelength=wavelength,
            direction_left=na.Cartesian3dVectorArray(0, 0, 1),
            direction_right=na.Cartesian3dVectorArray(0, 0, 1),
            polarization="s",
            n_left=1,
            n_right=sio2.n(wavelength),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

    Notes
    -----

    The refractive matrix is given by :cite:t:`Yeh1988` Equation 5.1-12,

    .. math::
        :label: refractive-matrix

        W_{kij} = \frac{1}{t_{kij}} \begin{pmatrix}
                                    1 & r_{kij} \\
                                    r_{kij} & 1 \\
                                  \end{pmatrix},

    where :math:`k=(s, p)` is the polarization state, :math:`i=j-1` is the index
    of the previous material, :math:`j` is the index of the current material,

    .. math::
        :label: fresnel-reflection

        r_{kij} = \frac{q_{ki} - q_{kj}}{q_{ki} + q_{kj}}

    is the Fresnel reflection coefficient between materials :math:`i` and :math:`j`,

    .. math::
        :label: fresnel-transmission

        t_{kij} = \frac{2 q_{ki}}{q_{ki} + q_{kj}}

    is the Fresnel transmission coefficient between materials :math:`i` and :math:`j`,

    .. math::

        q_{si} = n_i \cos \theta_i

    and

    .. math::

        q_{pi} = \frac{\cos \theta_i}{n_i}

    are the :math:`z` components of the wave's momentum for :math:`s` and
    :math:`p` polarization,
    :math:`n_i` is the index of refraction inside material :math:`i`,
    and :math:`\theta_i` is the angle between the wave's propagation direction
    and the vector normal to the interface inside material :math:`i`.
    """
    direction_i = direction_left
    direction_j = direction_right

    n_i = n_left
    n_j = n_right

    cos_theta_i = -direction_i @ normal
    cos_theta_j = -direction_j @ normal

    if polarization == "s":
        q_i = cos_theta_i * n_i
        q_j = cos_theta_j * n_j
    elif polarization == "p":
        q_i = cos_theta_i / n_i
        q_j = cos_theta_j / n_j
    else:  # pragma: nocover
        raise ValueError(
            f"Invalid polarization state '{polarization}', only 's' and 'p'"
            f"are valid polarization states."
        )

    a_ij = q_i + q_j

    r_ij = (q_i - q_j) / a_ij
    t_ij = 2 * q_i / a_ij

    if interface is not None:
        w_tilde = interface.reflectivity(
            wavelength=wavelength,
            direction=direction_i,
            normal=normal,
        )
        r_ij = w_tilde * r_ij

    result = na.Cartesian2dMatrixArray(
        x=na.Cartesian2dVectorArray(1, r_ij),
        y=na.Cartesian2dVectorArray(r_ij, 1),
    )

    result = result / t_ij

    return result


def propagation(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    thickness: u.Quantity | na.AbstractScalar,
    n: u.Quantity | na.AbstractScalar,
    normal: na.AbstractVectorArray,
) -> na.Cartesian2dMatrixArray:
    r"""
    Compute the propagation matrix, which propagates the electric field
    through a homogenous slab.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The propagation direction of the light within the material.
    thickness
        The thickness of the material.
    n
        The complex index of refraction of the material
    normal
        The vector perpendicular to the surface of the material.

    Examples
    --------

    Compute the propagation matrix for :math:`s`-polarized light normally
    incident on a layer of silicon dioxide

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the wavelength of the incident light
        wavelength = 100 * u.AA

        # Initialize a representation of silicon dioxide
        sio2 = optika.chemicals.Chemical("SiO2")

        # Compute the propagation matrix
        optika.materials.matrices.propagation(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
            thickness=10 * u.nm,
            n=sio2.n(wavelength),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

    Notes
    -----

    The propagation matrix for a homogenous slab is given by
    :cite:t:`Yeh1988` Equation 5.1-24,

    .. math::
        :label: propagation-matrix

        U = \begin{pmatrix}
                e^{-i \beta} & 0 \\
                0 & e^{i \beta} \\
            \end{pmatrix},

    where

    .. math::
        :label: propagation-phase

        \beta = \frac{2 \pi}{\lambda} n h \cos \theta

    is the phase change from propagating through the material,
    :math:`n` is the index of refraction inside the material,
    :math:`\lambda` is the wavelength of the incident light in vacuum,
    :math:`h` is the thickness of the material,
    and :math:`\theta` is the angle between the surface normal and the propagation
    direction of the incident light.
    """
    cos_theta = -direction @ normal

    beta = 2 * np.pi * thickness * n * cos_theta / wavelength

    return na.Cartesian2dMatrixArray(
        x=na.Cartesian2dVectorArray(np.exp(-1j * beta), 0),
        y=na.Cartesian2dVectorArray(0, np.exp(+1j * beta)),
    )


def transfer(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    polarization: Literal["s", "p"],
    thickness: u.Quantity | na.AbstractScalar,
    n: float | na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
    interface: None | optika.materials.profiles.AbstractInterfaceProfile = None,
) -> na.Cartesian2dMatrixArray:
    r"""
    Compute the transfer matrix for a homogenous slab of material using
    :func:`refraction` and :func:`propagation`.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The propagation direction of the incident light in vacuum.
    polarization
        Flag controlling whether the incident light is :math:`s`- or
        :math:`p`-polarized.
    thickness
        The thickness of the homogenous slab.
    n
        The index of refraction of the material
    normal
        The vector perpendicular to the surface of the slab.
    interface
        The interface profile of the right side of the slab.

    Examples
    --------

    Compute the transfer matrix of a 10-nm-thick slab of silicon dioxide for
    normally-incident :math:`s`-polarized light.

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the wavelength of the incident light
        wavelength = 100 * u.AA

        # Initialize a representation of silicon dioxide
        sio2 = optika.chemicals.Chemical("SiO2")

        # Compute the transfer matrix
        optika.materials.matrices.transfer(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
            polarization="s",
            thickness=10 * u.nm,
            n=sio2.n(wavelength),
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

    Notes
    -----

    If :math:`W_{kij}` is the refractive matrix for the interface on the left side
    of the slab (computed using :func:`refraction`),
    :math:`U_{kj}` is the propagation matrix for the slab (computed using
    :func:`propagation`), and :math:`W_{kji}` is the refractive matrix
    for the right side of the slab, then the transfer matrix for the slab can
    be computed using the product of these three matrices:

    .. math::
        :label: transfer-matrix

        T_{kj} = W_{kij} U_{kj} W_{kji}
    """

    direction_internal = snells_law(
        wavelength=wavelength,
        direction=direction,
        index_refraction=1,
        index_refraction_new=np.real(n),
        normal=normal,
    )

    matrix_refractive_left = refraction(
        wavelength=wavelength,
        direction_left=direction,
        direction_right=direction_internal,
        polarization=polarization,
        n_left=1,
        n_right=n,
        normal=normal,
        interface=None,
    )

    matrix_propagation = propagation(
        wavelength=wavelength,
        direction=direction_internal,
        thickness=thickness,
        n=n,
        normal=normal,
    )

    matrix_refractive_right = refraction(
        wavelength=wavelength,
        direction_left=direction_internal,
        direction_right=direction,
        polarization=polarization,
        n_left=n,
        n_right=1,
        normal=normal,
        interface=interface,
    )

    return matrix_refractive_left @ matrix_propagation @ matrix_refractive_right
