"""
A collection of transfer matrices for computing the efficiency of optical films and
multilayers.
"""

from __future__ import annotations
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
    direction_left: float | na.AbstractScalar,
    direction_right: float | na.AbstractScalar,
    polarized_s: bool | na.AbstractScalar,
    n_left: float | na.AbstractScalar,
    n_right: float | na.AbstractScalar,
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
        The component of the incident light's propagation direction before
        the interface antiparallel to the surface normal.
    direction_right
        The component of the incident light's propagation direction after
        the interface antiparallel to the surface normal.
    polarized_s
        If :obj:`True`, the incident light is :math:`s`-polarized.
        If :obj:`False`, the incident light is :math:`p`-polarized.
    n_left
        The complex index of refraction on the left side of the interface.
    n_right
        The complex index of refraction on the right side of the interface.
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
            direction_left=1,
            direction_right=1,
            polarized_s=True,
            n_left=1,
            n_right=sio2.n(wavelength),
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

    direction_i = np.where(polarized_s, direction_i, np.conj(direction_i))
    direction_j = np.where(polarized_s, direction_j, np.conj(direction_j))

    n_i = n_left
    n_j = n_right

    impedance_i = np.where(polarized_s, n_i, 1 / n_i)
    impedance_j = np.where(polarized_s, n_j, 1 / n_j)

    q_i = direction_i * impedance_i
    q_j = direction_j * impedance_j

    a_ij = q_i + q_j

    r_ij = (q_i - q_j) / a_ij
    t_ij = 2 * q_i / a_ij

    if interface is not None:
        r_ij = r_ij * interface.reflectivity(
            wavelength=wavelength,
            direction=direction_i,
            n=n_i,
        )

    result = na.Cartesian2dMatrixArray(
        x=na.Cartesian2dVectorArray(1, r_ij),
        y=na.Cartesian2dVectorArray(r_ij, 1),
    )

    result = result / t_ij

    return result


def propagation(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    thickness: u.Quantity | na.AbstractScalar,
    n: u.Quantity | na.AbstractScalar,
) -> na.Cartesian2dMatrixArray:
    r"""
    Compute the propagation matrix, which propagates the electric field
    through a homogenous slab.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The component of the incident light's propagation direction
        antiparallel to the surface normal.
    thickness
        The thickness of the material.
    n
        The complex index of refraction of the material

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
            direction=1,
            thickness=10 * u.nm,
            n=sio2.n(wavelength),
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

    beta = 2 * np.pi * thickness * n * direction / wavelength

    return na.Cartesian2dMatrixArray(
        x=na.Cartesian2dVectorArray(np.exp(-1j * beta), 0),
        y=na.Cartesian2dVectorArray(0, np.exp(+1j * beta)),
    )


def transfer(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    polarized_s: bool | na.ScalarArray,
    thickness: u.Quantity | na.AbstractScalar,
    n: float | na.AbstractScalar,
    normal: na.AbstractCartesian3dVectorArray,
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
    polarized_s
        If :obj:`True`, the incident light is :math:`s`-polarized.
        If :obj:`False`, the incident light is :math:`p`-polarized.
    thickness
        The thickness of the homogenous slab.
    n
        The index of refraction of the material
    normal
        The vector perpendicular to the surface of the slab.

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
            polarized_s=True,
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

    direction = -direction @ normal
    direction_internal = -direction_internal @ normal

    matrix_refractive_left = refraction(
        wavelength=wavelength,
        direction_left=direction,
        direction_right=direction_internal,
        polarized_s=polarized_s,
        n_left=1,
        n_right=n,
    )

    matrix_propagation = propagation(
        wavelength=wavelength,
        direction=direction_internal,
        thickness=thickness,
        n=n,
    )

    matrix_refractive_right = refraction(
        wavelength=wavelength,
        direction_left=direction_internal,
        direction_right=direction,
        polarized_s=polarized_s,
        n_left=n,
        n_right=1,
    )

    return matrix_refractive_left @ matrix_propagation @ matrix_refractive_right
