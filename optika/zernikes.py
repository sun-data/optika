r"""
Zernike polynomials, useful for describing wavefront and figure errors.

The Zernike polynomials form an orthonormal basis on the unit disk and are
indexed here using Noll's convention :cite:p:`Noll1976`, where each polynomial
is normalized to have unit RMS over the unit disk.
"""

import math
import numpy as np
import named_arrays as na

__all__ = [
    "noll",
    "zernike",
    "zernike_gradient",
]


def noll(j: int) -> tuple[int, int]:
    r"""
    Convert a Noll index :math:`j` into the corresponding Zernike quantum
    numbers :math:`(n, m)`.

    Parameters
    ----------
    j
        The Noll index of the Zernike polynomial.
        Must be greater than or equal to one.

    Examples
    --------

    Find the quantum numbers of the first four Zernike polynomials:
    piston, :math:`x` tilt, :math:`y` tilt, and defocus.

    .. jupyter-execute::

        import optika

        [optika.zernikes.noll(j) for j in (1, 2, 3, 4)]

    Notes
    -----
    The radial degree :math:`n` and the signed azimuthal degree :math:`m`
    follow Noll's ordering :cite:p:`Noll1976`:
    even :math:`j` corresponds to the cosine polynomials (:math:`m > 0`)
    and odd :math:`j` corresponds to the sine polynomials (:math:`m < 0`).
    """
    if j < 1:
        raise ValueError(f"Noll index must be greater than zero, got {j=}.")
    n = 0
    k = j - 1
    while k > n:
        n += 1
        k -= n
    m = (-1) ** j * ((n % 2) + 2 * ((k + ((n + 1) % 2)) // 2))
    return n, m


def _coefficients_radial(n: int, m: int) -> list[tuple[float, int]]:
    r"""
    The coefficients and exponents of the radial Zernike polynomial
    :math:`R_n^m(\rho) = \sum_k c_k \rho^{e_k}`.

    Parameters
    ----------
    n
        The radial degree of the Zernike polynomial.
    m
        The unsigned azimuthal degree of the Zernike polynomial.
    """
    result = []
    for k in range((n - m) // 2 + 1):
        c = (-1) ** k * math.factorial(n - k)
        c = c / math.factorial(k)
        c = c / math.factorial((n + m) // 2 - k)
        c = c / math.factorial((n - m) // 2 - k)
        result.append((c, n - 2 * k))
    return result


def zernike(
    position: na.AbstractCartesian2dVectorArray,
    j: int,
) -> na.AbstractScalar:
    r"""
    Evaluate the Zernike polynomial with the given Noll index at the given
    points on the unit disk.

    Parameters
    ----------
    position
        The normalized, dimensionless points at which to evaluate the
        polynomial.
        Points satisfying :math:`|\text{position}| \leq 1` are inside
        the unit disk.
    j
        The Noll index of the Zernike polynomial.
        Must be greater than or equal to one.

    Examples
    --------

    Plot the defocus polynomial, :math:`Z_4`, on the unit disk.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika

        position = na.Cartesian2dVectorLinearSpace(
            start=-1,
            stop=1,
            axis=na.Cartesian2dVectorArray("x", "y"),
            num=101,
        ).explicit

        z4 = optika.zernikes.zernike(position, 4)
        z4[position.length > 1] = np.nan

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.pcolormesh(position, C=z4, ax=ax)
        ax.set_aspect("equal");

    Notes
    -----
    The Zernike polynomial with quantum numbers :math:`(n, m)` is

    .. math::

        Z_n^m(\rho, \phi) = \begin{cases}
            \sqrt{n + 1} \, R_n^0(\rho), & m = 0 \\
            \sqrt{2 (n + 1)} \, R_n^m(\rho) \cos(m \phi), & m > 0 \\
            \sqrt{2 (n + 1)} \, R_n^{|m|}(\rho) \sin(|m| \phi), & m < 0,
        \end{cases}

    where the radial polynomial is

    .. math::

        R_n^m(\rho) = \sum_{k=0}^{(n - m) / 2}
            \frac{(-1)^k (n - k)!}{k! \left( \frac{n + m}{2} - k \right)!
            \left( \frac{n - m}{2} - k \right)!} \rho^{n - 2 k}.

    The normalization follows :cite:t:`Noll1976`, so that each polynomial has
    unit RMS over the unit disk.
    """
    n, m = noll(j)

    rho = position.length
    phi = np.arctan2(position.y, position.x)

    radial = 0 * rho
    for c, e in _coefficients_radial(n, abs(m)):
        radial = radial + c * rho**e

    if m == 0:
        return np.sqrt(n + 1) * radial
    elif m > 0:
        return np.sqrt(2 * (n + 1)) * radial * np.cos(m * phi)
    else:
        return np.sqrt(2 * (n + 1)) * radial * np.sin(-m * phi)


def zernike_gradient(
    position: na.AbstractCartesian2dVectorArray,
    j: int,
) -> na.Cartesian2dVectorArray:
    r"""
    Evaluate the gradient of the Zernike polynomial with the given Noll index
    at the given points on the unit disk.

    Parameters
    ----------
    position
        The normalized, dimensionless points at which to evaluate the
        gradient.
    j
        The Noll index of the Zernike polynomial.
        Must be greater than or equal to one.

    Notes
    -----
    The gradient is computed analytically using the chain rule in polar
    coordinates,

    .. math::

        \frac{\partial Z}{\partial x}
            &= N \left[ R'(\rho) \, T(m \phi) \cos \phi
               - \frac{R(\rho)}{\rho} \, T'(m \phi) \sin \phi \right] \\
        \frac{\partial Z}{\partial y}
            &= N \left[ R'(\rho) \, T(m \phi) \sin \phi
               + \frac{R(\rho)}{\rho} \, T'(m \phi) \cos \phi \right],

    where :math:`N` is the normalization constant, :math:`R` is the radial
    polynomial, and :math:`T` is the azimuthal sinusoid.
    Since the lowest-order term of :math:`R_n^m` is :math:`\rho^m`,
    the quotient :math:`R(\rho) / \rho` is itself a polynomial whenever
    :math:`m \geq 1`, and is evaluated as such to avoid dividing by zero at
    the origin.
    """
    n, m = noll(j)
    mu = abs(m)

    rho = position.length
    phi = np.arctan2(position.y, position.x)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    coefficients = _coefficients_radial(n, mu)

    d_radial = 0 * rho
    for c, e in coefficients:
        if e != 0:
            d_radial = d_radial + c * e * rho ** (e - 1)

    if m == 0:
        norm = np.sqrt(n + 1)
        return na.Cartesian2dVectorArray(
            x=norm * d_radial * cos_phi,
            y=norm * d_radial * sin_phi,
        )

    radial_over_rho = 0 * rho
    for c, e in coefficients:
        radial_over_rho = radial_over_rho + c * rho ** (e - 1)

    norm = np.sqrt(2 * (n + 1))
    if m > 0:
        t = np.cos(m * phi)
        dt = -m * np.sin(m * phi)
    else:
        t = np.sin(mu * phi)
        dt = mu * np.cos(mu * phi)

    return na.Cartesian2dVectorArray(
        x=norm * (d_radial * t * cos_phi - radial_over_rho * dt * sin_phi),
        y=norm * (d_radial * t * sin_phi + radial_over_rho * dt * cos_phi),
    )
