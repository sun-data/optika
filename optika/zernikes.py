r"""
Zernike polynomials, useful for describing wavefront and figure errors.

The Zernike polynomials form an orthonormal basis on the unit disk and are
indexed here using Noll's convention :cite:p:`Noll1976`, where each polynomial
is normalized to have unit RMS over the unit disk.
"""

import functools
import math
import numpy as np
import named_arrays as na

__all__ = [
    "noll",
    "zernike",
    "zernike_gradient",
    "zernike_sum",
    "zernike_sum_gradient",
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


@functools.lru_cache
def _tables(num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Tabulate the signed azimuthal degree, the normalization constant, and the
    monomial coefficients of the radial polynomial, for the first `num` Noll
    indices.

    Element ``radial[i, e]`` is the coefficient of :math:`\rho^e` in the
    radial polynomial of the Zernike polynomial with Noll index
    :math:`j = i + 1`.

    Parameters
    ----------
    num
        The number of Zernike polynomials to tabulate.
    """
    quantum_numbers = [noll(j) for j in range(1, num + 1)]

    n = np.array([n for n, m in quantum_numbers])
    m = np.array([m for n, m in quantum_numbers])

    radial = np.zeros((num, n.max() + 1))
    for i, (n_i, m_i) in enumerate(quantum_numbers):
        for c, e in _coefficients_radial(n_i, abs(m_i)):
            radial[i, e] += c

    norm = np.where(m == 0, np.sqrt(n + 1), np.sqrt(2 * (n + 1)))

    for array in (m, norm, radial):
        array.flags.writeable = False

    return m, norm, radial


def _harmonics(
    coefficients: na.AbstractScalar,
    axis: str,
) -> list[tuple[int, int, list[na.AbstractScalar]]]:
    r"""
    Collect a sum of Zernike polynomials into one polynomial in
    :math:`\rho^2` for each azimuthal harmonic.

    The result is a list of ``(mu, sign, a)``, where the harmonic contributes

    .. code-block:: text

        rho ** mu * sum(a[k] * rho ** (2 * k) for k in ...) * T(mu * phi),

    and :math:`T` is the cosine if ``sign`` is positive and the sine if it is
    negative.
    Since this sums over the Noll axis, which is small, the coefficients are
    contracted before any array of evaluation points is touched.

    Parameters
    ----------
    coefficients
        The magnitude of each Zernike polynomial, along `axis`.
    axis
        The logical axis of `coefficients` indexing the Noll terms.
    """
    if axis not in coefficients.shape:
        raise ValueError(
            f"`coefficients` must vary along `axis`, {axis!r}, "
            f"got an array with shape {coefficients.shape}."
        )

    m, norm, radial = _tables(coefficients.shape[axis])

    degree = radial.shape[~0] - 1

    result = []
    for mu in range(degree + 1):
        for sign in (1, -1):

            if mu == 0 and sign < 0:  # the sine of zero is not a harmonic
                continue

            where = m == (sign * mu)
            if not where.any():
                continue

            # the radial polynomial of R_n^mu has only the powers
            # rho ** mu, rho ** (mu + 2), ...
            a = []
            for e in range(mu, degree + 1, 2):
                weight = na.ScalarArray(
                    ndarray=np.where(where, norm * radial[:, e], 0),
                    axes=axis,
                )
                a.append((coefficients * weight).sum(axis=axis))

            result.append((mu, sign, a))

    return result


def zernike_sum(
    position: na.AbstractCartesian2dVectorArray,
    coefficients: na.AbstractScalar,
    axis: str,
) -> na.AbstractScalar:
    r"""
    Evaluate a weighted sum of Zernike polynomials at the given points on the
    unit disk.

    This is equivalent to summing :func:`zernike` over the Noll indices, but
    its cost per point scales with the radial degree of the basis rather than
    with the number of terms.

    Parameters
    ----------
    position
        The normalized, dimensionless points at which to evaluate the sum.
        Points satisfying :math:`|\text{position}| \leq 1` are inside
        the unit disk.
    coefficients
        The magnitude of each Zernike polynomial, along `axis`, where element
        :math:`i` is the coefficient of the polynomial with Noll index
        :math:`j = i + 1`.
    axis
        The logical axis of `coefficients` indexing the Noll terms.

    Examples
    --------

    Plot a wavefront composed of defocus, coma, and spherical aberration.

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

        coefficients = na.ScalarArray(
            ndarray=np.array([0, 0, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0.05]),
            axes="zernike",
        )

        w = optika.zernikes.zernike_sum(position, coefficients, axis="zernike")
        w[position.length > 1] = np.nan

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.pcolormesh(position, C=w, ax=ax)
        ax.set_aspect("equal");

    Notes
    -----
    Zernike polynomials which share an azimuthal degree :math:`m` share an
    azimuthal sinusoid, so the sum can be collected into one polynomial per
    harmonic,

    .. math::

        \sum_j c_j Z_j(\rho, \phi) = \sum_{m \geq 0}
            \left[ A_m(\rho) \cos(m \phi) + B_m(\rho) \sin(m \phi) \right].

    The radial polynomial :math:`R_n^m` contains only the powers
    :math:`\rho^{m}, \rho^{m + 2}, \dots`, so each collected polynomial
    factors as :math:`A_m(\rho) = \rho^m Q_m(\rho^2)`, and :math:`Q_m` is
    evaluated by Horner's method.

    Collecting the coefficients is a sum over the Noll index alone, so it
    happens on arrays the size of `coefficients` rather than the size of
    `position`.
    What remains is one sinusoid per harmonic instead of one per term, which
    is why the cost grows with the radial degree, roughly the square root of
    the number of terms.
    """
    harmonics = _harmonics(coefficients, axis)

    rho2 = np.square(position.x) + np.square(position.y)
    rho = np.sqrt(rho2)
    phi = np.arctan2(position.y, position.x)

    result = 0 * rho * coefficients[{axis: 0}]

    for mu, sign, a in harmonics:

        poly = a[~0]
        for k in reversed(range(len(a) - 1)):
            poly = poly * rho2 + a[k]

        if mu:
            poly = poly * rho**mu
            if sign > 0:
                poly = poly * np.cos(mu * phi)
            else:
                poly = poly * np.sin(mu * phi)

        result = result + poly

    return result


def zernike_sum_gradient(
    position: na.AbstractCartesian2dVectorArray,
    coefficients: na.AbstractScalar,
    axis: str,
) -> na.Cartesian2dVectorArray:
    r"""
    Evaluate the gradient of a weighted sum of Zernike polynomials at the
    given points on the unit disk.

    This is equivalent to summing :func:`zernike_gradient` over the Noll
    indices, but its cost per point scales with the radial degree of the basis
    rather than with the number of terms.

    Parameters
    ----------
    position
        The normalized, dimensionless points at which to evaluate the
        gradient.
    coefficients
        The magnitude of each Zernike polynomial, along `axis`, where element
        :math:`i` is the coefficient of the polynomial with Noll index
        :math:`j = i + 1`.
    axis
        The logical axis of `coefficients` indexing the Noll terms.

    Examples
    --------

    Evaluate the gradient of a wavefront made of defocus and coma.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na
        import optika

        coefficients = na.ScalarArray(
            ndarray=np.array([0, 0, 0, 0.2, 0, 0, 0, 0.1]),
            axes="zernike",
        )

        optika.zernikes.zernike_sum_gradient(
            position=na.Cartesian2dVectorArray(0.3, -0.4),
            coefficients=coefficients,
            axis="zernike",
        )

    Notes
    -----
    Differentiating the collected form described in :func:`zernike_sum` using
    the chain rule in polar coordinates gives

    .. math::

        \frac{\partial}{\partial \rho}
            \left[ \rho^m Q(\rho^2) \right]
            &= \rho^{m - 1} \left[ m Q(\rho^2)
               + 2 \rho^2 Q'(\rho^2) \right] \\
        \frac{1}{\rho} \left[ \rho^m Q(\rho^2) \right]
            &= \rho^{m - 1} Q(\rho^2),

    both of which are polynomials for :math:`m \geq 1`, while the
    :math:`m = 0` harmonic contributes nothing to the azimuthal derivative.
    Nothing divides by :math:`\rho`, so the gradient is exact at the center of
    the pupil, as it is in :func:`zernike_gradient`.
    """
    harmonics = _harmonics(coefficients, axis)

    rho2 = np.square(position.x) + np.square(position.y)
    rho = np.sqrt(rho2)
    phi = np.arctan2(position.y, position.x)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    zero = 0 * rho * coefficients[{axis: 0}]
    d_rho = zero
    d_phi_over_rho = zero

    for mu, sign, a in harmonics:

        # Q(rho ** 2) and its derivative, both by Horner's method.
        # The k = 0 term of Q' vanishes, so it must not shift the accumulator.
        q = a[~0]
        dq = 0 * a[~0]
        for k in reversed(range(len(a) - 1)):
            q = q * rho2 + a[k]
            dq = dq * rho2 + (k + 1) * a[k + 1]

        if mu:
            factor = rho ** (mu - 1) if mu > 1 else 1
            if sign > 0:
                t = np.cos(mu * phi)
                dt = -mu * np.sin(mu * phi)
            else:
                t = np.sin(mu * phi)
                dt = mu * np.cos(mu * phi)
            d_rho = d_rho + factor * (mu * q + 2 * rho2 * dq) * t
            d_phi_over_rho = d_phi_over_rho + factor * q * dt
        else:
            d_rho = d_rho + 2 * rho * dq

    return na.Cartesian2dVectorArray(
        x=d_rho * cos_phi - d_phi_over_rho * sin_phi,
        y=d_rho * sin_phi + d_phi_over_rho * cos_phi,
    )
