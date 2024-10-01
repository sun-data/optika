from __future__ import annotations
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "snells_law_scalar",
    "snells_law",
]


def snells_law_scalar(
    cos_incidence: float | na.AbstractScalar,
    index_refraction: float | na.AbstractScalar,
    index_refraction_new: float | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    A simple form of Snell's law which computes the cosine of the angle
    between the propagation direction inside the new medium and the interface
    normal.

    Parameters
    ----------
    cos_incidence
        The cosine of the incidence angle (the angle between the propagation
        direction and the interface normal.)
    index_refraction
        The index of refraction in the current medium.
    index_refraction_new
        The index of refraction in the new medium.
    """
    if na.unit(cos_incidence) is not None:
        cos_incidence = cos_incidence.to(u.dimensionless_unscaled).value
    sin_incidence = np.emath.sqrt(1 - np.square(cos_incidence))
    sin_transmitted = index_refraction * sin_incidence / index_refraction_new
    cos_transmitted = np.emath.sqrt(1 - np.square(sin_transmitted))
    return cos_transmitted


def snells_law(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: na.AbstractCartesian3dVectorArray,
    index_refraction: float | na.AbstractScalar,
    index_refraction_new: float | na.AbstractScalar,
    normal: None | na.AbstractCartesian3dVectorArray,
    is_mirror: bool | na.AbstractScalar = False,
    diffraction_order: int = 0,
    spacing_rulings: None | u.Quantity | na.AbstractScalar = None,
    normal_rulings: None | na.AbstractCartesian3dVectorArray = None,
) -> na.Cartesian3dVectorArray:
    r"""
    A `vector form of Snell's law <https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form>`_,
    which has been modified to include the effects of diffraction from a periodic ruling pattern.

    Parameters
    ----------
    wavelength
        The wavelength of the incoming light
    direction
        The propagation direction of the incoming light
    index_refraction
        The index of refraction in the current propagation medium, :math:`n_1`.
    index_refraction_new
        The index of refraction in the new propagation medium, :math:`n_2`.
    normal
        The unit vector perpendicular to the interface between :math:`n_1`
        and :math:`n_2`.
    is_mirror
        A boolean flag controlling whether to compute the direction of the
        reflected ray or the transmitted ray.
        The default is to compute the transmitted ray.
    diffraction_order
        The diffraction order, :math:`m` of the reflected/transmitted rays.
        If nonzero, ``spacing_rulings`` and ``normal_rulings`` must be specified.
    spacing_rulings
        The distance between two grooves in the ruling pattern.
        If ``diffraction_order`` is zero, then this argument has no effect.
    normal_rulings
        The vector perpendicular to the plane of the grooves.

    Examples
    --------
    Plot the reflected and transmitted rays from a specified input ray.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the propagation direction of the
        # incident ray
        angle = 45 * u.deg
        direction = na.Cartesian3dVectorArray(
            x=np.sin(angle),
            y=0,
            z=-np.cos(angle),
        )

        # Define the keyword arguments that are common
        # to both the reflected and transmitted ray
        kwargs = dict(
            wavelength=350 * u.nm,
            direction=direction,
            index_refraction=1,
            normal=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Calculate the propagation direction of the
        # reflected ray
        direction_reflected = optika.materials.snells_law(
            is_mirror=True,
            index_refraction_new=1,
            **kwargs,
        )

        # Calculate the propagation direction of the
        # transmitted ray
        direction_transmitted = optika.materials.snells_law(
            is_mirror=False,
            index_refraction_new=2,
            **kwargs,
        )

        # Plot the incident, reflected, and transmitted rays
        fig, ax = plt.subplots()
        na.plt.plot(
            na.stack([-direction, 0], axis="plot"),
            axis="plot",
            components=("x", "z"),
            label="incident",
        );
        na.plt.plot(
            na.stack([0, direction_transmitted], axis="plot"),
            axis="plot",
            components=("x", "z"),
            label="transmitted",
        );
        na.plt.plot(
            na.stack([0, direction_reflected], axis="plot"),
            axis="plot",
            components=("x", "z"),
            label="reflected",
        );
        ax.set_aspect("equal");
        ax.axvline(0, linestyle="dashed", color="black");
        ax.axhspan(ymin=-2, ymax=0, color="lightgray");
        ax.set_ylim(-1, None);
        ax.legend();

    Notes
    -----

    Our goal is to derive a 3D version of Snell's law that can model the
    diffraction from a periodic ruling pattern (diffraction grating).

    To start, consider an incident wave of the form:

    .. math::
        :label: incident-wave

        E_1(\mathbf{r}) = A_1 e^{i \mathbf{k}_1 \cdot \mathbf{r}},

    where :math:`E_1` is the magnitude of the incident electric field,
    :math:`A_1` is the amplitude of the incident wave, and :math:`\mathbf{k}_1` is
    the incident wavevector.

    Now define an interface at :math:`z = 0`, where the index of
    refraction changes, and/or there is a periodic ruling pattern inscribed.
    When the incident wave interacts with this interface, it will create an
    output wave of the form:

    .. math::
        :label: transmitted-wave

        E_2(\mathbf{r}) = A_2 e^{i \mathbf{k}_2 \cdot \mathbf{r}},

    where :math:`E_2` is the magnitude of the output electric field,
    :math:`A_2` is the amplitude of the output wave, and :math:`\mathbf{k}_2` is
    the output wavevector.

    Note in this case we care about only the reflected `or` the transmitted
    wave, not both, since we're only concerned with sequential optics.

    If the interface at :math:`z = 0` `doesn't` have a periodic ruling pattern,
    :math:`E_1` and :math:`E_2` satisfy homogenous Dirichlet boundary conditions.

    .. math::
        :label: boundary-condition

        A_1 \exp\left[i \mathbf{k}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
        = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]

    To include the ruling pattern, we model it as a phase shift of the wave at the interface,

    .. math::
        :label: phase-shift

        \phi(x, y) = i \boldsymbol{\kappa} \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})

    where the vector

    .. math::

        \boldsymbol{\kappa} = -\frac{2 \pi m}{d} \hat{\boldsymbol{\kappa}},

    :math:`m` is the diffraction order,
    :math:`d` is the groove spacing,
    and :math:`\hat{\boldsymbol{\kappa}}` is a unit vector normal to the
    planes of the rulings.

    With the inclusion of Equation :eq:`phase-shift`, Equation :eq:`boundary-condition` becomes:

    .. math::
        :label: boundary-condition-shifted

         A_1 \exp\left[i (\mathbf{k}_1 + \boldsymbol{\kappa}) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
        = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]

    For Equation :eq:`boundary-condition-shifted` to be true everywhere in
    the :math:`x`-:math:`y` plane, the exponents must be equal:

    .. math::
        :label: boundary-condition-exponents

        (\mathbf{k}_1 + \boldsymbol{\kappa}) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
        = \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

    If we take :math:`\mathbf{k}_i = k_i \hat{\mathbf{k}}_i = n_i k_0 \hat{\mathbf{k}}_i` for :math:`i=(1,2)`,
    where :math:`k_i` is the incident/output wavenumber,
    :math:`n_i` is the incident/output index of refraction,
    :math:`k_0` is the wavenumber in vacuum,
    and :math:`\hat{\mathbf{k}}_i` is the incident/output propagation direction,
    we get an expression in terms of the output direction, :math:`\hat{\mathbf{k}}_2`:

    .. math::
        :label: boundary-directions

        n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
        = n_2 \hat{\mathbf{k}}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

    Now, Equation :eq:`boundary-directions` can only be satisfied
    if the components are separately equal since if :math:`y=0`

    .. math::
        :label: k_x

        n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot \hat{\mathbf{x}}
        = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{x}},

    and if :math:`x=0`

    .. math::
        :label: k_y

        n_1 (\hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1) \cdot \hat{\mathbf{y}}
        = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{y}}.

    So, if we define an effective incident propagation direction

    .. math::

        \mathbf{k}_\text{e} = \hat{\mathbf{k}}_1 + \boldsymbol{\kappa} / k_1,

    we can collect Equations :eq:`k_x` and :eq:`k_y` into a single vector
    equation

    .. math::
        :label: snells-law

        n_1 [ \mathbf{k}_\text{e} - ( \mathbf{k}_\text{e} \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ]
        = n_2 [ \hat{\mathbf{k}}_2 - ( \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ],

    where :math:`\hat{\mathbf{n}} = \hat{\mathbf{z}}` is the vector normal to
    the interface.

    Our goal now is to solve Equation :eq:`snells-law` for
    the output propagation direction, :math:`\hat{\mathbf{k}}_2`,
    and the only unknown is the component of the output propagation direction
    parallel to the surface normal vector, :math:`(\hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} )`.
    We can write this in terms of a cross product,

    .. math::
        :label: k2_dot_n

        \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} = \pm \sqrt{1 - |\hat{\mathbf{k}}_2 \times \hat{\mathbf{n}}|^2},

    where the :math:`\pm` represents a transmission or reflection, and the
    cross product :math:`\mathbf{k}_2 \times \hat{\mathbf{n}}` can be found by crossing
    Equation :eq:`snells-law` with :math:`\hat{\mathbf{n}}`,

    .. math::
        :label: k2_cross_n

        \hat{\mathbf{k}}_2 \times \hat{\mathbf{n}}
        = \frac{n_1}{n_2} \mathbf{k}_\text{e} \times \hat{\mathbf{n}},

    which leads to Equation :eq:`k2_dot_n` becoming:

    .. math::
        :label: k2_dot_n_expanded

        \mathbf{k}_2 \cdot \hat{\mathbf{n}}
        &= \pm \sqrt{1 - \left( \frac{n_1}{n_2} \right)^2 |\mathbf{k}_\text{e} \times \hat{\mathbf{n}}|^2} \\
        &= \pm \frac{n_1}{n_2} \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}})^2 - k_\text{e}^2 }

    By plugging Equation :eq:`k2_dot_n_expanded` into Equation :eq:`snells-law`,
    and solving for :math:`\hat{\mathbf{k}}_2`, we achieve our goal, an expression for the output ray
    in terms of the input ray and other known quantities.

    .. math::

        \hat{\mathbf{k}}_2
        = \frac{n_1}{n_2} \left[ \mathbf{k}_\text{e}
        + \left(
            \left( -\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}} \right)
            \pm \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_\text{e} \cdot \hat{\mathbf{n}})^2 - k_\text{e}^2 }
        \right) \hat{\mathbf{n}} \right]
    """
    a = direction
    n1 = index_refraction
    n2 = index_refraction_new
    m = diffraction_order

    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    r = n1 / n2

    wavelength_new = wavelength / r

    if np.any(m != 0):
        d = spacing_rulings
        g = normal_rulings
        a = np.where(
            condition=np.isfinite(d),
            x=a - np.sign(-a @ normal) * (m * wavelength_new * g) / (n1 * d),
            y=a,
        )

    c = -a @ normal

    mirror = np.sign(c) * (2 * is_mirror - 1)

    t = np.sqrt(np.square(1 / r) + np.square(c) - np.square(a.length))
    b = r * (a + (c + mirror * t) * normal)

    return b
