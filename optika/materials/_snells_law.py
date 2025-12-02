import math
import numpy as np
import numba as nb
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
    direction: na.AbstractCartesian3dVectorArray,
    index_refraction: float | na.AbstractScalar,
    index_refraction_new: float | na.AbstractScalar,
    normal: None | na.AbstractCartesian3dVectorArray = None,
    is_mirror: bool | na.AbstractScalar = False,
) -> na.Cartesian3dVectorArray:
    r"""
    A `vector form of Snell's law <https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form>`_.

    Parameters
    ----------
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
    refraction of light in a homogenous material.
    To start, consider an incident wave of the form:

    .. math::
        :label: incident-wave

        E_1(\mathbf{r}) = A_1 e^{i \mathbf{k}_1 \cdot \mathbf{r}},

    where :math:`E_1` is the magnitude of the incident electric field,
    :math:`A_1` is the amplitude of the incident wave,
    :math:`\mathbf{k}_1` is the incident wavevector,
    and :math:`\mathbf{r}` is a vector from the origin to the test point.
    Now we define an interface at :math:`z = 0`,
    where the index of refraction changes from :math:`n_1` to :math:`n_2`.
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
    At the :math:`z = 0` interface,
    :math:`E_1` and :math:`E_2` satisfy homogenous Dirichlet boundary conditions,

    .. math::
        :label: boundary-condition

        A_1 \exp\left[i \mathbf{k}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right]
        = A_2 \exp\left[i \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}) \right].

    For Equation :eq:`boundary-condition` to be true everywhere in
    the :math:`x`-:math:`y` plane, the exponents must be equal:

    .. math::
        :label: boundary-condition-exponents

        \mathbf{k}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
        = \mathbf{k}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

    If we take :math:`\mathbf{k}_i = k_i \hat{\mathbf{k}}_i = n_i k_0 \hat{\mathbf{k}}_i` for :math:`i=(1,2)`,
    where :math:`k_i` is the incident/output wavenumber,
    :math:`n_i` is the incident/output index of refraction,
    :math:`k_0` is the wavenumber in vacuum,
    and :math:`\hat{\mathbf{k}}_i` is the incident/output propagation direction,
    we get an expression in terms of the output direction, :math:`\hat{\mathbf{k}}_2`:

    .. math::
        :label: boundary-directions

        n_1 \hat{\mathbf{k}}_1 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}})
        = n_2 \hat{\mathbf{k}}_2 \cdot (x \hat{\mathbf{x}} + y \hat{\mathbf{y}}).

    Now, Equation :eq:`boundary-directions` can only be satisfied
    if the components are separately equal since if :math:`y=0`

    .. math::
        :label: k_x

        n_1 \hat{\mathbf{k}}_1 \cdot \hat{\mathbf{x}}
        = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{x}},

    and if :math:`x=0`

    .. math::
        :label: k_y

        n_1 \hat{\mathbf{k}}_1 \cdot \hat{\mathbf{y}}
        = n_2 \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{y}}.

    We can collect Equations :eq:`k_x` and :eq:`k_y` into a single vector
    equation,

    .. math::
        :label: snells-law

        n_1 [ \mathbf{k}_1 - ( \mathbf{k}_1 \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ]
        = n_2 [ \hat{\mathbf{k}}_2 - ( \hat{\mathbf{k}}_2 \cdot \hat{\mathbf{n}} ) \hat{\mathbf{n}} ],

    where :math:`\hat{\mathbf{n}} = \hat{\mathbf{z}}` is the vector normal to
    the interface.
    Now we can solve Equation :eq:`snells-law` for
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
        = \frac{n_1}{n_2} \mathbf{k}_1 \times \hat{\mathbf{n}},

    which leads to Equation :eq:`k2_dot_n` becoming:

    .. math::
        :label: k2_dot_n_expanded

        \mathbf{k}_2 \cdot \hat{\mathbf{n}}
        &= \pm \sqrt{1 - \left( \frac{n_1}{n_2} \right)^2 |\mathbf{k}_1 \times \hat{\mathbf{n}}|^2} \\
        &= \pm \frac{n_1}{n_2} \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_1 \cdot \hat{\mathbf{n}})^2 - k_1^2 }

    By plugging Equation :eq:`k2_dot_n_expanded` into Equation :eq:`snells-law`,
    and solving for :math:`\hat{\mathbf{k}}_2`, we achieve our goal, an expression for the output ray
    in terms of the input ray and other known quantities.

    .. math::

        \boxed{\hat{\mathbf{k}}_2
        = \frac{n_1}{n_2} \left[ \mathbf{k}_1
        + \left(
            \left( -\mathbf{k}_1 \cdot \hat{\mathbf{n}} \right)
            \pm \sqrt{\left( n_2 / n_1 \right)^2 + (\mathbf{k}_1 \cdot \hat{\mathbf{n}})^2 - k_1^2 }
        \right) \hat{\mathbf{n}} \right]}
    """
    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    direction = direction << u.dimensionless_unscaled
    index_refraction = index_refraction << u.dimensionless_unscaled
    index_refraction_new = index_refraction_new << u.dimensionless_unscaled
    normal = normal << u.dimensionless_unscaled

    b_x, b_y, b_z = _snells_law_numba(
        direction.x.value,
        direction.y.value,
        direction.z.value,
        index_refraction.value,
        index_refraction_new.value,
        normal.x.value,
        normal.y.value,
        normal.z.value,
        is_mirror,
    )

    return na.Cartesian3dVectorArray(b_x, b_y, b_z)


@nb.guvectorize(
    [
        "void(float64,float64,float64,float64,float64,float64,float64,float64,bool,float64[:],float64[:],float64[:])"
    ],
    "(),(),(),(),(),(),(),(),()->(),(),()",
    target="parallel",
    nopython=True,
    cache=True,
)
def _snells_law_numba(
    direction_x: float,
    direction_y: float,
    direction_z: float,
    index_refraction: float,
    index_refraction_new: float,
    normal_x: float,
    normal_y: float,
    normal_z: float,
    is_mirror: bool,
    result_x: np.ndarray,
    result_y: np.ndarray,
    result_z: np.ndarray,
):  # pragma: nocover
    """
    A :mod:`numba`-accelerated version of Snell's law.

    Parameters
    ----------
    direction_x
        The :math:`x` component of the propagation direction of the incident light.
    direction_y
        The :math:`y` component of the propagation direction of the incident light.
    direction_z
        The :math:`z` component of the propagation direction of the incident light.
    index_refraction
        The index of refraction of the current medium.
    index_refraction_new
        The index of refraction of the new medium.
    normal_x
        The :math:`x` component of the vector perpendicular to the interface.
    normal_y
        The :math:`y` component of the vector perpendicular to the interface.
    normal_z
        The :math:`z` component of the vector perpendicular to the interface.
    is_mirror
        Whether the incident light is reflected or not.
    """
    a_x = direction_x
    a_y = direction_y
    a_z = direction_z

    n1 = index_refraction
    n2 = index_refraction_new

    u_x = normal_x
    u_y = normal_y
    u_z = normal_z

    a2 = a_x * a_x + a_y * a_y + a_z * a_z

    r = n1 / n2
    r2 = r * r

    au = a_x * u_x + a_y * u_y + a_z * u_z
    au2 = au * au

    sgn = -math.copysign(1, au)

    d = -au + sgn * (2 * is_mirror - 1) * math.sqrt(1 / r2 + au2 - a2)

    result_x[:] = r * (a_x + d * u_x)
    result_y[:] = r * (a_y + d * u_y)
    result_z[:] = r * (a_z + d * u_z)
