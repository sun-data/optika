from __future__ import annotations
from typing import Sequence
import abc
import dataclasses
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika
from . import (
    snells_law_scalar,
    matrices,
    AbstractMaterial,
    AbstractMirror,
    AbstractLayer,
    Layer,
    LayerSequence,
)

__all__ = [
    "multilayer_coefficients",
    "multilayer_efficiency",
    "layer_absorbance",
    "AbstractMultilayerMaterial",
    "AbstractMultilayerFilm",
    "MultilayerFilm",
    "AbstractMultilayerMirror",
    "MultilayerMirror",
]


def multilayer_coefficients(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: float | na.AbstractScalar = 1,
    layers: Sequence[AbstractLayer] | optika.materials.AbstractLayer = None,
    substrate: None | Layer = None,
) -> tuple[
    optika.vectors.PolarizationVectorArray,
    optika.vectors.PolarizationVectorArray,
]:
    r"""
    Calculate the reflection and transmission coefficients of a multilayer
    stack using the method described in :cite:t:`Yeh1988`.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The component of the incident light's propagation direction in the
        ambient medium antiparallel to the surface normal.
        Default is to assume normal incidence.
    n
        The complex index of refraction of the ambient medium.
    layers
        A sequence of layers representing the multilayer stack.
        If :obj:`None`, then this function computes the reflectivity and
        transmissivity of the ambient medium and the substrate.
    substrate
        A layer representing the substrate supporting the multilayer stack.
        The thickness of this layer is ignored.
        If :obj:`None`, then the substrate is assumed to be a vacuum.

    Notes
    -----

    The reflection and transmission coefficients of the multilayer stack can
    be calculated using the system transfer matrix method described in
    :cite:t:`Yeh1988`.

    The system transfer matrix is calculated using the transfer matrices of each
    layer, where each consists of two parts:
    the refractive matrix and propagation matrix.

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

    The propagation matrix takes the form

    .. math::
        :label: propagation-matrix

        U_{kj} = \begin{pmatrix}
                    e^{-i \beta_j} & 0 \\
                    0 & e^{i \beta_j} \\
                \end{pmatrix}

    where

    .. math::

        \beta_j = \frac{2 \pi}{\lambda} n_j h_j \cos \theta_j,

    is the phase change from propagating through material :math:`j`,
    :math:`\lambda` is the vacuum wavelength of the incident light,
    and :math:`h_j` is the thickness of material :math:`j`.

    To compute the system transfer matrix, we find the matrix product of the
    :math:`S=N+1` refractive matrices from each interface and the :math:`N`
    propagation matrices from each layer

    .. math::
        :label: sytem-transfer-matrix

        M_k = \left( \prod_{j=1}^N W_{kij} U_{kj} \right) W_{kNS}

    where :math:`i=j-1`.

    Once the system transfer matrix has been calculated, we can use
    :cite:t:`Yeh1988` Equation 5.2-3 to compute the system reflection coefficient

    .. math::
        :label: system-fresnel-reflection

        r_k = \frac{M_{k21}}{M_{k11}},

    and Equation 5.2-4 to compute the system transmission coefficient

    .. math::
        :label: system-fresnel-transmission

        t_k = \frac{1}{M_{k11}}.

    If we define the vectors

    .. math::

        \vec{r} = \begin{pmatrix}
            r_s \\
            r_p
        \end{pmatrix}

    and

    .. math::

        \vec{t} = \begin{pmatrix}
            t_s \\
            t_p
        \end{pmatrix},

    then the :class:`tuple` :math:`(\vec{r}, \vec{t})` is the quantity returned
    by this function.
    """
    direction_ambient = direction
    n_ambient = n

    if substrate is not None:
        substrate = dataclasses.replace(substrate, thickness=0 * u.nm)
    else:
        substrate = optika.materials.Layer(thickness=0 * u.nm)

    polarized_s = na.ScalarArray(
        ndarray=np.array([True, False]),
        axes="_polarization",
    )

    if layers is None:
        layers = []
    if not isinstance(layers, optika.materials.AbstractLayer):
        layers = optika.materials.LayerSequence(layers)

    n, direction, m_layers, where = layers.transfer(
        wavelength=wavelength,
        direction=direction_ambient,
        polarized_s=polarized_s,
        n=n_ambient,
    )
    n, direction, m_substrate, where = substrate.transfer(
        wavelength=wavelength,
        direction=direction,
        polarized_s=polarized_s,
        n=n,
        where=where,
    )
    m = m_layers @ m_substrate

    r = m.y.x / m.x.x
    t = 1 / m.x.x
    t[~where] = 0

    index_s = dict(_polarization=0)
    index_p = dict(_polarization=1)

    r = optika.vectors.PolarizationVectorArray(
        s=r[index_s],
        p=r[index_p],
    )

    t = optika.vectors.PolarizationVectorArray(
        s=t[index_s],
        p=t[index_p],
    )

    return r, t


def multilayer_efficiency(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: float | na.AbstractScalar = 1,
    layers: None | Sequence[AbstractLayer] | optika.materials.AbstractLayer = None,
    substrate: None | Layer = None,
) -> tuple[
    optika.vectors.PolarizationVectorArray,
    optika.vectors.PolarizationVectorArray,
]:
    r"""
    Calculate the reflectivity and transmissivity of a multilayer
    film or coating using the method in :cite:t:`Windt1998`.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The component of the incident light's propagation direction in the
        ambient medium antiparallel to the surface normal.
        Default is to assume normal incidence.
    n
        The complex index of refraction of the ambient medium.
    layers
        A sequence of layers representing the multilayer stack.
        If :obj:`None`, then this function computes the reflectivity and
        transmissivity of the ambient medium and the substrate.
    substrate
        A layer representing the substrate supporting the multilayer stack.
        The thickness of this layer is ignored.
        If :obj:`None`, then the substrate is assumed to be a vacuum.

    Examples
    --------

    Reproduce Example 2.3.1 in the
    `IMD User's Manual <http://www.rxollc.com/idl/IMD.pdf>`_,
    the transmittance of a :math:`\text{Zr}` filter.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define the wavelength of the incident light
        wavelength = na.linspace(100, 150, axis="wavelength", num=501) * u.AA

        # Define the Zr layer
        layers = optika.materials.Layer(
            chemical="Zr",
            thickness=1500 * u.AA,
        )

        # Compute the reflectivity and the transmissivity of this multilayer
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            wavelength=wavelength,
            layers=layers,
        )

        # Plot the transmissivity as a function of wavelength.
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            transmissivity.average,
            ax=ax,
            axis="wavelength",
            label="Zr",
        );
        ax.legend();
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("transmissivity");

    Reproduce Example 2.3.2 in the
    `IMD User's Manual <http://www.rxollc.com/idl/IMD.pdf>`_,
    the reflectivity of a :math:`\text{Si/Mo}` multilayer stack with
    interfacial roughness.

    .. jupyter-execute::

        # Define the wavelength of the incident light
        wavelength = na.linspace(100, 150, axis="wavelength", num=501) * u.AA

        # Period length of the multilayer sequence
        d = 66.5 * u.AA

        # Define the thickness to period ratios for each layer
        thickness_ratio = 0.6

        # Define the interface profile between successive layers
        interface = optika.materials.profiles.ErfInterfaceProfile(7 * u.AA)

        # Define the multilayer sequence
        layers = optika.materials.PeriodicLayerSequence(
            [
                optika.materials.Layer(
                    chemical="Si",
                    thickness=thickness_ratio * d,
                    interface=interface,
                ),
                optika.materials.Layer(
                    chemical="Mo",
                    thickness=(1 - thickness_ratio) * d,
                    interface=interface,
                ),
            ],
            num_periods=60,
        )

        # Compute the reflectivity and transmissivity of this multilayer stack
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            wavelength=wavelength,
            layers=layers,
        )

        # Plot the reflectivity as a function of wavelength
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            reflectivity.average,
            ax=ax,
            axis="wavelength",
            label=rf"Si/Mo $\times$ {layers.num_periods}",
        );
        ax.legend();
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");

    Reproduce Figure 9a in :cite:t:`Windt1998`, the reflectivity of a
    :math:`\text{Y/Al}` multilayer stack as a function of wavelength and of
    the ratio of the :math:`\text{Y}` thickness to the :math:`\text{Y + Al}`
    thickness, :math:`\Gamma`.

    .. jupyter-execute::

        # Period length of the multilayer sequence
        d = 98 * u.AA

        # wavelength of the incident light
        wavelength = na.linspace(170, 210, num=101, axis="wavelength") * u.AA

        # an array of thickness-to-period ratios for each layer
        thickness_ratio = na.linspace(0.2, 0.6, axis="thickness_ratio", num=5)

        # Define the multilayer sequence
        layers = optika.materials.PeriodicLayerSequence(
            [
                optika.materials.Layer(
                    chemical="Y",
                    thickness=thickness_ratio * d,
                ),
                optika.materials.Layer(
                    chemical="Al",
                    thickness=(1 - thickness_ratio) * d,
                )
            ],
            num_periods=40,
        )

        # Define the substrate layer
        substrate = optika.materials.Layer(
            chemical="Si",
        )

        # Compute the reflectivity and transmissivity of this multilayer stack
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            wavelength,
            layers=layers,
            substrate=substrate,
        )

        # Plot the reflectivity as a function of wavelength
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            reflectivity.average,
            ax=ax,
            axis="wavelength",
            label=r"$\Gamma=" + thickness_ratio.astype(str).astype(object) + "$",
        );
        ax.legend();
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");

    Notes
    -----

    The reflectivity of a multilayer stack can be calculated
    using Equation 5.2-5 in :cite:t:`Yeh1988`,

    .. math::
        :label: reflectivity

        R_k = |r_k|^2,

    and the transmissivity can be calculated using Equation 5.2-6,

    .. math::
        :label: transmissivity

        T_k = \text{Re} \left( \frac{q_{kS}}{q_{k0}} \right) |t_k|^2,

    where :math:`r_k` and :math:`t_k` are the system reflection and transmission
    coefficients calculated by :func:`multilayer_coefficients`,
    :math:`k = (s, p)` is the polarization state,

    .. math::

        q_{si} = n_i \cos \theta_i

    and

    .. math::

        q_{pi} = \frac{\cos \theta_i}{n_i}

    are the :math:`z` components of the wave's momentum for an arbitrary layer :math:`i`,
    :math:`n_i` is the index of refraction inside material :math:`i`,
    and :math:`\theta_i` is the angle between the wave's propagation direction
    and the vector normal to the interface inside material :math:`i`.

    If we define the vectors

    .. math::

        \vec{R} = \begin{pmatrix}
            R_s \\
            R_p
        \end{pmatrix}

    and

    .. math::

        \vec{T} = \begin{pmatrix}
            T_s \\
            T_p
        \end{pmatrix},

    then the :class:`tuple` :math:`(\vec{R}, \vec{T})` is the quantity returned
    by this function.
    """
    direction_ambient = direction
    n_ambient = n

    if substrate is not None:
        substrate = dataclasses.replace(substrate, thickness=0 * u.nm)
    else:
        substrate = optika.materials.Layer(thickness=0 * u.nm)

    r, t = multilayer_coefficients(
        wavelength=wavelength,
        direction=direction_ambient,
        n=n_ambient,
        layers=layers,
        substrate=substrate,
    )

    n_substrate = substrate.n(wavelength)
    direction_substrate = snells_law_scalar(
        cos_incidence=direction_ambient,
        index_refraction=n_ambient,
        index_refraction_new=n_substrate,
    )

    impedance_ambient = optika.vectors.PolarizationVectorArray(
        s=n_ambient,
        p=1 / n_ambient,
    )
    impedance_substrate = optika.vectors.PolarizationVectorArray(
        s=n_substrate,
        p=1 / n_substrate,
    )

    direction_ambient = optika.vectors.PolarizationVectorArray(
        s=direction_ambient,
        p=np.conj(direction_ambient),
    )
    direction_substrate = optika.vectors.PolarizationVectorArray(
        s=direction_substrate,
        p=np.conj(direction_substrate),
    )

    q_ambient = direction_ambient * impedance_ambient
    q_substrate = direction_substrate * impedance_substrate

    reflectivity = np.square(np.abs(r))
    transmissivity = np.square(np.abs(t)) * np.real(q_substrate / q_ambient)

    return reflectivity, transmissivity


def layer_absorbance(
    index: int,
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar = 1,
    n: float | na.AbstractScalar = 1,
    layers: None | Sequence[AbstractLayer] | optika.materials.AbstractLayer = None,
    substrate: None | Layer = None,
) -> optika.vectors.PolarizationVectorArray:
    """
    Compute the fraction of energy absorbed for a particular layer in the
    multilayer stack.

    Parameters
    ----------
    index
        The index of a :class:`AbstractLayer` in `layers` to find the absorbance of.
    wavelength
        The wavelength of the incident light in vacuum.
    direction
        The component of the incident light's propagation direction in the
        ambient medium antiparallel to the surface normal.
        Default is to assume normal incidence.
    n
        The complex index of refraction of the ambient medium.
    layers
        A sequence of layers representing the multilayer stack.
        If :obj:`None`, then this function computes the reflectivity and
        transmissivity of the ambient medium and the substrate.
    substrate
        A layer representing the substrate supporting the multilayer stack.
        The thickness of this layer is ignored.
        If :obj:`None`, then the substrate is assumed to be a vacuum.

    Examples
    --------

    Compute the amount of energy absorbed in a layer of silicon coated by a
    thin layer of silicon dioxide and compare it to the energy absorbed by
    both layers

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import optika

        # Define a grid of wavelengths at which to evaluate the absorbance
        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        # Define a thin layer of silicon dioxide and a thick layer of silicon
        layers = [
            optika.materials.Layer(
                chemical="SiO2",
                thickness=5 * u.nm,
            ),
            optika.materials.Layer(
                chemical="Si",
                thickness=16 * u.um,
            )
        ]

        # Compute the fraction of energy absorbed by the silicon
        absorbance = optika.materials.layer_absorbance(
            index=1,
            wavelength=wavelength,
            layers=layers,
        )

        # Compute the fraction of energy absorbed by both layers
        reflected, transmitted = optika.materials.multilayer_efficiency(
            wavelength=wavelength,
            layers=layers,
        )
        absorbance_total = 1 - reflected - transmitted

        # Plot the results
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            absorbance.average,
            ax=ax,
            label="absorbance",
        )
        na.plt.plot(
            wavelength,
            absorbance_total.average,
            ax=ax,
            label="total absorbance",
        )
        ax.set_xscale("log")
        ax.legend()
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("incident energy fraction");
    """
    if not isinstance(layers, AbstractLayer):
        layers = LayerSequence(layers)
    layers = layers.layer_sequence

    if substrate is not None:
        substrate = dataclasses.replace(substrate, thickness=0 * u.nm)
    else:
        substrate = optika.materials.Layer(thickness=0 * u.nm)

    r, t = multilayer_coefficients(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )
    r = na.stack([r.s, r.p], axis="_polarization")

    layer = layers[index]

    polarized_s = na.ScalarArray(
        ndarray=np.array([True, False]),
        axes="_polarization",
    )

    n_i, direction_i, m_i, where_i = layers[:index].transfer(
        wavelength=wavelength,
        direction=direction,
        polarized_s=polarized_s,
        n=n,
    )

    n_left = layer.n(wavelength)
    direction_left = snells_law_scalar(direction, n, n_left)
    m_left = m_i @ matrices.refraction(
        wavelength=wavelength,
        direction_left=direction_i,
        direction_right=direction_left,
        polarized_s=polarized_s,
        n_left=n_i,
        n_right=n_left,
        interface=layer.interface,
    )

    n_right, direction_right, m_right, where_right = layers[: index + 1].transfer(
        wavelength=wavelength,
        direction=direction,
        polarized_s=polarized_s,
        n=n,
    )

    amplitude_ambient = na.Cartesian2dVectorArray(1, r)

    a_right = m_right.inverse @ amplitude_ambient
    a_left = m_left.inverse @ amplitude_ambient

    a_right[~where_right] = 0

    a_right, b_right = a_right.x, a_right.y
    a_left, b_left = a_left.x, a_left.y

    angle_right = np.arccos(direction_right)
    angle_left = np.arccos(direction_left)

    rotation_right = na.Cartesian3dYRotationMatrixArray(angle_right)
    rotation_left = na.Cartesian3dYRotationMatrixArray(angle_left)

    antirotation_right = na.Cartesian3dYRotationMatrixArray(-angle_right)
    antirotation_left = na.Cartesian3dYRotationMatrixArray(-angle_left)

    a_right = a_right * np.where(
        polarized_s,
        na.Cartesian3dVectorArray(0, 1, 0),
        rotation_right @ na.Cartesian3dVectorArray(1, 0, 0),
    )
    a_left = a_left * np.where(
        polarized_s,
        na.Cartesian3dVectorArray(0, 1, 0),
        rotation_left @ na.Cartesian3dVectorArray(1, 0, 0),
    )

    b_right = b_right * np.where(
        polarized_s,
        na.Cartesian3dVectorArray(0, 1, 0),
        antirotation_right @ na.Cartesian3dVectorArray(1, 0, 0),
    )
    b_left = b_left * np.where(
        polarized_s,
        na.Cartesian3dVectorArray(0, 1, 0),
        antirotation_left @ na.Cartesian3dVectorArray(1, 0, 0),
    )

    impedance_ambient = np.where(
        polarized_s,
        n,
        1 / n,
    )
    impedance_right = np.where(
        polarized_s,
        n_right,
        1 / n_right,
    )
    impedance_left = np.where(
        polarized_s,
        n_left,
        1 / n_left,
    )

    ka_right = rotation_right @ na.Cartesian3dVectorArray(0, 0, 1)
    ka_left = rotation_left @ na.Cartesian3dVectorArray(0, 0, 1)
    kb_right = antirotation_right @ na.Cartesian3dVectorArray(0, 0, -1)
    kb_left = antirotation_left @ na.Cartesian3dVectorArray(0, 0, -1)

    ka_right = impedance_right * ka_right
    ka_left = impedance_left * ka_left
    kb_right = impedance_right * kb_right
    kb_left = impedance_left * kb_left

    c_right = ka_right.cross(a_right)
    c_left = ka_left.cross(a_left)

    d_right = kb_right.cross(b_right)
    d_left = kb_left.cross(b_left)

    flux_right = np.real((a_right + b_right).cross(np.conj(c_right + d_right)))
    flux_left = np.real((a_left + b_left).cross(np.conj(c_left + d_left)))

    flux_absorbed = (flux_left.z - flux_right.z) / (impedance_ambient * direction)

    index_s = dict(_polarization=0)
    index_p = dict(_polarization=1)

    flux_absorbed = optika.vectors.PolarizationVectorArray(
        s=flux_absorbed[index_s],
        p=flux_absorbed[index_p],
    )

    return flux_absorbed


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMultilayerMaterial(
    AbstractMaterial,
):
    @property
    @abc.abstractmethod
    def layers(self) -> AbstractLayer | Sequence[AbstractLayer]:
        """
        A sequence of layers representing the multilayer stack.
        """

    @property
    def _layers(self) -> AbstractLayer:
        result = self.layers
        if not isinstance(result, AbstractLayer):
            result = LayerSequence(result)
        return result

    @property
    def transformation(self) -> None:
        return None

    def index_refraction(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.index_refraction

    def attenuation(
        self,
        rays: optika.rays.RayVectorArray,
    ) -> na.ScalarLike:
        return rays.attenuation

    @abc.abstractmethod
    def plot_layers(
        self,
        width: u.Quantity = 100 * u.nm,
        ax: matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:
        """
        Plot the multilayer stack using :meth:`optika.materials.AbstractLayer.plot`.

        Parameters
        ----------
        width
            The width of the plotted multilayer stack in physical units.
        ax
            The matplotlib axes on which to plot the multilayer stack.
        kwargs
            Additional keyword arguments to pass along to
            :meth:`optika.materials.AbstractLayer.plot`.
        """


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMultilayerFilm(
    AbstractMultilayerMaterial,
):
    @property
    def is_mirror(self) -> bool:
        return False

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        Compute the efficiency of this multilayer film using
        :func:`optika.materials.multilayer_efficiency`.

        Parameters
        ----------
        rays
            the input rays with which to compute the efficiency of the multilayer film
        normal
            the vector normal to the interface between successive layers
        """
        wavelength = rays.wavelength
        k = rays.attenuation * wavelength / (4 * np.pi)
        n = rays.index_refraction + k * 1j

        reflectivity, transmissivity = multilayer_efficiency(
            wavelength=wavelength,
            direction=-rays.direction @ normal,
            n=n,
            layers=self.layers,
            substrate=None,
        )
        return transmissivity.average

    def plot_layers(
        self,
        width: u.Quantity = 100 * u.nm,
        ax: matplotlib.axes.Axes = None,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:
        return self._layers.plot(
            width=width,
            ax=ax,
            **kwargs,
        )


@dataclasses.dataclass(eq=False, repr=False)
class MultilayerFilm(
    AbstractMultilayerFilm,
):
    layers: None | AbstractLayer | Sequence[AbstractLayer] = None
    """A sequence of layers representing the multilayer stack."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            *[optika.shape(layer) for layer in self.layers],
        )


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMultilayerMirror(
    AbstractMultilayerMaterial,
    AbstractMirror,
):

    def efficiency(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        Compute the efficiency of this multilayer film using
        :func:`optika.materials.multilayer_efficiency`.

        Parameters
        ----------
        rays
            the input rays with which to compute the efficiency of the multilayer film
        normal
            the vector normal to the interface between successive layers
        """
        wavelength = rays.wavelength
        k = rays.attenuation * wavelength / (4 * np.pi)
        n = rays.index_refraction + k * 1j

        reflectivity, transmissivity = multilayer_efficiency(
            wavelength=wavelength,
            direction=-rays.direction @ normal,
            n=n,
            layers=self.layers,
            substrate=self.substrate,
        )
        return reflectivity.average

    def plot_layers(
        self,
        width: u.Quantity = 100 * u.nm,
        ax: matplotlib.axes.Axes = None,
        thickness_substrate: u.Quantity = 10 * u.nm,
        **kwargs,
    ) -> list[matplotlib.patches.Polygon]:

        if ax is None:
            ax = plt.gca()

        result = self._layers.plot(
            width=width,
            ax=ax,
            **kwargs,
        )

        substrate = self.substrate

        amplitude = 0.1 * thickness_substrate
        x_substrate = np.linspace(0, width, num=1001)
        y_substrate = amplitude * np.sin(2 * np.pi * x_substrate / width * u.rad)
        y_substrate = y_substrate - thickness_substrate

        kwargs_substrate = substrate.kwargs_plot
        if kwargs_substrate is None:
            kwargs_substrate = dict()

        ax.fill_between(
            x=x_substrate,
            y1=0,
            y2=y_substrate,
            **(kwargs_substrate | kwargs),
        )

        formula = substrate._chemical.formula_latex
        thickness = substrate.thickness

        ax.text(
            x=width / 2,
            y=-thickness_substrate / 2,
            s=rf"{formula} (${thickness.value:0.0f}\,${thickness.unit:latex_inline})",
            ha="center",
            va="center",
        )

        return result


@dataclasses.dataclass(eq=False, repr=False)
class MultilayerMirror(
    AbstractMultilayerMirror,
):
    r"""
    A model of a mirror coating consisting of alternating layers of different
    materials.

    Examples
    --------

    Reproduce Example 2.3.2 in the
    `IMD User's Manual <http://www.rxollc.com/idl/IMD.pdf>`_,
    the reflectivity of a :math:`\text{Si/Mo}` multilayer stack with
    interfacial roughness.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        # Period length of the multilayer sequence
        d = 66.5 * u.AA

        # Number of periods
        N = 60

        # Define the thickness to period ratios for each layer
        thickness_ratio = 0.6

        # Define the interface profile between successive layers
        interface = optika.materials.profiles.ErfInterfaceProfile(
            width=7 * u.AA,
        )

        # Define the periodic sequence of layers
        layers = optika.materials.PeriodicLayerSequence(
            [
                optika.materials.Layer(
                    chemical="Si",
                    thickness=thickness_ratio * d,
                    interface=interface,
                    kwargs_plot=dict(
                        color="tab:blue",
                        alpha=0.5,
                    ),
                ),
                optika.materials.Layer(
                    chemical="Mo",
                    thickness=(1- thickness_ratio) * d,
                    interface=interface,
                    kwargs_plot=dict(
                        color="tab:orange",
                        alpha=0.5,
                    ),
                ),
            ],
            num_periods=60,
        )

        # Define the substrate layer
        substrate = optika.materials.Layer(
            chemical="SiO2",
            thickness=10 * u.mm,
            interface=interface,
            kwargs_plot=dict(
                color="gray",
                alpha=0.5,
            ),
        )

        # Define a representation of multilayer coating
        multilayer = optika.materials.MultilayerMirror(
            layers=layers,
            substrate=substrate,
        )

        # Define the wavelengths of the incident light
        wavelength = na.linspace(100, 150, axis="wavelength", num=501) * u.AA

        # Define the rays incident on the multilayer coating
        rays = optika.rays.RayVectorArray(
            wavelength=wavelength,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )

        # Compute the reflectivity of this multilayer coating
        reflectivity = multilayer.efficiency(
            rays=rays,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity as a function of wavelength
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            reflectivity,
            ax=ax,
            axis="wavelength",
            label=rf"Si/Mo $\times$ {N}",
        );
        ax.legend();
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");

    Plot a visual representation of the multilayer coating

    .. jupyter-execute::

        fig, ax = plt.subplots(constrained_layout=True)
        ax.set_axis_off()
        with astropy.visualization.quantity_support():
            multilayer.plot_layers()
    """

    layers: None | AbstractLayer | Sequence[AbstractLayer] = None
    """A sequence of layers representing the multilayer stack."""

    substrate: None | Layer = None
    """A layer representing the substrate that the layers are deposited onto."""

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            *[optika.shape(layer) for layer in self.layers],
            optika.shape(self.substrate),
        )
