from __future__ import annotations
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import profiles, snells_law, AbstractMaterial

__all__ = [
    "multilayer_efficiency",
    "AbstractMultilayerMaterial",
    "AbstractMultilayerFilm",
    "MultilayerFilm",
]


def multilayer_efficiency(
    material_layers: na.AbstractScalarArray,
    thickness_layers: na.AbstractScalar,
    axis_layers: str,
    wavelength_ambient: u.Quantity | na.AbstractScalar,
    direction_ambient: na.AbstractCartesian3dVectorArray,
    n_ambient: complex | na.AbstractScalar,
    n_substrate: complex | na.AbstractScalar,
    normal: None | na.AbstractCartesian3dVectorArray = None,
    profile_interface: None | profiles.AbstractInterfaceProfile = None,
) -> tuple[na.AbstractScalar, na.AbstractScalar]:
    r"""
    Calculate the reflectivity and transmissivity of a multilayer
    film or coating using the method in :cite:t:`Windt1998`.

    Parameters
    ----------
    material_layers
        An array of chemical formulas for each layer.
    thickness_layers
        An array of thicknesses for each layer.
    axis_layers
        The logical axis along which the different layers are distributed.
    wavelength_ambient
        The wavelength of the incident light in the ambient medium.
    direction_ambient
        The direction of the incident light in the ambient medium.
    n_ambient
        The index of refraction of the ambient medium.
    n_substrate
        The index of refraction of the substrate, at the bottom of the multilayer
        stack.
    normal
        A vector perpendicular to the interface between successive layers.
        If :obj:`None`, the normal vector is assumed to be :math:`-\hat{z}`
    profile_interface
        An optional profile for modeling the roughness and/or
        diffusiveness of the interface between successive layers.

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

        # Define an array of chemical formulae for each layer.
        material_layers = na.ScalarArray(np.array(["Zr"], dtype=object), axes="layer")

        # Define an array of thicknesses for each layer.
        thickness_layers = na.ScalarArray([1500] * u.AA, axes="layer")

        # Define the wavelengths of the incident light
        wavelength = na.linspace(100, 150, axis="wavelength", num=501) * u.AA

        # Compute the reflectivity and the transmissivity of this multilyaer
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            material_layers=material_layers,
            thickness_layers=thickness_layers,
            axis_layers="layer",
            wavelength_ambient=wavelength,
            direction_ambient=na.Cartesian3dVectorArray(0, 0, 1),
            n_ambient=1,
            n_substrate=1,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the transmissivity as a function of wavelength.
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            transmissivity,
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

        # Period length of the multilayer sequence
        d = 66.5 * u.AA

        # Number of periods
        N = 60

        # Define an array of chemical formulas for each layer
        material_layers = na.ScalarArray(
            ndarray=np.array(N * ["Si", "Mo"]),
            axes="layer"
        )

        # Define the thickness to period ratios for each layer
        thickness_ratio = 0.6

        # Compute thicknesses for each layer
        thickness_layers = d * na.stack(
            arrays=N * [thickness_ratio, 1 - thickness_ratio],
            axis="layer"
        )

        # Define the interface profile between successive layers
        profile_interface = optika.materials.profiles.ErfInterfaceProfile(
            width=7 * u.AA,
        )

        # Compute the reflectivity and transmissivity of this multilayer stack
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            material_layers=material_layers,
            thickness_layers=thickness_layers,
            axis_layers="layer",
            wavelength_ambient=wavelength,
            direction_ambient=na.Cartesian3dVectorArray(0, 0, 1),
            n_ambient=1,
            n_substrate=1,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
            profile_interface=profile_interface,
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

    Reproduce Figure 9a in :cite:t:`Windt1998`, the reflectivity of a
    :math:`\text{Y/Al}` multilayer stack as a function of wavelength and of
    the ratio of the :math:`\text{Y}` thickness to the :math:`\text{Y + Al}`
    thickness, :math:`\Gamma`.

    .. jupyter-execute::

        # Period length of the multilayer sequence
        d = 98 * u.AA

        # Number of periods
        N = 40

        # array of chemical formulas for each layer
        material_layers = na.ScalarArray(
            ndarray=np.array(N * ["Y", "Al"]),
            axes="layer"
        )

        # an array of thickness to period ratios for each layer
        thickness_ratio = na.linspace(0.2, 0.6, axis="thickness_ratio", num=5)

        # thicknesses for each layer
        thickness_layers = d * na.stack(
            arrays=N * [thickness_ratio, 1 - thickness_ratio],
            axis="layer"
        )

        # wavelength of the incident light
        wavelength = na.linspace(170, 210, num=101, axis="wavelength") * u.AA

        # Compute the complex index of refraction for the silicon substrate
        silicon = optika.chemicals.Chemical("Si")
        index_refraction_substrate = na.interp(
            x=wavelength,
            xp=silicon.index_refraction.inputs,
            fp=silicon.index_refraction.outputs,
        )
        wavenumber_substrate = na.interp(
            x=wavelength,
            xp=silicon.wavenumber.inputs,
            fp=silicon.wavenumber.outputs,
        )
        n_substrate = index_refraction_substrate + wavenumber_substrate * 1j

        # Compute the reflectivity and transmissivity of this multilayer stack
        reflectivity, transmissivity = optika.materials.multilayer_efficiency(
            material_layers=material_layers,
            thickness_layers=thickness_layers,
            axis_layers="layer",
            wavelength_ambient=wavelength,
            direction_ambient=na.Cartesian3dVectorArray(0, 0, 1),
            n_ambient=1,
            n_substrate=n_substrate,
            normal=na.Cartesian3dVectorArray(0, 0, -1),
        )

        # Plot the reflectivity as a function of wavelength
        fig, ax = plt.subplots()
        na.plt.plot(
            wavelength,
            reflectivity,
            ax=ax,
            axis="wavelength",
            label=r"$\Gamma=" + thickness_ratio.astype(str).astype(object) + "$",
        );
        ax.legend();
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("reflectivity");

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

    With the system reflection and transmission coefficients, we can compute the
    reflectivity and transmissivity for each polarization state using the
    expressions

    .. math::
        :label: reflectivity

        R_k = |r_k|^2,

    and

    .. math::
        :label: transmissivity

        T_k = \Re \left( \frac{p_{kS}}{p_{k0}} \right) |t_k|^2.

    From Equations :eq:`reflectivity` and :eq:`transmissivity` we can finally
    compute the average reflectivity and transmissivity of the multilayer stack

    .. math::
        :label: avg-reflectivity

        R = \frac{R_s + R_p}{2}

    and

    .. math::
        :label: avg-transmissivity

        T = \frac{T_s + T_p}{2}.

    The :class:`tuple` :math:`(R, T)` is the quantity returned by this function.
    """
    shape_layers = na.shape_broadcasted(material_layers, thickness_layers)
    material = material_layers
    thickness = thickness_layers.broadcast_to(shape_layers)
    axis = axis_layers

    wavelength = wavelength_ambient * np.real(n_ambient)

    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    cos_theta_ambient = -direction_ambient @ normal
    q_sa = cos_theta_ambient * n_ambient
    q_pa = cos_theta_ambient / n_ambient

    direction_i = direction_ambient
    n_i = n_ambient
    q_si = q_sa
    q_pi = q_pa

    n_cache = dict()

    m_s11 = m_p11 = 1
    m_s12 = m_p12 = 0
    m_s21 = m_p21 = 0
    m_s22 = m_p22 = 1

    num_layers = material.shape[axis]
    num_interfaces = num_layers + 1

    for j in range(num_interfaces):
        if j == num_layers:
            thickness_j = 0 * u.AA
            n_j = n_substrate
        else:
            formula_j = material[{axis: j}].ndarray
            thickness_j = thickness[{axis: j}]

            if formula_j in n_cache:
                n_j = n_cache[formula_j]
            else:
                chemical_j = optika.chemicals.Chemical(
                    formula=formula_j,
                )

                index_refaction_j = chemical_j.index_refraction
                index_refaction_j = na.interp(
                    x=wavelength,
                    xp=index_refaction_j.inputs,
                    fp=index_refaction_j.outputs,
                    axis="wavelength",
                )

                wavenumber_j = chemical_j.wavenumber
                wavenumber_j = na.interp(
                    x=wavelength,
                    xp=wavenumber_j.inputs,
                    fp=wavenumber_j.outputs,
                    axis="wavelength",
                )

                n_j = index_refaction_j + wavenumber_j * 1j
                n_cache[formula_j] = n_j

        direction_j = snells_law(
            wavelength=wavelength / np.real(n_i),
            direction=direction_i,
            index_refraction=np.real(n_i),
            index_refraction_new=np.real(n_j),
            normal=normal,
        )
        cos_theta_j = -direction_j @ normal

        q_sj = cos_theta_j * n_j
        q_pj = cos_theta_j / n_j

        a_sij = q_si + q_sj
        a_pij = q_pi + q_pj

        r_sij = (q_si - q_sj) / a_sij
        r_pij = (q_pi - q_pj) / a_pij

        t_sij = 2 * q_si / a_sij
        t_pij = 2 * q_pi / a_pij

        if profile_interface is not None:
            w_tilde = profile_interface.reflectivity(
                wavelength=wavelength,
                direction=direction_i,
                normal=normal,
            )
            r_sij = w_tilde * r_sij
            r_pij = w_tilde * r_pij

        beta_j = 2 * np.pi * thickness_j * n_j * cos_theta_j / wavelength

        exp_negi_beta_j = np.exp(-1j * beta_j)
        exp_posi_beta_j = np.exp(+1j * beta_j)

        m_sj11 = exp_negi_beta_j / t_sij
        m_pj11 = exp_negi_beta_j / t_pij

        m_sj12 = exp_posi_beta_j * r_sij / t_sij
        m_pj12 = exp_posi_beta_j * r_pij / t_pij

        m_sj21 = exp_negi_beta_j * r_sij / t_sij
        m_pj21 = exp_negi_beta_j * r_pij / t_pij

        m_sj22 = exp_posi_beta_j / t_sij
        m_pj22 = exp_posi_beta_j / t_pij

        m_s11_new = m_s11 * m_sj11 + m_s12 * m_sj21
        m_s12_new = m_s11 * m_sj12 + m_s12 * m_sj22
        m_s21_new = m_s21 * m_sj11 + m_s22 * m_sj21
        m_s22_new = m_s21 * m_sj12 + m_s22 * m_sj22

        m_p11_new = m_p11 * m_pj11 + m_p12 * m_pj21
        m_p12_new = m_p11 * m_pj12 + m_p12 * m_pj22
        m_p21_new = m_p21 * m_pj11 + m_p22 * m_pj21
        m_p22_new = m_p21 * m_pj12 + m_p22 * m_pj22

        m_s11 = m_s11_new
        m_s12 = m_s12_new
        m_s21 = m_s21_new
        m_s22 = m_s22_new

        m_p11 = m_p11_new
        m_p12 = m_p12_new
        m_p21 = m_p21_new
        m_p22 = m_p22_new

        direction_i = direction_j
        n_i = n_j
        q_si = q_sj
        q_pi = q_pj

    rs = m_s21 / m_s11
    rp = m_p21 / m_p11

    ts = 1 / m_s11
    tp = 1 / m_p11

    Rs = np.square(np.abs(rs))
    Rp = np.square(np.abs(rp))

    Ts = np.square(np.abs(ts)) * np.real(q_sj / q_sa)
    Tp = np.square(np.abs(tp)) * np.real(q_pj / q_pa)

    R = (Rs + Rp) / 2
    T = (Ts + Tp) / 2

    return R, T


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMultilayerMaterial(
    AbstractMaterial,
):
    @property
    @abc.abstractmethod
    def material_layers(self) -> na.AbstractScalarArray:
        """
        an array of strings describing the chemical formula of each layer
        """

    @property
    @abc.abstractmethod
    def thickness_layers(self) -> na.AbstractScalar:
        """
        an array of the thicknesses for each layer
        """

    @property
    @abc.abstractmethod
    def axis_layers(self) -> str:
        """
        the logical axis along which the different layers are distributed
        """

    @property
    @abc.abstractmethod
    def profile_interface(self) -> optika.materials.profiles.AbstractInterfaceProfile:
        """
        The interface profile between successive layers
        """

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


@dataclasses.dataclass(eq=False, repr=False)
class AbstractMultilayerFilm(
    AbstractMultilayerMaterial,
):
    @property
    def is_mirror(self) -> bool:
        return False

    def transmissivity(
        self,
        rays: optika.rays.RayVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ) -> na.ScalarLike:
        """
        Compute the transmissivity of this multilayer film using
        :func:`optika.materials.multilayer_efficiency`.

        Parameters
        ----------
        rays
            the input rays with which to compute the transmissivity of the multilayer film
        normal
            the vector normal to the interface between successive layers
        """
        wavelength = rays.wavelength
        k_ambient = rays.attenuation * wavelength / (4 * np.pi)
        n_ambient = rays.index_refraction + k_ambient * 1j
        reflectivity, transmissivity = multilayer_efficiency(
            material_layers=self.material_layers,
            thickness_layers=self.thickness_layers,
            axis_layers=self.axis_layers,
            wavelength_ambient=wavelength,
            direction_ambient=rays.direction,
            n_ambient=n_ambient,
            n_substrate=n_ambient,
            normal=normal,
            profile_interface=self.profile_interface,
        )
        return transmissivity


@dataclasses.dataclass(eq=False, repr=False)
class MultilayerFilm(
    AbstractMultilayerFilm,
):
    material_layers: na.AbstractScalarArray = dataclasses.MISSING
    """an array of strings describing the chemical formula of each layer"""

    thickness_layers: na.AbstractScalar = dataclasses.MISSING
    """an array of the thicknesses for each layer"""

    axis_layers: str = dataclasses.MISSING
    """the logical axis along which the different layers are distributed"""

    profile_interface: None | optika.materials.profiles.AbstractInterfaceProfile = None
    """The interface profile between successive layers"""
