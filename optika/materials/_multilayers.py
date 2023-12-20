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

    The reflection and transmission of a plane wave from an ideal interface
    is described by the Fresnel equations :cite:p:`Born1980`.
    For :math:`s` polarization, the reflection and transmission coefficients are:

    .. math::
        :label: reflection-s

        r_{ij}^s = \frac{\tilde{n}_i \cos \theta_i - \tilde{n}_j \cos \theta_j}
                        {\tilde{n}_i \cos \theta_i + \tilde{n}_j \cos \theta_j}

    and

    .. math::
        :label: transmission-s

        t_{ij}^s = \frac{2 \tilde{n}_i \cos \theta_i}
                        {\tilde{n}_i \cos \theta_i + \tilde{n}_j \cos \theta_j}.

    For :math:`p` polarization, the reflection and transmission coefficients are:

    .. math::
        :label: reflection-p

        r_{ij}^p = \frac{\tilde{n}_i \cos \theta_j - \tilde{n}_j \cos \theta_i}
                        {\tilde{n}_i \cos \theta_j + \tilde{n}_j \cos \theta_i}

    and

    .. math::
        :label: transmission-p

        t_{ij}^p = \frac{2 \tilde{n}_i \cos \theta_i}
                        {\tilde{n}_i \cos \theta_j + \tilde{n}_j \cos \theta_i},

    where :math:`\tilde{n}_i = n_i + i k_i` is the complex index of refraction
    inside material :math:`i` and :math:`\theta_i` is the angle from the surface
    normal inside material :math:`i`.

    In the case of nonabrupt interfaces, we can use the method of :cite:t:`Stearns1989`
    to model the resulting loss of reflectivity.
    In this method, for a given average interface profile function, :math:`p(z)`,
    the loss of reflectivity is given by the Fourier transform of the derivative
    of the interface profile function,

    .. math::
        :label: interface-reflectivity

        r_{ij}' = \mathcal{F} \left\{ \frac{dp}{dz} \right\} r_{ij},

    for both :math:`s` and :math:`p` polarizations.
    All the interface profile functions described in :cite:t:`Windt1998`
    are implemented in the subpackage :mod:`optika.materials.profiles`.

    For a plane wave incident on a multilayer stack, the net reflection and
    transmission coefficients are given by :cite:t:`Born1980` as the
    following recursive relations:

    .. math::
        :label: net-reflection

        r_i^q = \frac{r_{ij}^q + r_j^q e^{2 i \beta_i}}
                   {1 + r_{ij}^q r_j^q e^{2 i \beta_i}}

    and

    .. math::
        :label: net-transmission

        t_i^q = \frac{t_{ij}^q t_j^q e^{i \beta_i}}
                   {1 + r_{ij}^q r_j^q e^{2 i \beta_i}},

    where :math:`q=(s, p)`,

    .. math::
        :label: beta

        \beta_i = 2 \pi d_i \tilde{n}_i \cos \theta_i / \lambda,

    :math:`d_i` is the thickness of material :math:`i`,
    and :math:`\lambda` is the wavelength of the incident light.

    Equations :eq:`net-reflection` and :eq:`net-transmission` are computed
    recursively starting at the bottom of the multilayer stack.
    The total reflectivity and transmissivity of the multilayer
    for each polarization direction is then given by

    .. math::
        :label: reflectivity_q

        R^q = |r_0^q|^2

    and

    .. math::
        :label: transmissivity_q

        T^q = \Re \left\{ \frac{\tilde{n}_s \cos \theta_s}{\tilde{n}_a \cos \theta_a} \right\} |t_0^q|^2,

    where the subscripts :math:`s` and :math:`a` denote the substrate and ambient materials respectively.

    From Equations :eq:`reflectivity_q` and :eq:`transmissivity_q` we can finally
    compute the average reflectivity and transmissivity of the multilayer stack

    .. math::
        :label: reflectivity

        R = \frac{R^s + R^p}{2}

    and

    .. math::
        :label: transmissivity

        T = \frac{T^s + T^p}{2}.

    The :class:`tuple` :math:`(R, T)` is the quantity returned by this function.
    """
    shape_layers = na.shape_broadcasted(material_layers, thickness_layers)
    material = material_layers
    thickness = thickness_layers.broadcast_to(shape_layers)
    axis = axis_layers

    wavelength = wavelength_ambient

    if normal is None:
        normal = na.Cartesian3dVectorArray(0, 0, -1)

    direction_substrate = snells_law(
        wavelength=wavelength,
        direction=direction_ambient,
        index_refraction=np.real(n_ambient),
        index_refraction_new=np.real(n_substrate),
        normal=normal,
    )

    direction_j = direction_substrate
    n_j = n_substrate
    rs_j = 0
    rp_j = 0
    ts_j = 1
    tp_j = 1

    n_cache = dict()

    for _i in range(material.shape[axis] + 1):
        i = ~_i

        if i == ~material.shape[axis]:
            n_i = n_ambient
            thickness_i = 0 * u.AA
        else:
            formula_i = material[{axis: i}].ndarray
            thickness_i = thickness[{axis: i}]

            if formula_i in n_cache:
                n_i = n_cache[formula_i]
            else:
                chemical_i = optika.chemicals.Chemical(
                    formula=formula_i,
                )

                index_refaction_i = chemical_i.index_refraction
                index_refaction_i = na.interp(
                    x=wavelength,
                    xp=index_refaction_i.inputs,
                    fp=index_refaction_i.outputs,
                    axis="wavelength",
                )

                wavenumber_i = chemical_i.wavenumber
                wavenumber_i = na.interp(
                    x=wavelength,
                    xp=wavenumber_i.inputs,
                    fp=wavenumber_i.outputs,
                    axis="wavelength",
                )

                n_i = index_refaction_i + wavenumber_i * 1j
                n_cache[formula_i] = n_i

        direction_i = snells_law(
            wavelength=wavelength,
            direction=direction_j,
            index_refraction=np.real(n_j),
            index_refraction_new=np.real(n_i),
            normal=normal,
        )

        kn_i = -direction_i @ normal
        kn_j = -direction_j @ normal

        es_ij_denom = n_i * kn_i + n_j * kn_j
        ep_ij_denom = n_i * kn_j + n_j * kn_i

        rs_ij = (n_i * kn_i - n_j * kn_j) / es_ij_denom
        rp_ij = (n_i * kn_j - n_j * kn_i) / ep_ij_denom

        if profile_interface is not None:
            w_tilde = profile_interface.reflectivity(
                wavelength=wavelength,
                direction=direction_i,
                normal=normal,
            )
            rs_ij = rs_ij * w_tilde
            rp_ij = rp_ij * w_tilde

        t_ij_numerator = 2 * n_i * kn_i
        ts_ij = t_ij_numerator / es_ij_denom
        tp_ij = t_ij_numerator / ep_ij_denom

        beta_i = 2 * np.pi * thickness_i * n_i * kn_i / wavelength

        exp_2i_beta_i = np.exp(2j * beta_i)

        es_i_denom = 1 + rs_ij * rs_j * exp_2i_beta_i
        ep_i_denom = 1 + rp_ij * rp_j * exp_2i_beta_i

        rs_i = (rs_ij + rs_j * exp_2i_beta_i) / es_i_denom
        rp_i = (rp_ij + rp_j * exp_2i_beta_i) / ep_i_denom

        ts_i = ts_ij * ts_j * np.exp(1j * beta_i) / es_i_denom
        tp_i = tp_ij * tp_j * np.exp(1j * beta_i) / ep_i_denom

        direction_j = direction_i
        n_j = n_i
        rs_j = rs_i
        rp_j = rp_i
        ts_j = ts_i
        tp_j = tp_i

    rs = rs_j
    rp = rp_j

    ts = ts_j
    tp = tp_j

    Rs = np.square(np.abs(rs))
    Rp = np.square(np.abs(rp))

    f_substrate = n_substrate * (-direction_substrate @ normal)
    f_ambient = n_ambient * (-direction_ambient @ normal)
    T_ratio = np.real(f_substrate / f_ambient)

    Ts = T_ratio * np.square(np.abs(ts))
    Tp = T_ratio * np.square(np.abs(tp))

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
