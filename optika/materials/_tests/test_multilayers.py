from typing import Callable, Sequence
import dataclasses
import pytest
import pathlib
import numpy as np
import matplotlib
import astropy.units as u
import astropy.visualization
import named_arrays as na
import optika
from optika.materials import AbstractLayer, Layer
from . import test_materials

_wavelength = na.linspace(100, 200, axis="wavelength", num=4) * u.AA


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        np.cos(na.linspace(-1, 1, axis="angle", num=5)),
    ],
)
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize(
    argnames="layers",
    argvalues=[
        Layer(
            chemical="SiC",
            thickness=10 * u.nm,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="substrate",
    argvalues=[
        None,
        Layer("SiO2"),
    ],
)
def test_multilayer_coefficients(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    layers: Sequence[AbstractLayer] | AbstractLayer,
    substrate: None | Layer,
):
    r, t = optika.materials.multilayer_coefficients(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )

    assert np.all(np.real(r) >= -1)
    assert np.all(np.real(r) <= 1)

    assert np.all(np.imag(r) >= -1)
    assert np.all(np.imag(r) <= 1)


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        np.cos(na.linspace(-1, 1, axis="angle", num=5)),
    ],
)
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize(
    argnames="layers",
    argvalues=[
        Layer(
            chemical="SiC",
            thickness=10 * u.nm,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="substrate",
    argvalues=[
        None,
        Layer("SiO2"),
    ],
)
def test_multilayer_efficiency(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    layers: Sequence[AbstractLayer] | AbstractLayer,
    substrate: None | Layer,
):
    reflected, transmitted = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )

    assert np.all(reflected >= 0)
    assert np.all(reflected <= 1)
    assert np.all(transmitted >= 0)
    assert np.all(transmitted <= 1)
    assert np.all(reflected + transmitted <= 1)
    assert np.all(np.imag(reflected) == 0)
    assert np.all(np.imag(transmitted) == 0)


@pytest.mark.parametrize("index", [0])
@pytest.mark.parametrize("wavelength", [100 * u.AA, _wavelength])
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        np.cos(na.linspace(-1, 1, axis="angle", num=5)),
    ],
)
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize(
    argnames="layers",
    argvalues=[
        Layer(
            chemical="SiC",
            thickness=10 * u.nm,
        ),
        [
            Layer(
                chemical="SiC",
                thickness=10 * u.nm,
            )
        ],
    ],
)
@pytest.mark.parametrize(
    argnames="substrate",
    argvalues=[
        None,
        Layer("SiO2"),
    ],
)
def test_layer_absorbance(
    index: int,
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    layers: None | Sequence[AbstractLayer] | optika.materials.AbstractLayer,
    substrate: None | Layer,
):
    result = optika.materials.layer_absorbance(
        index=index,
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )

    assert np.all(np.imag(result) == 0)
    assert np.all(result >= 0)
    assert np.all(result <= 1)

    reflectivity, transmissivity = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )

    result_expected = 1 - reflectivity - transmissivity

    assert np.allclose(result, result_expected)


@pytest.mark.parametrize("formula", ["Si", "Au", "SiO2"])
@pytest.mark.parametrize("angle", [0, 20, 45, 70, 85] * u.deg)
def test_multilayer_efficiency_fresnel(
    formula: str,
    angle: u.Quantity,
):
    """
    The reflectivity and transmissivity of a bare, absorbing substrate
    follow the Fresnel equations for both polarizations, at any angle
    of incidence.
    """
    wavelength = na.linspace(100, 500, axis="wavelength", num=5) * u.AA
    substrate = Layer(formula)

    reflectivity, transmissivity = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=np.cos(angle),
        substrate=substrate,
    )

    n = substrate.n(wavelength).ndarray
    cos_i = np.cos(angle).value
    n_cos_t = np.sqrt(np.square(n) - np.square(np.sin(angle).value))

    r_s = (cos_i - n_cos_t) / (cos_i + n_cos_t)
    t_s = 2 * cos_i / (cos_i + n_cos_t)
    r_p = (np.square(n) * cos_i - n_cos_t) / (np.square(n) * cos_i + n_cos_t)
    t_p = 2 * n * cos_i / (np.square(n) * cos_i + n_cos_t)

    flux_s = np.real(n_cos_t) / cos_i
    flux_p = np.real(np.conj(n) * n_cos_t / n) / cos_i

    assert np.allclose(reflectivity.s.ndarray, np.square(np.abs(r_s)))
    assert np.allclose(reflectivity.p.ndarray, np.square(np.abs(r_p)))
    assert np.allclose(transmissivity.s.ndarray, np.square(np.abs(t_s)) * flux_s)
    assert np.allclose(transmissivity.p.ndarray, np.square(np.abs(t_p)) * flux_p)


def _absorbance_field_integral(
    wavelength: float,
    angle: float,
    n: list[complex],
    thickness: list[float],
    n_substrate: complex,
    polarized_s: bool,
) -> np.ndarray:
    r"""
    The fraction of the incident power absorbed in each layer of a stack in
    vacuum, :math:`k_0 \int \text{Im}(\epsilon) |E|^2 dz / \cos \theta`,
    with the electric field found by matching the tangential fields at each
    interface, independently of :mod:`optika.materials`.

    Parameters
    ----------
    wavelength
        The vacuum wavelength of the incident light.
    angle
        The angle of incidence in radians.
    n
        The complex index of refraction of each layer, from the top down.
    thickness
        The thickness of each layer, in the same units as `wavelength`.
    n_substrate
        The complex index of refraction of the substrate.
    polarized_s
        If :obj:`True`, the light is :math:`s`-polarized,
        otherwise it is :math:`p`-polarized.
    """
    k0 = 2 * np.pi / wavelength
    media = np.array([1, *n, n_substrate], dtype=complex)
    d = np.array([0, *thickness, 0], dtype=float)
    kz = k0 * np.sqrt(np.square(media) - np.square(np.sin(angle)))
    cos = kz / (k0 * media)
    sin = np.sin(angle) / media

    def tangential(j: int) -> np.ndarray:
        """Map forward and backward amplitudes of E to the tangential E and H."""
        if polarized_s:
            return np.array([[1, 1], [kz[j], -kz[j]]])
        return np.array([[cos[j], cos[j]], [media[j], -media[j]]])

    amplitude = [np.array([1, 0], dtype=complex)]
    for j in reversed(range(len(media) - 1)):
        phase = np.diag(np.exp([1j * kz[j] * d[j], -1j * kz[j] * d[j]]))
        a = np.linalg.solve(tangential(j) @ phase, tangential(j + 1) @ amplitude[0])
        amplitude.insert(0, a)
    amplitude = [a / amplitude[0][0] for a in amplitude]

    result = []
    for j in range(1, len(media) - 1):
        z = np.linspace(0, d[j], num=4001)
        forward = amplitude[j][0] * np.exp(1j * kz[j] * z)
        backward = amplitude[j][1] * np.exp(-1j * kz[j] * z)
        if polarized_s:
            e2 = np.square(np.abs(forward + backward))
        else:
            e2 = np.square(np.abs(cos[j] * (forward + backward)))
            e2 = e2 + np.square(np.abs(sin[j] * (backward - forward)))
        absorbed = k0 * np.imag(np.square(media[j])) * np.trapezoid(e2, z)
        result.append(absorbed / np.cos(angle))
    return np.array(result)


@pytest.mark.parametrize("angle", [0, 45, 70] * u.deg)
def test_layer_absorbance_field_integral(
    angle: u.Quantity,
):
    """
    The absorbance of each layer of a stack of absorbing layers matches the
    power absorbed by the electric field inside it, for both polarizations.
    """
    wavelength = 300 * u.AA
    layers = [
        Layer("SiC", thickness=10 * u.nm),
        Layer("Cr", thickness=5 * u.nm),
        Layer("Sc", thickness=20 * u.nm),
    ]
    substrate = Layer("SiO2")

    for polarized_s, component in [(True, "s"), (False, "p")]:
        result = [
            getattr(
                optika.materials.layer_absorbance(
                    index=index,
                    wavelength=wavelength,
                    direction=np.cos(angle),
                    layers=layers,
                    substrate=substrate,
                ),
                component,
            )
            for index in range(len(layers))
        ]
        result = [na.as_named_array(r).ndarray for r in result]
        expected = _absorbance_field_integral(
            wavelength=wavelength.to_value(u.AA),
            angle=angle.to_value(u.rad),
            n=[complex(layer.n(wavelength).ndarray) for layer in layers],
            thickness=[layer.thickness.to_value(u.AA) for layer in layers],
            n_substrate=complex(substrate.n(wavelength).ndarray),
            polarized_s=polarized_s,
        )
        assert np.allclose(np.real(result), expected, atol=1e-6)


def _layer_absorbance_first(
    **kwargs,
) -> optika.vectors.PolarizationVectorArray:
    """
    The absorbance of the first layer of a stack.

    Parameters
    ----------
    kwargs
        The other arguments of :func:`optika.materials.layer_absorbance`.
    """
    return optika.materials.layer_absorbance(index=0, **kwargs)


_functions_direction = [
    optika.materials.multilayer_coefficients,
    optika.materials.multilayer_efficiency,
    _layer_absorbance_first,
]
"""The functions of a multilayer stack which take the angle of incidence."""


@pytest.mark.parametrize("function", _functions_direction)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        -0.5,
        -np.cos(45 * u.deg),
        np.cos(na.linspace(0, 100, axis="angle", num=5) * u.deg),
        na.NormalUncertainScalarArray(
            nominal=np.cos(na.linspace(0, 80, axis="angle", num=5) * u.deg),
            width=0.3,
            num_distribution=10,
            seed=42,
        ),
    ],
)
@pytest.mark.parametrize("num_interpolation", [None, 8])
def test_direction_behind(
    function: Callable[..., object],
    direction: float | na.AbstractScalar,
    num_interpolation: None | int,
):
    """
    Light which strikes a stack from behind, even only in some of the
    samples of an uncertain angle of incidence, raises an error instead of
    giving an efficiency greater than one.

    Parameters
    ----------
    function
        The function of the stack to evaluate.
    direction
        The cosine of the angle of incidence, negative somewhere.
    num_interpolation
        The number of nodes to interpolate the stack over, ignored by
        :func:`optika.materials.multilayer_coefficients`.
    """
    kwargs = dict(
        wavelength=_wavelength,
        direction=direction,
        layers=Layer(chemical="SiC", thickness=10 * u.nm),
        substrate=Layer("SiO2"),
    )
    if function is not optika.materials.multilayer_coefficients:
        kwargs["num_interpolation"] = num_interpolation
    with pytest.raises(ValueError, match="from behind"):
        function(**kwargs)


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("function", _functions_direction)
def test_direction_nan(
    function: Callable[..., object],
):
    """
    A ray which misses its surface carries a :obj:`~numpy.nan` direction,
    which passes through the check for light striking the stack from behind.

    Parameters
    ----------
    function
        The function of the stack to evaluate.
    """
    direction = na.ScalarArray(np.array([np.nan, 1]), axes="ray")
    result = function(
        wavelength=_wavelength,
        direction=direction,
        layers=Layer(chemical="SiC", thickness=10 * u.nm),
        substrate=Layer("SiO2"),
    )
    if not isinstance(result, tuple):
        result = (result,)
    for item in result:
        assert np.all(np.isfinite(item[dict(ray=1)]))


@pytest.mark.parametrize(
    argnames=[
        "file",
        "direction",
        "n",
        "layers",
        "substrate",
        "is_mirror",
    ],
    argvalues=[
        (
            pathlib.Path(__file__).parent / "_data/Si.txt",
            1,
            1,
            None,
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiO2.txt",
            1,
            1,
            Layer("SiO2", thickness=50 * u.AA),
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiO2_100A.txt",
            1,
            1,
            Layer("SiO2", thickness=100 * u.AA),
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiC_Cr.txt",
            1,
            1,
            [
                Layer("SiC", thickness=25 * u.nm),
                Layer("Cr", thickness=5 * u.nm),
            ],
            Layer("SiO2"),
            True,
        ),
        pytest.param(
            pathlib.Path(__file__).parent / "_data/SiC_Cr_Rough.txt",
            1,
            1,
            [
                Layer(
                    chemical="SiC",
                    thickness=25 * u.nm,
                    interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
                ),
                Layer(
                    chemical="Cr",
                    thickness=5 * u.nm,
                    interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
                ),
            ],
            Layer(
                chemical="SiO2",
                interface=optika.materials.profiles.ErfInterfaceProfile(2 * u.nm),
            ),
            True,
            marks=pytest.mark.xfail(
                reason="IMD incorrectly uses the vacuum wavelength"
            ),
        ),
    ],
)
def test_multilayer_efficiency_vs_file(
    file: pathlib.Path,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    layers: Sequence[AbstractLayer] | optika.materials.AbstractLayer,
    substrate: None | Layer,
    is_mirror: bool,
):
    skip_header = 0
    with open(file, "r") as f:
        for line in f:
            if line.startswith(";"):
                skip_header += 1
            else:
                break

    wavelength, efficiency_file = np.genfromtxt(
        fname=file,
        skip_header=skip_header,
        unpack=True,
    )
    wavelength = na.ScalarArray(wavelength, axes="wavelength") << u.AA
    efficiency_file = na.ScalarArray(efficiency_file, axes="wavelength")

    reflectivity, transmissivity = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=direction,
        n=n,
        layers=layers,
        substrate=substrate,
    )

    if is_mirror:
        efficiency = reflectivity.average
    else:
        efficiency = transmissivity.average

    assert np.allclose(efficiency, efficiency_file, rtol=1e-4)


_layers_interpolated = [
    optika.materials.Layer(chemical="SiO2", thickness=50 * u.AA),
]
_substrate_interpolated = optika.materials.Layer(chemical="Si", thickness=1 * u.um)
_direction_interpolated = np.cos(na.linspace(0, 10, axis="ray", num=101) * u.deg)


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize("num_interpolation", [None, 8, 64])
def test_multilayer_efficiency_interpolated(
    wavelength: u.Quantity | na.AbstractScalar,
    num_interpolation: None | int,
):
    """
    Interpolating the stack over the angle of incidence reproduces solving
    it for every direction, each polarization state on its own, and a single
    direction is solved directly.
    """
    kwargs = dict(
        wavelength=wavelength,
        n=1,
        layers=_layers_interpolated,
        substrate=_substrate_interpolated,
    )
    expected = optika.materials.multilayer_efficiency(
        direction=_direction_interpolated,
        **kwargs,
    )
    result = optika.materials.multilayer_efficiency(
        direction=_direction_interpolated,
        num_interpolation=num_interpolation,
        **kwargs,
    )
    for r, e in zip(result, expected):
        assert isinstance(r, optika.vectors.PolarizationVectorArray)
        assert na.shape(r) == na.shape(e)
        for component in ["s", "p"]:
            r_c, e_c = getattr(r, component), getattr(e, component)
            scale = np.abs(e_c).max()
            if num_interpolation is None:
                assert np.all(r_c == e_c)
            else:
                assert np.abs(r_c - e_c).max() < 1e-3 * scale

    # with nothing to interpolate over the stack is solved directly
    direct = optika.materials.multilayer_efficiency(
        direction=1,
        num_interpolation=num_interpolation,
        **kwargs,
    )
    exact = optika.materials.multilayer_efficiency(direction=1, **kwargs)
    for r, e in zip(direct, exact):
        assert np.all(r.s == e.s) and np.all(r.p == e.p)


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize("num_interpolation", [None, 8, 64])
def test_layer_absorbance_interpolated(
    wavelength: u.Quantity | na.AbstractScalar,
    num_interpolation: None | int,
):
    """
    The absorbance of a layer interpolates over the angle of incidence the
    same way the reflectivity and transmissivity of the stack do.
    """
    kwargs = dict(
        index=1,
        wavelength=wavelength,
        n=1,
        layers=_layers_interpolated + [_substrate_interpolated],
    )
    expected = optika.materials.layer_absorbance(
        direction=_direction_interpolated,
        **kwargs,
    )
    result = optika.materials.layer_absorbance(
        direction=_direction_interpolated,
        num_interpolation=num_interpolation,
        **kwargs,
    )
    assert isinstance(result, optika.vectors.PolarizationVectorArray)
    assert na.shape(result) == na.shape(expected)
    for component in ["s", "p"]:
        r_c, e_c = getattr(result, component), getattr(expected, component)
        if num_interpolation is None:
            assert np.all(r_c == e_c)
        else:
            assert np.abs(r_c - e_c).max() < 1e-3 * np.abs(e_c).max()


class AbstractTestAbstractMultilayerMaterial(
    test_materials.AbstractTestAbstractMaterial,
):
    def test_num_interpolation(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.num_interpolation
        assert (result is None) or (result > 0)

    def test_efficiency_interpolated(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
    ):
        """
        Interpolating the stack over the angle of incidence reproduces solving
        it for every ray, and does so better the more nodes it is given.
        """
        angle = na.linspace(0, 20, axis="angle", num=25) * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=_wavelength,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        expected = a.efficiency(rays, normal)

        # measured against the size of the response rather than pointwise,
        # since a multilayer stack can pass through a deep null where the
        # relative error of any approximation is unbounded but the absolute
        # error is negligible
        scale = np.abs(expected).max()

        def error(num: int) -> float:
            result = dataclasses.replace(a, num_interpolation=num).efficiency(
                rays=rays,
                normal=normal,
            )
            return np.abs(result - expected).max() / scale

        coarse, fine = error(8), error(64)

        assert coarse < 0.05
        assert fine <= coarse

    def test_efficiency_interpolated_uncertain(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
    ):
        """
        The interpolation works on rays whose direction carries an uncertainty
        distribution, which is how a system built with `num_distribution` set
        traces them.
        """
        angle = na.NormalUncertainScalarArray(
            na.linspace(0, 20, axis="angle", num=25) * u.deg,
            width=0.5 * u.deg,
            num_distribution=3,
        )
        rays = optika.rays.RayVectorArray(
            wavelength=_wavelength,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        expected = a.efficiency(rays, normal)
        result = dataclasses.replace(a, num_interpolation=64).efficiency(
            rays=rays,
            normal=normal,
        )

        assert np.abs(result - expected).max() < 0.05 * np.abs(expected).max()

    def test_efficiency_interpolated_scalar(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
    ):
        """With a single angle of incidence there is nothing to interpolate
        over, so the stack is solved directly."""
        rays = optika.rays.RayVectorArray(
            wavelength=200 * u.AA,
            direction=na.Cartesian3dVectorArray(0, 0, 1),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        result = dataclasses.replace(a, num_interpolation=8).efficiency(
            rays=rays,
            normal=normal,
        )

        assert np.all(result == a.efficiency(rays, normal))

    @pytest.mark.parametrize("angle_diffraction", [20 * u.deg, 55 * u.deg])
    def test_efficiency_grating(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
        angle_diffraction: u.Quantity,
    ):
        """
        The effective rays of a grating, whose component along the normal is
        that of the incident rays and whose component along the surface is
        that of the diffracted rays, see the stack at the angle whose cosine
        is the mean of the cosines of the angles of incidence and
        diffraction.
        """
        angle_incidence = 40 * u.deg
        normal = na.Cartesian3dVectorArray(0, 0, -1)
        rays = optika.rays.RayVectorArray(
            wavelength=_wavelength,
            direction=na.Cartesian3dVectorArray(
                x=np.sin(angle_diffraction),
                y=0,
                z=np.cos(angle_incidence),
            ),
        )

        cos_effective = (np.cos(angle_incidence) + np.cos(angle_diffraction)) / 2
        rays_effective = dataclasses.replace(
            rays,
            direction=na.Cartesian3dVectorArray(
                x=np.sqrt(1 - np.square(cos_effective)),
                y=0,
                z=cos_effective,
            ),
        )

        result = a.efficiency(rays, normal)
        expected = a.efficiency(rays_effective, normal)

        assert np.allclose(result, expected)

    def test_layers(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.layers
        if not isinstance(result, optika.materials.AbstractLayer):
            for layer in result:
                assert isinstance(layer, optika.materials.AbstractLayer)

    @pytest.mark.parametrize("ax", [None])
    def test_plot_layers(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
        ax: None | matplotlib.axes.Axes,
    ):
        with astropy.visualization.quantity_support():
            result = a.plot_layers(
                ax=ax,
            )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, matplotlib.patches.Rectangle)

    def test_efficiency_behind(
        self,
        a: optika.materials.AbstractMultilayerMaterial,
    ):
        """
        Light striking the stack from the side its normal points away from
        raises an error instead of giving an efficiency greater than one.
        """
        angle = 45 * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=_wavelength,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        normal = na.Cartesian3dVectorArray(0, 0, +1)
        with pytest.raises(ValueError, match="from behind"):
            a.efficiency(rays, normal)


class AbstractTestAbstractMultilayerFilm(
    AbstractTestAbstractMultilayerMaterial,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.MultilayerFilm(
            layers=optika.materials.LayerSequence(
                [
                    optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                    optika.materials.Layer("Al", thickness=100 * u.nm),
                    optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                ],
            ),
        ),
        optika.materials.MultilayerFilm(
            layers=optika.materials.PeriodicLayerSequence(
                [
                    optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                    optika.materials.Layer("Al", thickness=100 * u.nm),
                ],
                num_periods=2,
            ),
        ),
    ],
)
class TestMultilayerFilm(
    AbstractTestAbstractMultilayerFilm,
):
    pass


class AbstractTestAbstractMultilayerMirror(
    AbstractTestAbstractMultilayerMaterial,
):
    pass


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.MultilayerMirror(
            layers=[
                optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                optika.materials.Layer("Al", thickness=100 * u.nm),
                optika.materials.Layer("Al2O3", thickness=5 * u.nm),
            ],
            substrate=optika.materials.Layer("SiO2", thickness=10 * u.mm),
        ),
        optika.materials.MultilayerMirror(
            layers=optika.materials.PeriodicLayerSequence(
                [
                    optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                    optika.materials.Layer("Al", thickness=100 * u.nm),
                    optika.materials.Layer("Al2O3", thickness=5 * u.nm),
                ],
                num_periods=2,
            ),
            substrate=optika.materials.Layer("SiO2", thickness=10 * u.mm),
        ),
    ],
)
class TestMultilayerMirror(
    AbstractTestAbstractMultilayerMirror,
):
    pass
