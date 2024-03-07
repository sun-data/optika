from typing import Sequence, Literal
import pytest
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.materials import AbstractLayer, Layer
from . import test_materials

_wavelength = na.linspace(100, 200, axis="wavelength", num=4) * u.AA


@pytest.mark.parametrize("wavelength", [_wavelength])
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        na.Cartesian3dVectorArray(0, 0, 1),
        na.Cartesian3dVectorArray(
            x=np.sin(na.linspace(-1, 1, axis="angle", num=5)),
            y=0,
            z=np.cos(na.linspace(-1, 1, axis="angle", num=5)),
        ),
    ],
)
@pytest.mark.parametrize("polarization", [None, "s"])
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
@pytest.mark.parametrize(
    argnames="normal",
    argvalues=[
        None,
    ],
)
def test_multilayer_efficiency(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: None | na.AbstractCartesian3dVectorArray,
    polarization: None | Literal["s", "p"],
    n: float | na.AbstractScalar,
    layers: Sequence[AbstractLayer] | AbstractLayer,
    substrate: None | Layer,
    normal: None | na.AbstractCartesian3dVectorArray,
):
    reflected, transmitted = optika.materials.multilayer_efficiency(
        wavelength=wavelength,
        direction=direction,
        polarization=polarization,
        n=n,
        layers=layers,
        substrate=substrate,
        normal=normal,
    )

    assert np.all(reflected >= 0)
    assert np.all(reflected <= 1)
    assert np.all(transmitted >= 0)
    assert np.all(transmitted <= 1)
    assert np.all(reflected + transmitted <= 1)
    assert np.all(np.imag(reflected) == 0)
    assert np.all(np.imag(transmitted) == 0)


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
            None,
            1,
            None,
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiO2.txt",
            None,
            1,
            Layer("SiO2", thickness=50 * u.AA),
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiO2_100A.txt",
            None,
            1,
            Layer("SiO2", thickness=100 * u.AA),
            Layer("Si"),
            False,
        ),
        (
            pathlib.Path(__file__).parent / "_data/SiC_Cr.txt",
            None,
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
            None,
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
    direction: None | na.AbstractCartesian3dVectorArray,
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
        polarization=None,
        n=n,
        layers=layers,
        substrate=substrate,
        normal=None,
    )

    if is_mirror:
        efficiency = reflectivity
    else:
        efficiency = transmissivity

    assert np.allclose(efficiency, efficiency_file, rtol=1e-4)


class AbstractTestAbstractMultilayerMaterial(
    test_materials.AbstractTestAbstractMaterial,
):
    def test_layers(self, a: optika.materials.AbstractMultilayerMaterial):
        result = a.layers
        if not isinstance(result, optika.materials.AbstractLayer):
            for layer in result:
                assert isinstance(layer, optika.materials.AbstractLayer)


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
            substrate=optika.materials.Layer("SiO2"),
        ),
    ],
)
class TestMultilayerMirror(
    AbstractTestAbstractMultilayerMirror,
):
    pass
