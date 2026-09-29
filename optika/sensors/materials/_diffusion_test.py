import warnings
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize(
    argnames="absorption",
    argvalues=[
        1 / u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        15 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_depletion",
    argvalues=[
        5 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="width_max",
    argvalues=[
        None,
        4 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="width_depleted",
    argvalues=[
        None,
        0.8 * u.um,
    ],
)
def test_charge_diffusion(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
    width_max: None | u.Quantity | na.AbstractScalar,
    width_depleted: None | u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.charge_diffusion(
        absorption=absorption,
        thickness_substrate=thickness_substrate,
        thickness_depletion=thickness_depletion,
        width_max=width_max,
        width_depleted=width_depleted,
    )

    assert result > 0 * u.um


@pytest.mark.parametrize(
    argnames="absorption",
    argvalues=[
        na.geomspace(1e-4, 1e3, axis="absorption", num=8) / u.um,
    ],
)
def test_charge_diffusion_janesick(
    absorption: u.Quantity | na.AbstractScalar,
):
    """The defaults are the closed form of Janesick (2001)."""
    s = 14 * u.um
    d = 8.7 * u.um
    f = s - d
    a = absorption

    result = optika.sensors.charge_diffusion(
        absorption=a,
        thickness_substrate=s,
        thickness_depletion=d,
    )

    expected = np.sqrt(f * (a * f + np.exp(-a * f) - 1) / (a * (1 - np.exp(-a * s))))

    assert np.allclose(result, expected, rtol=1e-6)

    explicit = optika.sensors.charge_diffusion(
        absorption=a,
        thickness_substrate=s,
        thickness_depletion=d,
        width_max=f,
        width_depleted=0 * u.um,
    )

    assert np.allclose(explicit, result, rtol=1e-12)


@pytest.mark.parametrize(
    argnames="thickness_depletion",
    argvalues=[
        0 * u.um,
        14 * u.um,
    ],
)
def test_charge_diffusion_degenerate(
    thickness_depletion: u.Quantity | na.AbstractScalar,
):
    """
    A sensor with no depletion region or no field-free region gives finite
    widths, without dividing by the thickness of the missing region.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = optika.sensors.charge_diffusion(
            absorption=na.geomspace(1e-4, 1e3, axis="absorption", num=8) / u.um,
            thickness_substrate=14 * u.um,
            thickness_depletion=thickness_depletion,
            width_max=4 * u.um,
            width_depleted=0.8 * u.um,
        )
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    argnames="width_diffusion",
    argvalues=[
        10 * u.um,
        na.linspace(1, 10, "width", 5) * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="width_pixel",
    argvalues=[
        15 * u.um,
    ],
)
def test_mean_charge_capture(
    width_diffusion: u.Quantity | na.AbstractScalar,
    width_pixel: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.mean_charge_capture(
        width_diffusion=width_diffusion,
        width_pixel=width_pixel,
    )
    assert np.all(result > 0)
    assert np.all(result < 1)


@pytest.mark.parametrize(
    argnames="width_diffusion",
    argvalues=[
        10 * u.um,
        na.linspace(1, 10, "width", 5) * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="width_pixel",
    argvalues=[
        15 * u.um,
    ],
)
def test_kernel_diffusion(
    width_diffusion: u.Quantity | na.AbstractScalar,
    width_pixel: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.kernel_diffusion(
        width_diffusion=width_diffusion,
        width_pixel=width_pixel,
        axis_x="x",
        axis_y="y",
    )

    assert isinstance(result, na.FunctionArray)
    assert isinstance(result.outputs, na.AbstractScalar)
    assert isinstance(result.inputs, na.Cartesian2dVectorArray)
    assert np.all(result.outputs.sum(("x", "y")) == 1)
