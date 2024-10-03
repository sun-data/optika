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
def test_charge_diffusion(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.charge_diffusion(
        absorption=absorption,
        thickness_substrate=thickness_substrate,
        thickness_depletion=thickness_depletion,
    )

    assert result > 0 * u.um


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
