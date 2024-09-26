import pytest
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
    argnames="thickness_implant",
    argvalues=[
        100 * u.nm,
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
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.charge_diffusion(
        absorption=absorption,
        thickness_implant=thickness_implant,
        thickness_substrate=thickness_substrate,
        thickness_depletion=thickness_depletion,
    )

    assert result > 0 * u.um
