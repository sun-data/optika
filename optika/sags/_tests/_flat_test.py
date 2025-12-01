import pytest
import astropy.units as u
import named_arrays as na
import optika
from ._abc_test import AbstractTestAbstractSag


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sags.NoSag(
            transformation=na.transformations.Cartesian3dRotationX(20 * u.deg),
        ),
        optika.sags.NoSag(
            parameters_slope_error=optika.metrology.SlopeErrorParameters(
                kernel_size=2 * u.mm,
                step_size=4 * u.mm,
            ),
            parameters_roughness=optika.metrology.RoughnessParameters(
                period_min=2 * u.mm,
                period_max=4 * u.mm,
            ),
            parameters_microroughness=optika.metrology.RoughnessParameters(
                period_min=0.1 * u.mm,
                period_max=2 * u.mm,
            ),
        ),
    ],
)
class TestNoSag(
    AbstractTestAbstractSag,
):
    pass
