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
    argnames="absorption",
    argvalues=[
        0.01 / u.um,
        0.3 / u.um,
        10 / u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_depletion",
    argvalues=[
        0 * u.um,
        8.7 * u.um,
        14 * u.um,
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
def test_charge_diffusion_average(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_depletion: u.Quantity | na.AbstractScalar,
    width_max: None | u.Quantity | na.AbstractScalar,
    width_depleted: None | u.Quantity | na.AbstractScalar,
):
    """
    The closed form is the average of the profile over the absorption depth,
    including a sensor with no depletion region and one with no field-free
    region.
    """
    s = 14 * u.um
    d = thickness_depletion

    result = optika.sensors.charge_diffusion(
        absorption=absorption,
        thickness_substrate=s,
        thickness_depletion=d,
        width_max=width_max,
        width_depleted=width_depleted,
    )

    axis = "depth"
    num = 100000
    depth = (na.arange(0, num, axis=axis) + 0.5) * s / num
    profile = optika.sensors.charge_diffusion_profile(
        depth=depth,
        thickness_substrate=s,
        thickness_depletion=d,
        width_max=width_max,
        width_depleted=width_depleted,
    )
    weight = np.exp(-absorption * depth)
    variance = (np.square(profile) * weight).sum(axis) / weight.sum(axis)

    assert np.allclose(result, np.sqrt(variance), rtol=1e-5, atol=1e-9 * u.um)


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
    kwargs = dict(
        thickness_substrate=14 * u.um,
        thickness_depletion=thickness_depletion,
        width_max=4 * u.um,
        width_depleted=0.8 * u.um,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        average = optika.sensors.charge_diffusion(
            absorption=na.geomspace(1e-4, 1e3, axis="absorption", num=8) / u.um,
            **kwargs,
        )
        profile = optika.sensors.charge_diffusion_profile(
            depth=na.linspace(0, 14, axis="depth", num=15) * u.um,
            **kwargs,
        )
    assert np.all(np.isfinite(average))
    assert np.all(np.isfinite(profile))


@pytest.mark.parametrize(
    argnames="depth",
    argvalues=[
        0 * u.um,
        na.linspace(0, 14, axis="depth", num=15) * u.um,
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
def test_charge_diffusion_profile(
    depth: u.Quantity | na.AbstractScalar,
    width_max: None | u.Quantity | na.AbstractScalar,
    width_depleted: None | u.Quantity | na.AbstractScalar,
):
    s = 14 * u.um
    d = 8.7 * u.um
    f = s - d

    kwargs = dict(
        thickness_substrate=s,
        thickness_depletion=d,
        width_max=width_max,
        width_depleted=width_depleted,
    )

    result = optika.sensors.charge_diffusion_profile(depth=depth, **kwargs)

    assert np.all(result >= 0 * u.um)

    width_max = f if width_max is None else width_max
    width_depleted = 0 * u.um if width_depleted is None else width_depleted

    back = optika.sensors.charge_diffusion_profile(depth=0 * u.um, **kwargs)
    expected = np.sqrt(np.square(width_max) + np.square(width_depleted))
    assert np.allclose(back, expected)

    front = optika.sensors.charge_diffusion_profile(depth=s, **kwargs)
    assert np.allclose(front, 0 * u.um)

    edge = optika.sensors.charge_diffusion_profile(depth=f, **kwargs)
    assert np.allclose(edge, width_depleted)

    assert np.all(result <= back)


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
