import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika.materials._tests.test_materials import AbstractTestAbstractMaterial


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        na.linspace(100, 200, axis="wavelength", num=4) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        0.5,
    ],
)
@pytest.mark.parametrize(
    argnames="n",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_oxide",
    argvalues=[
        10 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        10 * u.um,
    ],
)
def test_transmittance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.transmittance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        na.linspace(100, 200, axis="wavelength", num=4) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        0.5,
    ],
)
@pytest.mark.parametrize(
    argnames="n",
    argvalues=[
        1,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_oxide",
    argvalues=[
        10 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        10 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "Beer-Lambert",
        "exact",
    ],
)
def test_absorbance(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    n: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    method: str,
):
    result = optika.sensors.absorbance(
        wavelength=wavelength,
        direction=direction,
        n=n,
        thickness_oxide=thickness_oxide,
        thickness_substrate=thickness_substrate,
        method=method,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="absorption",
    argvalues=[
        1 / u.mm,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        1000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.2,
        1,
    ],
)
def test_charge_collection_efficiency(
    absorption: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.charge_collection_efficiency(
        absorption=absorption,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        na.linspace(100, 200, axis="wavelength", num=4) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="direction",
    argvalues=[
        1,
        0.5,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_oxide",
    argvalues=[
        10 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        1000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_substrate",
    argvalues=[
        1 * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.2,
        1,
    ],
)
def test_quantum_efficiency_effective(
    wavelength: u.Quantity | na.AbstractScalar,
    direction: float | na.AbstractScalar,
    thickness_oxide: u.Quantity | na.AbstractScalar,
    thickness_implant: u.Quantity | na.AbstractScalar,
    thickness_substrate: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.quantum_efficiency_effective(
        wavelength=wavelength,
        direction=direction,
        thickness_oxide=thickness_oxide,
        thickness_implant=thickness_implant,
        thickness_substrate=thickness_substrate,
        cce_backsurface=cce_backsurface,
    )
    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="iqy",
    argvalues=[1.61 * u.electron / u.photon],
)
@pytest.mark.parametrize(
    argnames="cce",
    argvalues=[0.9],
)
def test_probability_measurement(
    iqy: u.Quantity | na.AbstractScalar,
    cce: float | na.AbstractScalar,
):
    result = optika.sensors.probability_measurement(
        iqy=iqy,
        cce=cce,
    )
    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize(
    argnames="photons_absorbed",
    argvalues=[
        (100 * u.photon).astype(int),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        100 * u.nm,
        na.geomspace(1, 10000, axis="wavelength", num=5) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="thickness_implant",
    argvalues=[
        2000 * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="cce_backsurface",
    argvalues=[
        0.5,
    ],
)
@pytest.mark.parametrize("temperature", [300 * u.K])
def test_electrons_measured_approx(
    photons_absorbed: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.ScalarArray,
    thickness_implant: u.Quantity | na.AbstractScalar,
    cce_backsurface: u.Quantity | na.AbstractScalar,
    temperature: u.Quantity | na.ScalarArray,
):
    result = optika.sensors.electrons_measured_approx(
        photons_absorbed=photons_absorbed,
        wavelength=wavelength,
        thickness_implant=thickness_implant,
        cce_backsurface=cce_backsurface,
        temperature=temperature,
    )

    assert np.all(result >= 0 * u.electron)

    shape = na.shape_broadcasted(
        photons_absorbed,
        wavelength,
        thickness_implant,
        cce_backsurface,
        temperature,
    )

    assert result.shape == shape


@pytest.mark.parametrize(
    argnames="photons_expected",
    argvalues=[
        100 * u.photon,
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1000 * u.AA,
        na.geomspace(1, 10000, axis="wavelength", num=9) * u.AA,
    ],
)
@pytest.mark.parametrize(
    argnames="method",
    argvalues=[
        "monte-carlo",
        "expected",
    ],
)
def test_signal(
    photons_expected: u.Quantity | na.AbstractScalar,
    wavelength: u.Quantity | na.AbstractScalar,
    method: str,
):
    result = optika.sensors.signal(
        photons_expected=photons_expected,
        wavelength=wavelength,
        method=method,
    )
    assert np.all(result >= 0 * u.electron)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1000 * u.AA,
        na.geomspace(1, 10000, axis="wavelength", num=9) * u.AA,
    ],
)
def test_vmr_signal(
    wavelength: u.Quantity | na.AbstractScalar,
):
    result = optika.sensors.vmr_signal(
        wavelength=wavelength,
    )
    assert np.all(result >= 0 * u.electron)


@pytest.mark.parametrize(
    argnames="width_pixel",
    argvalues=[
        27 * u.um,
        na.Cartesian2dVectorArray(0, 4) * u.um,
    ],
)
@pytest.mark.parametrize(
    argnames="model",
    argvalues=[
        optika.sensors.diffusion.JanesickDiffusionModel,
        optika.sensors.diffusion.SlabDiffusionModel,
    ],
)
def test_vmr_signal_diffusion(
    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
    model: type[optika.sensors.diffusion.AbstractDiffusionModel],
):
    wavelength = 304 * u.AA
    diffusion = model(thickness_depletion=2 * u.um)
    axis_xy = ("detector_x", "detector_y")

    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=16, detector_y=16),
    )

    signal = optika.sensors.signal(
        photons_expected=photons_expected,
        wavelength=wavelength,
        thickness_substrate=7 * u.um,
        diffusion=diffusion,
        width_pixel=width_pixel,
        axis_xy=axis_xy,
        wrap=True,
        shape_random=dict(experiment=500),
    )

    vmr_measured = signal.vmr(("experiment",) + axis_xy)

    result = optika.sensors.vmr_signal(
        wavelength=wavelength,
        thickness_substrate=7 * u.um,
        diffusion=diffusion,
        width_pixel=width_pixel,
    )
    result_no_diffusion = optika.sensors.vmr_signal(wavelength=wavelength)

    assert np.all(result > 0 * u.electron)
    assert np.all(result < result_no_diffusion)
    assert np.abs(vmr_measured - result) < 0.1 * result

    # a model with no field-free region and no spread in the depletion
    # region does not spread the charge, so it is the same as no model
    result_sharp = optika.sensors.vmr_signal(
        wavelength=wavelength,
        diffusion=diffusion.replace(thickness_depletion=7 * u.um),
        thickness_substrate=7 * u.um,
        width_pixel=width_pixel,
    )
    result_none = optika.sensors.vmr_signal(
        wavelength=wavelength,
        thickness_substrate=7 * u.um,
    )
    assert np.allclose(result_sharp, result_none, rtol=1e-12)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        304 * u.AA,
        20 * u.AA,
    ],
)
def test_vmr_signal_depleted(
    wavelength: u.Quantity | na.AbstractScalar,
):
    """
    On a fully depleted sensor the only spread is acquired drifting across the
    depletion region, which the analytic VMR must reproduce, for photons
    absorbed at the back surface and for photons absorbed throughout the
    depletion region.
    """
    axis_xy = ("detector_x", "detector_y")

    model = optika.sensors.diffusion.JanesickDiffusionModel(
        thickness_depletion=14 * u.um,
        width_depletion=2 * u.um,
    )

    kwargs = dict(
        wavelength=wavelength,
        thickness_substrate=14 * u.um,
        diffusion=model,
        width_pixel=4 * u.um,
    )

    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=16, detector_y=16),
    )

    signal = optika.sensors.signal(
        photons_expected=photons_expected,
        axis_xy=axis_xy,
        wrap=True,
        shape_random=dict(experiment=200),
        **kwargs,
    )

    vmr_measured = signal.vmr(("experiment",) + axis_xy)

    result = optika.sensors.vmr_signal(**kwargs)
    result_sharp = optika.sensors.vmr_signal(
        **(kwargs | dict(diffusion=model.replace(width_depletion=None)))
    )

    assert np.all(result < result_sharp)
    assert np.abs(vmr_measured - result) < 0.05 * result


def test_diffusion_requires_pixels():
    """
    Diffusion spreads charge over pixels, so a model of diffusion needs the
    width of a pixel, and the Monte Carlo the axes of the pixel grid.
    """
    diffusion = optika.sensors.diffusion.JanesickDiffusionModel(
        thickness_depletion=2 * u.um
    )
    wavelength = 304 * u.AA
    thickness_substrate = 7 * u.um

    with pytest.raises(ValueError, match="width_pixel"):
        optika.sensors.vmr_signal(
            wavelength=wavelength,
            thickness_substrate=thickness_substrate,
            diffusion=diffusion,
        )
    with pytest.raises(ValueError, match="width_pixel"):
        optika.sensors.kernel_signal(
            wavelength=wavelength,
            axis_x="kernel_x",
            axis_y="kernel_y",
            thickness_substrate=thickness_substrate,
            diffusion=diffusion,
        )

    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=4, detector_y=4),
    )
    for method in ["expected", "monte-carlo"]:
        with pytest.raises(ValueError, match="axis_xy"):
            optika.sensors.signal(
                photons_expected=photons_expected,
                wavelength=wavelength,
                thickness_substrate=thickness_substrate,
                diffusion=diffusion,
                width_pixel=27 * u.um,
                method=method,
            )
        with pytest.raises(ValueError, match="width_pixel"):
            optika.sensors.signal(
                photons_expected=photons_expected,
                wavelength=wavelength,
                thickness_substrate=thickness_substrate,
                diffusion=diffusion,
                axis_xy=("detector_x", "detector_y"),
                method=method,
            )

    # the expected signal spreads the charge over the pixel grid,
    # so the photons must vary along both of its axes
    with pytest.raises(ValueError, match="both axes"):
        optika.sensors.signal(
            photons_expected=photons_expected[dict(detector_y=0)],
            wavelength=wavelength,
            thickness_substrate=thickness_substrate,
            diffusion=diffusion,
            width_pixel=27 * u.um,
            axis_xy=("detector_x", "detector_y"),
            method="expected",
        )


def test_diffusion_requires_substrate():
    """
    The depletion region of a model of diffusion is only meaningful relative
    to the substrate, so a model needs the thickness of the substrate rather
    than the default, which could silently leave no field-free region at all.
    """
    diffusion = optika.sensors.diffusion.JanesickDiffusionModel(
        thickness_depletion=8 * u.um
    )
    wavelength = 304 * u.AA
    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=4, detector_y=4),
    )
    kwargs = dict(
        diffusion=diffusion,
        width_pixel=27 * u.um,
    )
    with pytest.raises(ValueError, match="thickness_substrate"):
        optika.sensors.vmr_signal(wavelength, **kwargs)
    with pytest.raises(ValueError, match="thickness_substrate"):
        optika.sensors.kernel_signal(
            wavelength,
            axis_x="kernel_x",
            axis_y="kernel_y",
            **kwargs,
        )
    for method in ["expected", "monte-carlo"]:
        with pytest.raises(ValueError, match="thickness_substrate"):
            optika.sensors.signal(
                photons_expected,
                wavelength,
                method=method,
                axis_xy=("detector_x", "detector_y"),
                **kwargs,
            )
    with pytest.raises(ValueError, match="thickness_substrate"):
        optika.sensors.electrons_measured(
            photons_expected.astype(int),
            wavelength,
            axis_xy=("detector_x", "detector_y"),
            **kwargs,
        )

    # without a model, the default substrate is still that of Stern (1994)
    assert np.all(
        optika.sensors.vmr_signal(wavelength)
        == optika.sensors.vmr_signal(wavelength, thickness_substrate=7 * u.um)
    )


@pytest.mark.parametrize("diffusion", [False, True, 8 * u.um])
def test_diffusion_is_a_model(
    diffusion: object,
):
    """
    The `diffusion` argument is a model of diffusion, not a flag or a
    parameter of one.
    """
    wavelength = 304 * u.AA
    photons_expected = na.broadcast_to(
        100 * u.photon,
        shape=dict(detector_x=4, detector_y=4),
    )
    kwargs = dict(
        diffusion=diffusion,
        width_pixel=27 * u.um,
    )
    with pytest.raises(TypeError, match="AbstractDiffusionModel"):
        optika.sensors.vmr_signal(wavelength, **kwargs)
    with pytest.raises(TypeError, match="AbstractDiffusionModel"):
        optika.sensors.kernel_signal(
            wavelength,
            axis_x="kernel_x",
            axis_y="kernel_y",
            **kwargs,
        )
    for method in ["expected", "monte-carlo"]:
        with pytest.raises(TypeError, match="AbstractDiffusionModel"):
            optika.sensors.signal(
                photons_expected,
                wavelength,
                method=method,
                axis_xy=("detector_x", "detector_y"),
                **kwargs,
            )
    with pytest.raises(TypeError, match="AbstractDiffusionModel"):
        optika.sensors.electrons_measured(
            photons_expected.astype(int),
            wavelength,
            axis_xy=("detector_x", "detector_y"),
            **kwargs,
        )


def test_keyword_only():
    """
    Only the leading arguments are positional, so that removing or adding a
    parameter can never silently shift the meaning of the others.
    """
    wavelength = 304 * u.AA
    photons = 100 * u.photon
    with pytest.raises(TypeError, match="positional"):
        optika.sensors.vmr_signal(wavelength, 1)
    with pytest.raises(TypeError, match="positional"):
        optika.sensors.kernel_signal(wavelength, "kernel_x", "kernel_y")
    with pytest.raises(TypeError, match="positional"):
        optika.sensors.signal(photons, wavelength, 1)
    with pytest.raises(TypeError, match="positional"):
        optika.sensors.electrons_measured(photons, wavelength, 1 / u.um)


@pytest.mark.parametrize(
    argnames="width_backsurface,width_depletion",
    argvalues=[
        (None, None),
        (None, 0.8 * u.um),
        (4 * u.um, 3 * u.um),
    ],
)
def test_vmr_signal_quadrature(
    width_backsurface: None | u.Quantity | na.AbstractScalar,
    width_depletion: None | u.Quantity | na.AbstractScalar,
):
    """
    The charge-diffusion integral should be converged at the default number of
    quadrature nodes, across the full range of optical depths that silicon
    spans between 1 and 10000 angstroms, over the depletion region as well as
    the field-free region.
    """
    ccd = optika.sensors.materials.e2v_ccd97()

    kwargs = dict(
        wavelength=na.geomspace(1, 10000, axis="wavelength", num=101) * u.AA,
        thickness_implant=ccd.thickness_implant,
        thickness_substrate=ccd.thickness_substrate,
        diffusion=optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=ccd.diffusion.thickness_depletion,
            width_backsurface=width_backsurface,
            width_depletion=width_depletion,
        ),
        width_pixel=16 * u.um,
        cce_backsurface=ccd.cce_backsurface,
        temperature=ccd.temperature,
    )

    result = optika.sensors.vmr_signal(**kwargs)

    quadrature = optika.sensors.diffusion._quadrature
    num = quadrature._num_gauss_legendre
    try:
        quadrature._num_gauss_legendre = 8 * num
        expected = optika.sensors.vmr_signal(**kwargs)
    finally:
        quadrature._num_gauss_legendre = num

    assert np.all(np.abs(result / expected - 1) < 1e-5)


@pytest.mark.parametrize(
    argnames="diffusion",
    argvalues=[
        None,
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=7.85 * u.um,
        ),
        optika.sensors.diffusion.SlabDiffusionModel(
            thickness_depletion=7.55 * u.um,
            width_depletion=1.5 * u.um,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="width_pixel",
    argvalues=[
        16 * u.um,
        na.Cartesian2dVectorArray(0, 13) * u.um,
    ],
)
def test_kernel_signal(
    diffusion: None | optika.sensors.diffusion.AbstractDiffusionModel,
    width_pixel: u.Quantity | na.AbstractCartesian2dVectorArray,
):
    """
    The kernel sums to the expected number of electrons per absorbed photon,
    and with a back surface which loses no charge it is the kernel of the
    model of diffusion times the quantum yield.

    Without diffusion the sum is the charge collection efficiency of a
    substrate of infinite thickness, and with it, the efficiency for the
    photons absorbed in the substrate, which differ for the X-rays the
    substrate absorbs weakly.
    """
    axis = ("kernel_x", "kernel_y")
    wavelength = na.geomspace(10, 3000, axis="wavelength", num=11) * u.AA
    thickness_implant = 0.2 * u.um
    thickness_substrate = 14 * u.um
    cce_backsurface = 0.2
    kwargs = dict(
        thickness_implant=thickness_implant,
        thickness_substrate=thickness_substrate,
        diffusion=diffusion,
        width_pixel=width_pixel,
    )

    result = optika.sensors.kernel_signal(
        wavelength=wavelength,
        axis_x=axis[0],
        axis_y=axis[1],
        cce_backsurface=cce_backsurface,
        **kwargs,
    )
    assert isinstance(result, na.FunctionArray)
    assert result.outputs.unit.is_equivalent(u.electron / u.photon)
    assert np.all(result.outputs >= 0 * u.electron / u.photon)

    absorption = optika.chemicals.Chemical("Si").absorption(wavelength)
    iqy = optika.sensors.quantum_yield_ideal(wavelength)

    # the fraction of the electrons lost in the implant, per photon absorbed
    # in a substrate of infinite thickness
    aW = (absorption * thickness_implant).to(u.dimensionless_unscaled).value
    lost = (1 - cce_backsurface) * (aW + np.expm1(-aW)) / aW
    cce = 1 - lost
    cce_substrate = 1 - lost / -np.expm1(-absorption * thickness_substrate)

    if diffusion is None:
        assert np.allclose(result.outputs.sum(axis), iqy * cce, rtol=1e-12)
        assert na.shape(result.outputs) == dict(wavelength=11, kernel_x=1, kernel_y=1)
        result = optika.sensors.kernel_signal(
            wavelength=wavelength,
            axis_x=axis[0],
            axis_y=axis[1],
            thickness_implant=thickness_implant,
            cce_backsurface=cce_backsurface,
            num=3,
        )
        assert np.allclose(result.outputs.sum(axis), iqy * cce, rtol=1e-12)
        assert result.outputs[dict(kernel_x=0)].sum() == 0
        return

    assert np.allclose(result.outputs.sum(axis), iqy * cce_substrate, rtol=1e-5)
    assert not np.allclose(result.outputs.sum(axis), iqy * cce, rtol=1e-5)

    result = optika.sensors.kernel_signal(
        wavelength=wavelength,
        axis_x=axis[0],
        axis_y=axis[1],
        cce_backsurface=1,
        **kwargs,
    )
    kernel = diffusion.kernel_average(
        absorption=absorption,
        thickness_substrate=14 * u.um,
        width_pixel=width_pixel,
        axis_x=axis[0],
        axis_y=axis[1],
    )
    expected = iqy * kernel.outputs
    assert np.allclose(result.outputs, expected, atol=1e-6 * expected.max().ndarray)


@pytest.mark.parametrize(
    argnames="diffusion",
    argvalues=[
        optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=7.85 * u.um,
        ),
        optika.sensors.diffusion.SlabDiffusionModel(
            thickness_depletion=7.55 * u.um,
        ),
    ],
)
def test_kernel_signal_monte_carlo(
    diffusion: optika.sensors.diffusion.AbstractDiffusionModel,
):
    """
    The kernel is the mean of the electrons the Monte Carlo simulation
    measures around each absorbed photon,
    including the charge lost to recombination near the back surface,
    for photons absorbed within the implant layer.
    """
    num = 9
    num_photon = 100000
    axis_xy = ("pixel_x", "pixel_y")
    kwargs = dict(
        wavelength=304 * u.AA,
        thickness_implant=0.2 * u.um,
        thickness_substrate=14 * u.um,
        diffusion=diffusion,
        width_pixel=13 * u.um,
        cce_backsurface=0.2,
    )

    photons = np.zeros((num, num))
    photons[num // 2, num // 2] = num_photon
    photons = na.ScalarArray(photons << u.photon, axes=axis_xy).astype(int)

    electrons = optika.sensors.electrons_measured(
        photons_absorbed=photons,
        axis_xy=axis_xy,
        **kwargs,
    )
    result = electrons / (num_photon * u.photon)

    kernel = optika.sensors.kernel_signal(
        axis_x=axis_xy[0],
        axis_y=axis_xy[1],
        num=num,
        **kwargs,
    )

    gain = kernel.outputs.sum(axis_xy)
    assert np.allclose(result, kernel.outputs, atol=0.004 * gain.ndarray)


@pytest.mark.parametrize(
    argnames="wrap",
    argvalues=[False, True],
)
def test_signal_expected_diffusion(
    wrap: bool,
):
    """
    Without noise, the electrons of the photons absorbed in each pixel are
    spread over the pixels around it with the kernel,
    and the charge which leaves the grid is lost unless it wraps around.
    """
    axis_xy = ("pixel_x", "pixel_y")
    num_x, num_y = 7, 6
    i_x, i_y = 1, 3
    num_photon = 1000

    kwargs = dict(
        wavelength=304 * u.AA,
        thickness_substrate=14 * u.um,
        diffusion=optika.sensors.diffusion.SlabDiffusionModel(
            thickness_depletion=7.55 * u.um,
        ),
        width_pixel=8 * u.um,
    )

    photons = np.zeros((num_x, num_y))
    photons[i_x, i_y] = num_photon
    photons = na.ScalarArray(photons << u.photon, axes=axis_xy)

    result = optika.sensors.signal(
        photons_expected=photons,
        absorbance=1,
        method="expected",
        axis_xy=axis_xy,
        wrap=wrap,
        **kwargs,
    )

    kernel = optika.sensors.kernel_signal(
        axis_x="kernel_x",
        axis_y="kernel_y",
        **kwargs,
    )
    kernel = kernel.outputs.ndarray_aligned(("kernel_x", "kernel_y"))
    half_x, half_y = kernel.shape[0] // 2, kernel.shape[1] // 2

    expected = np.zeros((num_x, num_y)) * u.electron
    for k_x in range(kernel.shape[0]):
        for k_y in range(kernel.shape[1]):
            j_x = i_x + k_x - half_x
            j_y = i_y + k_y - half_y
            if wrap:
                j_x, j_y = j_x % num_x, j_y % num_y
            elif not (0 <= j_x < num_x and 0 <= j_y < num_y):
                continue
            expected[j_x, j_y] += num_photon * u.photon * kernel[k_x, k_y]

    assert np.allclose(result.ndarray_aligned(axis_xy), expected)
    if wrap:
        total = num_photon * u.photon * kernel.sum()
        assert np.isclose(result.sum(axis_xy).ndarray, total)


def test_signal_expected_direction():
    """
    A direction which varies over the pixel grid gives the photons absorbed
    in each pixel the kernel of their own absorption coefficient,
    interpolated to about one part in a million.
    """
    axis_xy = ("pixel_x", "pixel_y")
    shape = dict(pixel_x=5, pixel_y=4)
    angle = na.linspace(0, 30, axis="pixel_x", num=5) * u.deg
    angle = angle + na.linspace(0, 5, axis="pixel_y", num=4) * u.deg
    direction = np.cos(angle)
    photons = na.random.uniform(0, 100, shape_random=shape) * u.photon

    kwargs = dict(
        wavelength=304 * u.AA,
        thickness_substrate=14 * u.um,
        diffusion=optika.sensors.diffusion.SlabDiffusionModel(
            thickness_depletion=7.55 * u.um,
        ),
        width_pixel=8 * u.um,
    )

    result = optika.sensors.signal(
        photons_expected=photons,
        direction=direction,
        absorbance=1,
        method="expected",
        axis_xy=axis_xy,
        **kwargs,
    )

    # the electrons of each pixel spread with the kernel of its own direction
    expected = 0 * u.electron
    for i_x in range(shape["pixel_x"]):
        for i_y in range(shape["pixel_y"]):
            index = dict(pixel_x=i_x, pixel_y=i_y)
            source = np.zeros(tuple(shape.values())) * u.photon
            source[i_x, i_y] = photons[index].ndarray
            expected = expected + optika.sensors.signal(
                photons_expected=na.ScalarArray(source, axes=axis_xy),
                direction=direction[index],
                absorbance=1,
                method="expected",
                axis_xy=axis_xy,
                **kwargs,
            )

    assert np.allclose(result, expected, atol=1e-6 * expected.max().ndarray)


class AbstractTestAbstractSensorMaterial(
    AbstractTestAbstractMaterial,
):
    @pytest.mark.parametrize(
        argnames="photons",
        argvalues=[
            1e-6 * u.erg,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    @pytest.mark.parametrize(
        argnames="noise",
        argvalues=[True, False],
    )
    def test_signal(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        photons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
        noise: bool,
    ):
        result = a.signal(
            photons=photons,
            wavelength=wavelength,
            direction=direction,
            noise=noise,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result >= 0 * u.electron)

    @pytest.mark.parametrize(
        argnames="electrons",
        argvalues=[
            100 * u.electron,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, 1),
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            na.Cartesian3dVectorArray(0, 0, -1),
        ],
    )
    def test_photons_incident(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.photons_incident(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.photon)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    def test_kernel(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
    ):
        """
        Without pixels the kernel is the noiseless signal of one photon in a
        single pixel, and with them it spreads the same signal over the
        pixels around it.
        """
        axis = ("kernel_x", "kernel_y")
        signal = a.signal(
            photons=1 * u.photon,
            wavelength=wavelength,
            direction=direction,
            noise=False,
        )

        result = a.kernel(wavelength, *axis, direction=direction)
        assert isinstance(result, na.FunctionArray)
        assert result.outputs.unit.is_equivalent(u.electron / u.photon)
        assert na.shape(result.outputs) == dict(kernel_x=1, kernel_y=1)
        assert np.allclose(result.outputs.sum(axis) * u.photon, signal)

        result = a.kernel(wavelength, *axis, direction=direction, width_pixel=15 * u.um)
        center = result.outputs[dict(kernel_x=result.outputs.shape["kernel_x"] // 2)]
        center = center[dict(kernel_y=result.outputs.shape["kernel_y"] // 2)]
        assert np.all(result.outputs >= 0 * u.electron / u.photon)
        assert np.all(result.outputs <= center)
        assert np.allclose(result.outputs.sum(axis) * u.photon, signal, rtol=1e-5)

        with pytest.raises(ValueError, match="num"):
            a.kernel(wavelength, *axis, direction=direction, num=2)

    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
            na.linspace(0.8, 1, axis="detector_x", num=12),
        ],
    )
    @pytest.mark.parametrize(
        argnames="width_pixel",
        argvalues=[
            None,
            15 * u.um,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wrap",
        argvalues=[False, True],
    )
    def test_signal_transposed(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        direction: float | na.AbstractScalar,
        width_pixel: None | u.Quantity,
        wrap: bool,
    ):
        """
        The transpose is the adjoint of the noiseless signal,
        whether or not the charge diffuses,
        and whether or not each pixel has its own kernel.
        """
        axis_xy = ("detector_x", "detector_y")
        shape = dict(detector_x=12, detector_y=10)
        wavelength = 100 * u.AA
        photons = na.random.uniform(0, 100, shape_random=shape) * u.photon
        electrons = na.random.uniform(0, 100, shape_random=shape) * u.electron
        kwargs = dict(
            wavelength=wavelength,
            direction=direction,
            width_pixel=width_pixel,
            axis_xy=axis_xy,
            wrap=wrap,
        )

        forward = a.signal(photons, noise=False, **kwargs)
        result = a.signal_transposed(electrons, **kwargs)

        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.allclose(
            (forward * electrons).sum(axis_xy),
            (photons * result).sum(axis_xy),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    def test_backproject(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        direction: float | na.AbstractScalar,
    ):
        """
        The backprojection inverts the noiseless signal (which uses unit
        absorbance) of the photons absorbed in a single pixel,
        and of photons absorbed uniformly across a periodic grid of pixels
        over which the charge diffuses.
        """
        wavelength = 100 * u.AA
        photons = 100 * u.photon
        electrons = a.signal(
            photons=photons,
            wavelength=wavelength,
            direction=direction,
            noise=False,
        )
        result = a.backproject(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.photon)
        assert np.allclose(result, photons)

        axis_xy = ("detector_x", "detector_y")
        photons = na.broadcast_to(photons, dict(detector_x=12, detector_y=10))
        kwargs = dict(
            wavelength=wavelength,
            direction=direction,
            width_pixel=15 * u.um,
            axis_xy=axis_xy,
            wrap=True,
        )
        electrons = a.signal(photons, noise=False, **kwargs)
        result = a.backproject(electrons, **kwargs)
        assert np.allclose(result, photons, rtol=1e-5)

    @pytest.mark.parametrize(
        argnames="electrons",
        argvalues=[
            1000 * u.electron,
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            1,
            0.5,
        ],
    )
    def test_uncertainty(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        electrons: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: float | na.AbstractScalar,
    ):
        result = a.uncertainty(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert result.unit.is_equivalent(u.electron)
        assert np.all(result >= 0 * u.electron)

        result_diffusion = a.uncertainty(
            electrons=electrons,
            wavelength=wavelength,
            direction=direction,
            width_pixel=15 * u.um,
        )
        assert np.all(result_diffusion <= result)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    def test_direction_refracted(
        self,
        a: optika.sensors.materials.AbstractSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        # with the default direction and normal (normal incidence) the
        # refracted cosine is unity
        result = a.direction_refracted(wavelength=wavelength)
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(np.real(result) > 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.IdealSensorMaterial(),
    ],
)
class TestIdealSensorMaterial(
    AbstractTestAbstractSensorMaterial,
):
    pass


class AbstractTestAbstractSiliconSensorMaterial(
    AbstractTestAbstractSensorMaterial,
):

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            500 * u.nm,
        ],
    )
    def test_quantum_yield_ideal(
        self,
        a: optika.sensors.materials.AbstractSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.quantum_yield_ideal(wavelength)
        assert result >= 0

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            3 * u.eV,
            na.geomspace(1, 10000, axis="wavelength", num=5) * u.AA,
        ],
    )
    def test_fano_factor(
        self,
        a: optika.sensors.materials.AbstractSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.fano_factor(wavelength)
        assert np.all(result >= 0 * u.electron / u.photon)


class AbstractTestAbstractBackIlluminatedSiliconSensorMaterial(
    AbstractTestAbstractSiliconSensorMaterial,
):
    def test_thickness_oxide(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_oxide
        assert result >= 0 * u.mm

    def test_thickness_implant(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_implant
        assert result >= 0 * u.mm

    def test_thickness_substrate(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.thickness_substrate
        assert result >= 0 * u.mm

    def test_cce_backsurface(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.cce_backsurface
        assert result >= 0

    def test_num_interpolation(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.num_interpolation
        assert (result is None) or (result > 0)

    def test_efficiency_interpolated(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        """
        Interpolating the absorbance over the angle of incidence reproduces
        computing it for every ray, and does so better the more nodes it is
        given.
        """
        angle = na.linspace(0, 20, axis="angle", num=25) * u.deg
        rays = optika.rays.RayVectorArray(
            wavelength=na.linspace(100, 1000, axis="wavelength", num=5) * u.AA,
            direction=na.Cartesian3dVectorArray(np.sin(angle), 0, np.cos(angle)),
        )
        normal = na.Cartesian3dVectorArray(0, 0, -1)

        expected = a.efficiency(rays, normal)
        scale = np.abs(expected).max()

        def error(num: int) -> float:
            result = dataclasses.replace(a, num_interpolation=num).efficiency(
                rays=rays,
                normal=normal,
            )
            assert na.shape(result) == na.shape(expected)
            return np.abs(result - expected).max() / scale

        coarse, fine = error(4), error(32)

        assert coarse < 0.01
        assert fine <= coarse

    def test_efficiency_interpolated_scalar(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        """With a single angle of incidence there is nothing to interpolate
        over, so the absorbance is computed directly."""
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

    def test_diffusion(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        result = a.diffusion
        if result is not None:
            assert isinstance(result, optika.sensors.diffusion.AbstractDiffusionModel)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    def test_width_charge_diffusion(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.width_charge_diffusion(wavelength=wavelength)
        assert np.all(result >= 0 * u.um)

        # without a model the charge does not spread at any wavelength,
        # including those silicon does not absorb at all
        if a.diffusion is None:
            transparent = 20000 * u.AA
            assert a._chemical.absorption(transparent) == 0 / u.um
            wavelength = na.stack([wavelength, transparent], axis="wavelength")
            result = a.width_charge_diffusion(wavelength=wavelength)
            assert np.all(result == 0 * u.um)

    def test_signal_diffusion(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
    ):
        """
        Charge diffuses over the pixel grid whenever the width of a pixel is
        given, as in :meth:`uncertainty`, so a material with a model of
        diffusion needs the axes of the grid too.
        """
        axis_xy = ("detector_x", "detector_y")
        photons = na.broadcast_to(
            100 * u.photon,
            shape=dict(detector_x=4, detector_y=4),
        )
        kwargs = dict(
            wavelength=304 * u.AA,
            width_pixel=15 * u.um,
        )
        for noise in [True, False]:
            result = a.signal(photons, axis_xy=axis_xy, noise=noise, **kwargs)
            assert result.shape == photons.shape
            assert np.all(result >= 0 * u.electron)

            if a.diffusion is not None:
                with pytest.raises(ValueError, match="axis_xy"):
                    a.signal(photons, noise=noise, **kwargs)
                with pytest.raises(ValueError, match="axis_xy"):
                    a.signal_transposed(result, **kwargs)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_transmittance(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.transmittance(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_absorbance(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.absorbance(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_charge_collection_efficiency(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.charge_collection_efficiency(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_quantum_efficiency(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.quantum_efficiency(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert result > 0 * u.electron / u.photon

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_probability_measurement(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.probability_measurement(
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    @pytest.mark.parametrize(
        argnames="photons_absorbed",
        argvalues=[
            (100 * u.photon).astype(int),
        ],
    )
    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            100 * u.AA,
        ],
    )
    @pytest.mark.parametrize(
        argnames="direction",
        argvalues=[
            None,
        ],
    )
    @pytest.mark.parametrize(
        argnames="normal",
        argvalues=[
            None,
        ],
    )
    def test_electrons_measured(
        self,
        a: optika.sensors.materials.AbstractBackIlluminatedSiliconSensorMaterial,
        photons_absorbed: u.Quantity | na.AbstractScalar,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: None | na.AbstractCartesian3dVectorArray,
        normal: None | na.AbstractCartesian3dVectorArray,
    ):
        result = a.electrons_measured(
            photons_absorbed=photons_absorbed,
            wavelength=wavelength,
            direction=direction,
            normal=normal,
        )
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert np.all(result >= 0 * u.electron)


def _e2v_ccd97_janesick() -> (
    optika.sensors.materials.BackIlluminatedSiliconSensorMaterial
):
    """
    An e2v CCD97 with the model of Janesick (2001), whose charge spreads in
    the depletion region too.
    """
    result = optika.sensors.materials.e2v_ccd97()
    return result.replace(
        diffusion=optika.sensors.diffusion.JanesickDiffusionModel(
            thickness_depletion=result.diffusion.thickness_depletion,
            width_backsurface=5 * u.um,
            width_depletion=0.8 * u.um,
        ),
    )


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.sensors.materials.tektronix_tk512cb(),
        optika.sensors.materials.e2v_ccd97(),
        optika.sensors.materials.e2v_ccd203(),
        _e2v_ccd97_janesick(),
        optika.sensors.materials.e2v_ccd97().replace(diffusion=None),
    ],
)
class TestBackIlluminatedSiliconSensorMaterial(
    AbstractTestAbstractBackIlluminatedSiliconSensorMaterial,
):
    def test_eqe_measured(
        self,
        a: optika.sensors.materials.BackIlluminatedSiliconSensorMaterial,
    ):
        result = a.eqe_measured
        assert isinstance(result, na.AbstractFunctionArray)
        assert np.all(result.outputs >= 0)
        assert np.all(result.outputs <= 1.1)
        assert np.all(result.inputs >= 0 * u.nm)
