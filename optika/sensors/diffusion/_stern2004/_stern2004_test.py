import pytest
import numpy as np
import astropy.units as u
import optika


@pytest.mark.parametrize(
    argnames="kind,thickness_substrate",
    argvalues=[
        ("thick", 15 * u.um),
        ("thin", 8 * u.um),
    ],
)
def test_mcc_stern2004(kind: str, thickness_substrate: u.Quantity):
    """
    The measurement carries the CCD64 it was made on, which every
    comparison of a model against it needs.
    """
    result = optika.sensors.diffusion.mcc_stern2004(kind)
    assert isinstance(result, optika.sensors.diffusion.MeanChargeCapture)
    assert np.all(result.inputs > 0 * u.AA)
    assert np.all(result.outputs > 0)
    assert np.all(result.outputs <= 1)
    assert result.thickness_substrate == thickness_substrate
    assert result.width_pixel == 16 * u.um
    assert result.chemical_substrate == "Si"

    # the fields follow the measurement when it is indexed
    first = result[dict(wavelength=slice(0, 1))]
    assert first.thickness_substrate == thickness_substrate


@pytest.mark.parametrize(
    argnames="model,kind",
    argvalues=[
        (optika.sensors.diffusion.e2v_ccd64_thick, "thick"),
        (optika.sensors.diffusion.e2v_ccd64_thin, "thin"),
    ],
)
def test_e2v_ccd64(model, kind: str):
    """
    Each model is of the field-free region, with a depletion region inside
    the substrate that reproduces the measurement to within the scatter of
    repeated measurements.
    """
    result = model()
    assert isinstance(result, optika.sensors.diffusion.SlabDiffusionModel)

    measured = optika.sensors.diffusion.mcc_stern2004(kind)
    thickness_substrate = measured.thickness_substrate
    assert 0 * u.um < result.thickness_depletion < thickness_substrate

    mcc = result.mean_charge_capture(
        absorption=optika.chemicals.Chemical("Si").absorption(measured.inputs),
        thickness_substrate=thickness_substrate,
        width_pixel=16 * u.um,
    )
    rms = np.sqrt(np.mean(np.square(mcc - measured.outputs)))
    assert rms < 0.03
