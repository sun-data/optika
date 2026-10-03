import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


@pytest.mark.parametrize("kind", ["thick", "thin"])
def test_mcc_stern2004(kind: str):
    result = optika.sensors.diffusion.mcc_stern2004(kind)
    assert isinstance(result, na.FunctionArray)
    assert np.all(result.inputs > 0 * u.AA)
    assert np.all(result.outputs > 0)
    assert np.all(result.outputs <= 1)


@pytest.mark.parametrize(
    argnames="model,kind",
    argvalues=[
        (optika.sensors.diffusion.e2v_ccd64_thick, "thick"),
        (optika.sensors.diffusion.e2v_ccd64_thin, "thin"),
    ],
)
def test_e2v_ccd64(model, kind: str):
    """
    Each model is Janesick's, with a depletion region inside the substrate
    that reproduces the measurement to a few percent.
    """
    result = model()
    assert isinstance(result, optika.sensors.diffusion.JanesickDiffusionModel)

    thickness_substrate = (
        optika.sensors.diffusion._stern2004._stern2004._thickness_substrate[kind]
    )
    assert 0 * u.um < result.thickness_depletion < thickness_substrate

    measured = optika.sensors.diffusion.mcc_stern2004(kind)
    mcc = result.mean_charge_capture(
        absorption=optika.chemicals.Chemical("Si").absorption(measured.inputs),
        thickness_substrate=thickness_substrate,
        width_pixel=16 * u.um,
    )
    rms = np.sqrt(np.mean(np.square(mcc - measured.outputs)))
    assert rms < 0.05
