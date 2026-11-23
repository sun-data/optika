import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins


class AbstractTestAbstractMesh(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
):
    def test_chemical(self, a: optika.materials.meshes.AbstractMesh):
        result = a.chemical
        assert isinstance(result, (str, optika.chemicals.AbstractChemical))

    def test_efficiency(self, a: optika.materials.meshes.AbstractMesh):
        result = a.efficiency
        assert result > 0

    def test_pitch(self, a: optika.materials.meshes.AbstractMesh):
        result = a.pitch
        assert result > (0 / u.mm)

    def test_efficiency_order(self, a: optika.materials.meshes.AbstractMesh) -> None:
        """
        The orders carry the light which passes the mesh, the zeroth carries
        the square of the efficiency, and the orders are symmetric and
        separable.
        """
        num = 1000
        order = na.Cartesian2dVectorArray(
            x=na.arange(-num, num + 1, axis="m"),
            y=na.arange(-num, num + 1, axis="n"),
        )
        result = a.efficiency_order(order)
        assert np.all(result >= 0)

        # The light in the orders beyond `num` is about 1 / (pi^2 num) of
        # the light along each axis.
        assert np.allclose(result.sum(("m", "n")), a.efficiency, rtol=1e-3)
        assert np.allclose(result[dict(m=num, n=num)], np.square(a.efficiency))

        reversed_ = result[dict(m=slice(None, None, -1), n=slice(None, None, -1))]
        assert np.all(np.abs(reversed_ - result) <= 1e-12 * result)

        separable = (
            result[dict(n=num)] * result[dict(m=num)] / result[dict(m=num, n=num)]
        )
        assert np.allclose(separable, result, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.meshes.Mesh(
            chemical="Ni",
            efficiency=0.8,
            pitch=70 / u.mm,
        )
    ],
)
class TestMesh(
    AbstractTestAbstractMesh,
):
    pass
