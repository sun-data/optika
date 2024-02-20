from typing import Literal
import pytest
import numpy as np
import matplotlib
import astropy.units as u
import astropy.visualization
import named_arrays as na
import optika._tests.test_mixins


class AbstractTestAbstractLayer(
    optika._tests.test_mixins.AbstractTestPrintable,
):

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            na.geomspace(100, 1000, axis="wavelength", num=4) * u.AA,
        ],
    )
    @pytest.mark.parametrize("direction", [na.Cartesian3dVectorArray(0, 0, 1)])
    @pytest.mark.parametrize("polarization", ["s", "p"])
    @pytest.mark.parametrize("normal", [na.Cartesian3dVectorArray(0, 0, -1)])
    def test_matrix_transfer(
        self,
        a: optika.materials.AbstractLayer,
        wavelength: u.Quantity | na.AbstractScalar,
        direction: na.AbstractCartesian3dVectorArray,
        polarization: Literal["s", "p"],
        normal: na.AbstractCartesian3dVectorArray,
    ):
        result = a.matrix_transfer(
            wavelength=wavelength,
            direction=direction,
            polarization=polarization,
            normal=normal,
        )
        assert isinstance(result, na.AbstractCartesian2dMatrixArray)
        assert np.all(result.determinant != 0)

    @pytest.mark.parametrize("z", [0 * u.nm])
    @pytest.mark.parametrize("ax", [None])
    def test_plot(
        self,
        a: optika.materials.AbstractLayer,
        z: u.Quantity,
        ax: None | matplotlib.axes.Axes,
    ):
        with astropy.visualization.quantity_support():
            result = a.plot(
                z=z,
                ax=ax,
            )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, matplotlib.patches.Polygon)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Layer(
            material="Si",
            thickness=10 * u.nm,
            kwargs_plot=dict(
                color="tab:orange",
                alpha=0.5,
            ),
        ),
        optika.materials.Layer(
            material="Si",
            thickness=10 * u.nm,
            x_label=-0.1,
        ),
        optika.materials.Layer(
            material="Si",
            thickness=10 * u.nm,
            x_label=1.1,
        ),
    ],
)
class TestLayer(
    AbstractTestAbstractLayer,
):
    pass
