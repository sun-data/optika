import pytest
import numpy as np
import matplotlib
import astropy.units as u
import astropy.visualization
import named_arrays as na
import optika._tests.test_mixins


class AbstractTestAbstractLayer(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestShaped,
):

    @pytest.mark.parametrize("wavelength", [100 * u.AA])
    def test_n(
        self,
        a: optika.materials.AbstractLayer,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.n(wavelength)
        assert np.all(np.real(result) >= 0)
        assert np.all(np.imag(result) >= 0)

    def test_thickness(
        self,
        a: optika.materials.AbstractLayer,
    ):
        result = a.thickness
        assert np.all(result >= 0 * u.nm)

    def test_interface(
        self,
        a: optika.materials.AbstractLayer,
    ):
        result = a.interface
        if result is not None:
            assert isinstance(
                result, optika.materials.profiles.AbstractInterfaceProfile
            )

    def test_layer_sequence(
        self,
        a: optika.materials.AbstractLayer,
    ):
        result = a.layer_sequence
        assert isinstance(result, optika.materials.LayerSequence)
        for layer in result.layers:
            assert isinstance(layer, optika.materials.AbstractLayer)

    @pytest.mark.parametrize(
        argnames="wavelength",
        argvalues=[
            na.geomspace(100, 1000, axis="wavelength", num=4) * u.AA,
        ],
    )
    @pytest.mark.parametrize("direction", [1])
    @pytest.mark.parametrize("polarized_s", [False, True])
    @pytest.mark.parametrize("n", [1])
    class TestMatrixTransfer:
        def test_transfer(
            self,
            a: optika.materials.AbstractLayer,
            wavelength: u.Quantity | na.AbstractScalar,
            direction: float | na.AbstractScalar,
            polarized_s: bool | na.AbstractScalar,
            n: float | na.AbstractScalar,
        ):
            n, direction, result, where = a.transfer(
                wavelength=wavelength,
                direction=direction,
                polarized_s=polarized_s,
                n=n,
            )
            assert np.all(n > 0)
            assert isinstance(direction, na.AbstractScalar)
            assert isinstance(result, na.AbstractCartesian2dMatrixArray)
            assert np.all(result.determinant != 0)
            assert np.all(np.isfinite(result))

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
            assert isinstance(item, matplotlib.patches.Rectangle)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.Layer(
            chemical="Si",
            thickness=10 * u.nm,
            kwargs_plot=dict(
                color="tab:orange",
                alpha=0.5,
            ),
        ),
        optika.materials.Layer(
            chemical="Si",
            thickness=10 * u.nm,
            x_label=-0.1,
        ),
        optika.materials.Layer(
            chemical="Si",
            thickness=10 * u.nm,
            x_label=1.1,
        ),
    ],
)
class TestLayer(
    AbstractTestAbstractLayer,
):
    pass


class AbstractTestAbstractLayerSequence(
    AbstractTestAbstractLayer,
):
    def test_layers(self, a: optika.materials.AbstractLayerSequence):
        result = a.layers
        for layer in result:
            assert isinstance(layer, optika.materials.AbstractLayer)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.LayerSequence(
            layers=[
                optika.materials.Layer(
                    chemical="SiO2",
                    thickness=10 * u.nm,
                ),
                optika.materials.Layer(
                    chemical="Si",
                    thickness=1 * u.um,
                ),
            ]
        ),
        optika.materials.LayerSequence(
            layers=[
                optika.materials.Layer(
                    chemical="SiO2",
                    thickness=10 * u.nm,
                ),
                optika.materials.LayerSequence(
                    [
                        optika.materials.Layer(
                            chemical="Si",
                            thickness=1 * u.um,
                        ),
                        optika.materials.Layer(
                            chemical="Mo",
                            thickness=1 * u.um,
                        ),
                    ]
                ),
            ]
        ),
    ],
)
class TestLayerSequence(
    AbstractTestAbstractLayerSequence,
):

    @pytest.mark.parametrize("item", [0, slice(0, 1)])
    def test__getitem__(
        self,
        a: optika.materials.LayerSequence,
        item: int | slice,
    ):
        result = a[item]
        if isinstance(item, int):
            assert isinstance(result, optika.materials.AbstractLayer)
        else:
            assert isinstance(result, optika.materials.LayerSequence)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.materials.PeriodicLayerSequence(
            layers=[
                optika.materials.Layer(
                    chemical="Si",
                    thickness=10 * u.nm,
                    interface=optika.materials.profiles.ErfInterfaceProfile(
                        width=2 * u.nm,
                    ),
                ),
                optika.materials.Layer(
                    chemical="Mo",
                    thickness=10 * u.nm,
                    interface=optika.materials.profiles.ErfInterfaceProfile(
                        width=2 * u.nm,
                    ),
                ),
            ],
            num_periods=10,
        ),
        optika.materials.PeriodicLayerSequence(
            layers=[
                optika.materials.PeriodicLayerSequence(
                    layers=[
                        optika.materials.Layer(
                            chemical="Si",
                            thickness=10 * u.nm,
                            interface=optika.materials.profiles.ErfInterfaceProfile(
                                width=2 * u.nm,
                            ),
                        ),
                        optika.materials.Layer(
                            chemical="Mo",
                            thickness=10 * u.nm,
                            interface=optika.materials.profiles.ErfInterfaceProfile(
                                width=2 * u.nm,
                            ),
                        ),
                    ],
                    num_periods=2,
                )
            ],
            num_periods=5,
        ),
    ],
)
class TestPeriodicLayerSequence(
    AbstractTestAbstractLayerSequence,
):
    class TestMatrixTransfer(
        AbstractTestAbstractLayerSequence.TestMatrixTransfer,
    ):
        def test_transfer(
            self,
            a: optika.materials.PeriodicLayerSequence,
            wavelength: u.Quantity | na.AbstractScalar,
            direction: float | na.AbstractScalar,
            polarized_s: bool | na.AbstractScalar,
            n: float | na.AbstractScalar,
        ):
            super().test_transfer(
                a=a,
                wavelength=wavelength,
                direction=direction,
                polarized_s=polarized_s,
                n=n,
            )

            b = optika.materials.LayerSequence(list(a.layers) * a.num_periods)

            (
                n_test,
                direction_test,
                result_test,
                where_test,
            ) = a.transfer(
                wavelength=wavelength,
                direction=direction,
                polarized_s=polarized_s,
                n=n,
            )

            (
                n_expected,
                direction_expected,
                result_expected,
                where_expected,
            ) = b.transfer(
                wavelength=wavelength,
                direction=direction,
                polarized_s=polarized_s,
                n=n,
            )

            assert np.allclose(n_test, n_expected)
            assert np.allclose(direction_test, direction_expected)
            assert np.allclose(result_test, result_expected)
            assert np.allclose(where_test, where_expected)
