import pytest
import abc
import numpy as np
import matplotlib.axes
import matplotlib.artist
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika.plotting


kwargs_plot_parameterization = [
    dict(),
    dict(color="red"),
]


class AbstractTestPlottable(abc.ABC):
    @pytest.mark.parametrize(
        argnames="ax",
        argvalues=[
            None,
            plt.subplots()[1],
        ],
    )
    @pytest.mark.parametrize(
        argnames="transform",
        argvalues=[
            None,
            optika.transforms.TransformList(
                [
                    optika.transforms.Translation(
                        na.Cartesian3dVectorArray(x=5 * u.mm),
                    ),
                    optika.transforms.RotationX(54 * u.deg),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize(
        argnames="component_map",
        argvalues=[
            None,
            dict(x="y", y="x"),
        ],
    )
    class TestPlot(abc.ABC):
        def test_plot(
            self,
            a: optika.plotting.Plottable,
            ax: None | matplotlib.axes.Axes | na.ScalarArray,
            transform: None | optika.transforms.AbstractTransform,
            component_map: None | dict[str, str],
        ):
            result = a.plot(
                ax=ax,
                transform=transform,
                component_map=component_map,
            )

            if ax is None or ax is np._NoValue:
                ax_normalized = plt.gca()
            else:
                ax_normalized = ax
            ax_normalized = na.as_named_array(ax_normalized)

            for index in ax_normalized.ndindex():
                assert ax_normalized[index].ndarray.has_data()

            assert isinstance(result, na.AbstractScalar)
            assert result.dtype == matplotlib.artist.Artist
