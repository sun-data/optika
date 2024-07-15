import pytest
import abc
import dataclasses
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import named_arrays as na
import optika


transformation_parameterization = [
    None,
    na.transformations.TransformationList(
        [
            na.transformations.Cartesian3dTranslation(x=5 * u.mm),
            na.transformations.Cartesian3dRotationZ(53 * u.deg),
            na.transformations.Cartesian3dTranslation(x=6 * u.mm),
        ]
    ),
    na.transformations.Cartesian3dRotationZ(
        na.ScalarLinearSpace(0 * u.deg, 90 * u.deg, axis="transformation", num=3)
    ),
    na.transformations.Cartesian3dRotationZ(
        na.NormalUncertainScalarArray(53 * u.deg, width=5 * u.deg)
    ),
]

kwargs_plot_parameterization = [
    dict(),
    dict(color="red"),
]


class AbstractTestShaped(
    abc.ABC,
):
    def test_shape(self, a: optika.mixins.Shaped):
        result = a.shape
        assert isinstance(result, dict)
        for k in result:
            assert isinstance(k, str)
            assert isinstance(result[k], int)


class AbstractTestPrintable(
    abc.ABC,
):
    @pytest.mark.parametrize(
        argnames="prefix",
        argvalues=[None, "   "],
    )
    def test_to_string(
        self,
        a: optika.mixins.Printable,
        prefix: None | str,
    ):
        result = a.to_string(prefix=prefix)
        assert isinstance(result, str)


class AbstractTestPlottable(abc.ABC):
    @pytest.mark.parametrize(
        argnames="ax",
        argvalues=[
            None,
            plt.subplots()[1],
        ],
    )
    @pytest.mark.parametrize(
        argnames="transformation",
        argvalues=transformation_parameterization[:2],
    )
    class TestPlot(abc.ABC):
        def test_plot(
            self,
            a: optika.mixins.Plottable,
            ax: None | matplotlib.axes.Axes | na.ScalarArray,
            transformation: None | na.transformations.AbstractTransformation,
        ):
            result = a.plot(
                ax=ax,
                transformation=transformation,
            )

            if ax is None or ax is np._NoValue:
                ax_normalized = plt.gca()
            else:
                ax_normalized = ax
            ax_normalized = na.as_named_array(ax_normalized)

            for index in ax_normalized.ndindex():
                assert ax_normalized[index].ndarray.has_data()

            assert isinstance(result, (na.AbstractScalar, dict))


class AbstractTestTransformable:
    def test_transformation(self, a: optika.mixins.Transformable):
        t = a.transformation
        if t is not None:
            assert isinstance(t, na.transformations.AbstractTransformation)


class AbstractTestTranslatable(
    AbstractTestTransformable,
):
    def test_translation(
        self,
        a: optika.mixins.Translatable,
    ):
        result = a.translation
        assert na.unit_normalized(result).is_equivalent(u.mm)


@dataclasses.dataclass(eq=False, repr=False)
class Translatable(
    optika.mixins.Translatable,
):
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        Translatable(),
        Translatable(na.Cartesian3dVectorArray(1, 2, 3) * u.mm),
    ],
)
class TestTranslatable(
    AbstractTestTranslatable,
):
    pass


class AbstractTestPitchable(
    AbstractTestTransformable,
):
    def test_pitch(
        self,
        a: optika.mixins.Pitchable,
    ):
        result = a.pitch
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)


@dataclasses.dataclass(eq=False, repr=False)
class Pitchable(
    optika.mixins.Pitchable,
):
    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        Pitchable(),
        Pitchable(10 * u.deg),
    ],
)
class TestPitchable(
    AbstractTestPitchable,
):
    pass


class AbstractTestYawable(
    AbstractTestTransformable,
):
    def test_yaw(
        self,
        a: optika.mixins.Yawable,
    ):
        result = a.yaw
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)


@dataclasses.dataclass(eq=False, repr=False)
class Yawable(
    optika.mixins.Yawable,
):
    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        Yawable(),
        Yawable(10 * u.deg),
    ],
)
class TestYawable(
    AbstractTestYawable,
):
    pass


class AbstractTestRollable(
    AbstractTestTransformable,
):
    def test_roll(
        self,
        a: optika.mixins.Rollable,
    ):
        result = a.roll
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)


@dataclasses.dataclass(eq=False, repr=False)
class Rollable(
    optika.mixins.Rollable,
):
    roll: u.Quantity | na.AbstractScalar = 0 * u.deg


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        Rollable(),
        Rollable(10 * u.deg),
    ],
)
class TestRollable(
    AbstractTestRollable,
):
    pass
