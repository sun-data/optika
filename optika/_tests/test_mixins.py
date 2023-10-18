import pytest
import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
import optika.mixins


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
