import pytest
import abc
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
        print(result)
        assert isinstance(result, str)


class AbstractTestTransformable:
    def test_transformation(self, a: optika.mixins.Transformable):
        t = a.transformation
        if t is not None:
            assert isinstance(t, na.transformations.AbstractTransformation)
