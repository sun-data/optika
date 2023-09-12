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


class AbstractTestTransformable:
    def test_transformation(self, a: optika.mixins.Transformable):
        if a.transformation is not None:
            assert isinstance(a.transformation, na.transformations.AbstractTransformation)
