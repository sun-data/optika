from __future__ import annotations
from typing import Iterator
from typing_extensions import Self
import abc
import dataclasses
import copy
import astropy.units as u  # type: ignore[import]
import named_arrays as na  # type: ignore[import]
import optika.mixins

__all__ = [
    "AbstractTransform",
    "Translation",
    "AbstractRotation",
    "RotationX",
    "RotationY",
    "RotationZ",
    "TransformList",
    "Transformable",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractTransform(
    optika.mixins.Printable,
):
    """
    An interface for an arbitrary vector transform
    """

    @property
    @abc.abstractmethod
    def explicit(self) -> Transform:
        """
        resolve this transform into its explicit form
        """

    @property
    def matrix(self) -> na.AbstractCartesian3dMatrixArray:
        """
        The matrix component of the transformation.

        This can contribute to scaling, shearing, or rotating an instance of
        :class:`named_arrays.AbstractVectorArray`
        """
        return self.explicit.matrix

    @property
    def vector(self) -> na.AbstractCartesian3dVectorArray:
        """
        The vector component of the transformation

        This can contribute to translation of an instance of
        :class:`named_arrays.AbstractVectorArray`.
        """
        return self.explicit.vector

    def __call__(
        self,
        value: na.AbstractCartesian3dVectorArray,
        use_matrix: bool = True,
        use_vector: bool = True,
    ) -> na.AbstractCartesian3dVectorArray:
        """
        evaluate the transformation

        Parameters
        ----------
        value
            the vector to transform
        use_matrix
            a flag controlling whether the matrix component of the transformation
            is applied
        use_vector
            a flag controlling whether the vector component of the transformation
            is applied
        """

        explicit = self.explicit
        if use_matrix:
            matrix = explicit.matrix
            if matrix is not None:
                value = matrix @ value
        if use_vector:
            vector = explicit.vector
            if vector is not None:
                value = value + vector
        return value

    @abc.abstractmethod
    def __invert__(self: Self) -> Self:
        pass

    @property
    def inverse(self: Self) -> Self:
        """
        A new transformation that is the inverse of this transformation.
        """
        return self.__invert__()

    @abc.abstractmethod
    def __matmul__(self, other: AbstractTransform) -> AbstractTransform:
        """
        Compose multiple transformations into a single transformation.

        Parameters
        ----------
        other
            another transformation to compose with this one

        Examples
        --------

        Compose two transformations

        .. jupyter-execute::

            import astropy.units as u
            import named_arrays as na
            import optika

            t1 = optika.transforms.Translation(na.Cartesian3dVectorArray(x=5) * u.mm)
            t2 = optika.transforms.RotationZ(50 * u.deg)

            t_composed = t1 @ t2
            t_composed

        use the composed transformation to transform a vector

        .. jupyter-execute::

            v = na.Cartesian3dVectorArray(1, 2, 3) * u.mm

            t_composed(v)

        transform the same vector by applying each transformation separately
        and note that it's the same result as using the composed transformation

        .. jupyter-execute::

            t1(t2(v))
        """
        if isinstance(self, AbstractTransform):
            if isinstance(self, TransformList):
                self_normalized = self
            else:
                self_normalized = TransformList([self])
        else:
            return NotImplemented

        if isinstance(other, AbstractTransform):
            if isinstance(other, TransformList):
                other_normalized = other
            else:
                other_normalized = TransformList([other])
        else:
            return NotImplemented

        return self_normalized + other_normalized


@dataclasses.dataclass
class Transform(
    AbstractTransform,
):
    matrix: None | na.AbstractCartesian3dMatrixArray = None
    vector: None | na.AbstractCartesian3dVectorArray = None


@dataclasses.dataclass(eq=False, repr=False)
class Translation(AbstractTransform):
    """
    A translation-only vector transformation.

    Examples
    --------

    Translate a vector using a single transformation

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import optika

        displacement = na.Cartesian3dVectorArray(
            x=12 * u.mm,
            y=12 * u.mm,
        )

        transform = optika.transforms.Translation(displacement)

        square = na.Cartesian3dVectorArray(
            x=na.ScalarArray([-10, 10, 10, -10, -10] * u.mm, axes="vertex"),
            y=na.ScalarArray([-10, -10, 10, 10, -10] * u.mm, axes="vertex"),
        )

        square_transformed = transform(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed, label="translated");
            plt.legend();

    |

    Translate a vector using an array of transformations

    .. jupyter-execute::

        displacement_2 = na.Cartesian3dVectorArray(
            x=na.ScalarArray([12, -12] * u.mm, axes="transform"),
            y=9 * u.mm,
        )

        transform_2 = optika.transforms.Translation(displacement_2)

        square_transformed_2 = transform_2(square)

        with astropy.visualization.quantity_support():
            plt.figure();
            plt.gca().set_aspect("equal");
            na.plt.plot(square, label="original");
            na.plt.plot(square_transformed_2, axis="vertex", label="translated");
            plt.legend();
    """

    displacement: na.Cartesian3dVectorArray = dataclasses.field(
        default_factory=na.Cartesian3dVectorArray
    )
    """The magnitude of the translation along each axis"""

    @property
    def explicit(self) -> Transform:
        return Transform(
            vector=self.displacement,
        )

    def __invert__(self: Self) -> Self:
        return type(self)(displacement=-self.displacement)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractRotation(AbstractTransform):
    """
    Any arbitrary rotation
    """

    angle: na.ScalarLike
    """The angle of the rotation"""

    def __invert__(self: Self) -> Self:
        return type(self)(angle=-self.angle)


@dataclasses.dataclass(eq=False, repr=False)
class RotationX(AbstractRotation):
    """
    A rotation about the :math:`x` axis
    """

    @property
    def explicit(self) -> Transform:
        return Transform(
            matrix=na.Cartesian3dXRotationMatrixArray(self.angle),
        )


@dataclasses.dataclass(eq=False, repr=False)
class RotationY(AbstractRotation):
    """
    A rotation about the :math:`y` axis
    """

    @property
    def explicit(self) -> Transform:
        return Transform(
            matrix=na.Cartesian3dYRotationMatrixArray(self.angle),
        )


@dataclasses.dataclass(eq=False, repr=False)
class RotationZ(AbstractRotation):
    """
    A rotation about the :math:`z` axis
    """

    @property
    def explicit(self) -> Transform:
        return Transform(
            matrix=na.Cartesian3dZRotationMatrixArray(self.angle),
        )


@dataclasses.dataclass(eq=False, repr=False)
class TransformList(
    AbstractTransform,
    optika.mixins.DataclassList[AbstractTransform],
):
    """
    A sequence of transformations
    """

    intrinsic: bool = True

    @property
    def extrinsic(self) -> bool:
        return not self.intrinsic

    @property
    def transforms(self) -> Iterator[AbstractTransform]:
        if self.intrinsic:
            return reversed(list(self))
        else:
            return iter(self)

    @property
    def explicit(self) -> Transform:
        matrix = na.Cartesian3dIdentityMatrixArray()
        vector = na.Cartesian3dVectorArray()

        for transform in reversed(list(self.transforms)):
            if transform is not None:
                transform = transform.explicit
                if transform.matrix is not None:
                    matrix = matrix @ transform.matrix
                if transform.vector is not None:
                    vector = matrix @ transform.vector + vector

        return Transform(
            matrix=matrix,
            vector=vector,
        )

    def __invert__(self: Self) -> Self:
        other = copy.copy(self)
        other.data = []
        for transform in self:
            if transform is not None:
                transform = transform.__invert__()
            other.append(transform)
        other.reverse()
        return other


@dataclasses.dataclass(eq=False, repr=False)
class Transformable(abc.ABC):
    @property
    @abc.abstractmethod
    def transform(self) -> None | AbstractTransform:
        """
        the coordinate transformation between the global coordinate system
        and this object's local coordinate system
        """
