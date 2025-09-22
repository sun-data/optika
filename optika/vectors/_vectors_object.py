from __future__ import annotations
from typing import TypeVar
from typing_extensions import Self
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from . import AbstractPupilVectorArray, PupilVectorArray
from . import AbstractSceneVectorArray, SceneVectorArray

__all__ = [
    "AbstractObjectVectorArray",
    "ObjectVectorArray",
]


WavelengthT = TypeVar("WavelengthT", bound=na.ScalarLike)
FieldT = TypeVar("FieldT", bound=na.AbstractCartesian2dVectorArray)
PupilT = TypeVar("PupilT", bound=na.AbstractCartesian2dVectorArray)


@dataclasses.dataclass(eq=False, repr=False)
class AbstractObjectVectorArray(
    AbstractPupilVectorArray,
    AbstractSceneVectorArray,
):
    """An interface describing a field position, pupil position, and wavelength."""

    @property
    def type_abstract(self) -> type[AbstractObjectVectorArray]:
        return AbstractObjectVectorArray

    @property
    def type_explicit(self) -> type[ObjectVectorArray]:
        return ObjectVectorArray

    @property
    def type_matrix(self) -> type[na.AbstractMatrixArray]:
        raise NotImplementedError

    def cell_area(
        self,
        axis_wavelength: str,
        axis_field: tuple[str, str],
        axis_pupil: tuple[str, str],
    ) -> na.AbstractScalar:
        r"""
        Compute the 5-dimensional area of each grid cell in units of wavelength
        :math:`\times` area :math:`\times` solid angle.

        This method does not work in general.
        It only works for the special case where the wavelength grid
        is not dependent on the field or pupil position,
        and the field grid does not depend on the pupil position.

        Parameters
        ----------
        axis_wavelength
            The logical axis corresponding to changing wavelength.
        axis_field
            The two logical axes corresponding to changing field position.
        axis_pupil
            the two logical axes corresponding to changing pupil position.
        """

        wavelength = self.wavelength
        field = self.field
        pupil = self.pupil

        shape_wavelength = wavelength.shape
        shape_field = field.shape
        shape_pupil = pupil.shape

        if axis_wavelength not in shape_wavelength:
            raise ValueError(  # pragma: nocover
                f"{axis_wavelength=} must be in {shape_wavelength=}",
            )
        if not set(axis_field).issubset(shape_field):
            raise ValueError(  # pragma: nocover
                f"{axis_field=} must be a subset of {shape_field=}",
            )
        if set(axis_field).intersection(shape_wavelength):
            raise ValueError(  # pragma: nocover
                f"{axis_field=} must not intersect {shape_wavelength=}"
            )
        if not set(axis_pupil).issubset(shape_pupil):
            raise ValueError(  # pragma: nocover
                f"{axis_pupil=} must be a subset of {shape_pupil=}",
            )
        if set(axis_pupil).intersection(shape_wavelength | shape_field):
            raise ValueError(  # pragma: nocover
                f"{axis_pupil=} must not intersect {shape_wavelength=} or {shape_field=}"
            )

        area_wavelength = wavelength.volume_cell(axis_wavelength)

        shape_field = na.broadcast_shapes(
            shape_wavelength,
            shape_field,
        )
        field = field.broadcast_to(shape_field)

        shape_pupil = na.broadcast_shapes(
            shape_wavelength,
            shape_field,
            shape_pupil,
        )
        pupil = pupil.broadcast_to(shape_pupil)

        if na.unit_normalized(field).is_equivalent(u.deg):
            area_field = optika.direction(field).solid_angle_cell(axis_field)
        else:
            area_field = field.volume_cell(axis_field)

        if na.unit_normalized(pupil).is_equivalent(u.deg):
            area_pupil = optika.direction(pupil).solid_angle_cell(axis_pupil)
        else:
            area_pupil = pupil.volume_cell(axis_pupil)

        area_field = np.abs(area_field)
        area_pupil = np.abs(area_pupil)

        area_field = area_field.cell_centers(
            axis=axis_wavelength,
        )

        area_pupil = area_pupil.cell_centers(
            axis=(axis_wavelength,) + axis_field,
        )

        return area_wavelength * area_field * area_pupil


@dataclasses.dataclass(eq=False, repr=False)
class ObjectVectorArray(
    PupilVectorArray,
    SceneVectorArray,
    AbstractObjectVectorArray,
):
    """A vector describing a field position, pupil position, and wavelength."""

    @classmethod
    def from_scalar(
        cls,
        scalar: na.AbstractScalar,
        like: None | Self = None,
    ) -> Self:
        if like is not None:
            return type(like)(
                wavelength=scalar,
                field=scalar,
                pupil=scalar,
            )
        else:
            return cls(
                wavelength=scalar,
                field=scalar,
                pupil=scalar,
            )
