from __future__ import annotations
from typing import Any
import abc
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.axes
import astropy.units as u
import named_arrays as na

__all__ = [
    "Shaped",
    "Printable",
    "Plottable",
    "Transformable",
    "Translatable",
    "Pitchable",
    "Yawable",
    "Rollable",
]


@dataclasses.dataclass(repr=False)
class Shaped(abc.ABC):

    @property
    @abc.abstractmethod
    def shape(self) -> dict[str, int]:
        """
        The array shape of this object.
        """


@dataclasses.dataclass(repr=False)
class Printable:
    @classmethod
    def _val_to_string(
        cls,
        val: Any,
        pre: str,
        tab: str,
        field_str: str,
    ) -> str:
        if isinstance(val, Printable):
            val_str = val.to_string(prefix=f"{pre}{tab}")
        elif isinstance(val, na.AbstractArray):
            val_str = val.to_string(prefix=f"{pre}{tab}")
        elif isinstance(val, np.ndarray):
            val_str = np.array2string(
                a=val,
                separator=", ",
                prefix=field_str,
            )
            if isinstance(val, u.Quantity):
                val_str = f"{val_str} {val.unit}"
        elif isinstance(val, list):
            val_str = "[\n"
            for v in val:
                val_str += f"{pre}{tab}{tab}"
                val_str += cls._val_to_string(
                    val=v,
                    pre=f"{pre}{tab}",
                    tab=tab,
                    field_str=f"{pre}{tab}",
                )
                val_str += ",\n"
            val_str += f"{pre}{tab}]"
        else:
            val_str = repr(val)

        return val_str

    def to_string(
        self,
        prefix: None | str = None,
    ) -> str:
        """
        Public-facing version of the ``__repr__`` method that allows for
        defining a prefix string, which can be used to calculate how much
        whitespace to add to the beginning of each line of the result.

        Parameters
        ----------
        prefix
            an optional string, the length of which is used to calculate how
            much whitespace to add to the result.
        """

        fields = dataclasses.fields(self)

        delim_field = "\n"
        pre = " " * len(prefix) if prefix is not None else ""
        tab = " " * 4

        result_fields = ""
        for i, f in enumerate(fields):
            field_str = f"{pre}{tab}{f.name}="
            val = getattr(self, f.name)
            val_str = self._val_to_string(
                val=val,
                pre=pre,
                tab=tab,
                field_str=field_str,
            )
            field_str += val_str
            field_str += f",{delim_field}"
            result_fields += field_str

        if result_fields:
            result_fields = f"\n{result_fields}{pre}"

        result = f"{self.__class__.__qualname__}({result_fields})"

        return result

    def __repr__(self):
        return self.to_string()


@dataclasses.dataclass(eq=False, repr=False)
class Plottable(abc.ABC):
    @property
    @abc.abstractmethod
    def kwargs_plot(self) -> None | dict:
        """
        Extra keyword arguments that will be used in the call to
        :func:`named_arrays.plt.plot` within the :meth:`plot` method.
        """

    @abc.abstractmethod
    def plot(
        self,
        ax: None | matplotlib.axes.Axes | na.ScalarArray[npt.NDArray] = None,
        transformation: None | na.transformations.AbstractTransformation = None,
        components: None | tuple[str, ...] = None,
        **kwargs,
    ) -> na.AbstractScalar | dict[str, na.AbstractScalar]:
        """
        Plot the selected components onto the given axes.

        Parameters
        ----------
        ax
            The matplotlib axes to plot onto
        transformation
            Any extra transformations to apply to the coordinate system before
            plotting
        components
            Which 3d components to plot, helpful if plotting in 2d.
        kwargs
            Additional keyword arguments that will be passed along to
            :func:`named_arrays.plt.plot()`
        """


@dataclasses.dataclass(eq=False, repr=False)
class Transformable(abc.ABC):
    @property
    @abc.abstractmethod
    def transformation(self) -> None | na.transformations.AbstractTransformation:
        """
        the coordinate transformation between the global coordinate system
        and this object's local coordinate system
        """
        return na.transformations.IdentityTransformation()


@dataclasses.dataclass(eq=False, repr=False)
class Translatable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def translation(self) -> u.Quantity | na.AbstractScalar | na.AbstractVectorArray:
        """translate the coordinate system"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        translation = na.asanyarray(self.translation, like=na.Cartesian3dVectorArray())
        return super().transformation @ na.transformations.Translation(translation)


@dataclasses.dataclass(eq=False, repr=False)
class Pitchable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def pitch(self) -> u.Quantity | na.ScalarLike:
        """pitch angle of this object"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dRotationX(
            angle=self.pitch
        )


@dataclasses.dataclass(eq=False, repr=False)
class Yawable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def yaw(self) -> u.Quantity | na.ScalarLike:
        """yaw angle of this object"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dRotationY(
            angle=self.yaw,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Rollable(
    Transformable,
):
    @property
    @abc.abstractmethod
    def roll(self) -> u.Quantity | na.ScalarLike:
        """roll angle of this object"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        return super().transformation @ na.transformations.Cartesian3dRotationZ(
            angle=self.roll
        )
