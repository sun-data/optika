import abc
import functools
import pathlib
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "AbstractChemical",
    "Chemical",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractChemical(
    optika.mixins.Printable,
):
    """
    Interface defining the optical constants for a given chemical.
    """

    @property
    @abc.abstractmethod
    def formula(self) -> str:
        """
        the `empirical formula <https://en.wikipedia.org/wiki/Empirical_formula>`_
        of the chemical compound.

        For example, water would be expressed as ``"H2O"``
        and hydrogen peroxide would be expressed as ``"H2O2"``.
        """

    @property
    @abc.abstractmethod
    def is_amorphous(self) -> bool:
        """
        Boolean flag controlling whether the chemical is amorphous
        or crystalline.
        """

    @property
    @abc.abstractmethod
    def table(self) -> None | str:
        """
        Name of the table of chemical constants.
        """

    @property
    def file_nk(self) -> na.ScalarArray:
        """
        The path to the :cite:t:`Windt1998` file containing the
        index of refaction, :math:`n`, and the wavenumber, :math:`k`.
        """
        formula = na.as_named_array(self.formula)
        is_amorphous = na.as_named_array(self.is_amorphous)
        table = na.as_named_array(self.table)

        shape = na.shape_broadcasted(formula, is_amorphous, table)

        formula = formula.broadcast_to(shape)
        is_amorphous = is_amorphous.broadcast_to(shape)
        table = table.broadcast_to(shape)

        result = na.ScalarArray.empty(shape=shape, dtype=pathlib.Path)

        path_base = pathlib.Path(__file__).parent / "nk"
        for index in result.ndindex():
            file = f"{formula[index].ndarray}"

            if table[index].ndarray is not None:
                file = f"{file}_{table[index].ndarray}"

            if is_amorphous[index]:
                file = f"a-{file}"

            file = f"{file}.nk"

            result[index] = path_base / file

        return result

    @functools.cached_property
    def _wavelength_n_k(self) -> tuple[na.AbstractScalar, ...]:
        """
        Return arrays of wavelength, :math:`n`, and :math:`k` from :attr:`file_nk`.
        """
        file_nk = self.file_nk
        shape_base = file_nk.shape

        for i, index in enumerate(file_nk.ndindex()):
            file_nk_index = file_nk[index].ndarray

            skip_header = 0
            with open(file_nk_index, "r") as f:
                for line in f:
                    if line.startswith(";"):
                        skip_header += 1
                    else:
                        break

            wavelength_index, n_index, k_index = np.genfromtxt(
                fname=file_nk_index,
                skip_header=skip_header,
                unpack=True,
            )

            wavelength_index = wavelength_index << u.AA
            wavelength_index = na.ScalarArray(wavelength_index, axes="wavelength")
            n_index = na.ScalarArray(n_index, axes="wavelength")
            k_index = na.ScalarArray(k_index, axes="wavelength")

            shape_index = na.shape_broadcasted(wavelength_index, n_index, k_index)
            shape = na.broadcast_shapes(shape_base, shape_index)

            if i == 0:
                wavelength = na.ScalarArray.empty(shape) << u.AA
                n = na.ScalarArray.empty(shape)
                k = na.ScalarArray.empty(shape)

            wavelength[index] = wavelength_index
            n[index] = n_index
            k[index] = k_index

        return wavelength, n, k

    @property
    def index_refraction(self) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
        """
        The index of refraction of this material as a function of wavelength.
        """
        wavelength, n, k = self._wavelength_n_k
        return na.FunctionArray(inputs=wavelength, outputs=n)

    @property
    def wavenumber(self) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
        """
        The wavenumber of this material as a function of wavelength.
        """
        wavelength, n, k = self._wavelength_n_k
        return na.FunctionArray(inputs=wavelength, outputs=k)


@dataclasses.dataclass(eq=False, repr=False)
class Chemical(
    AbstractChemical,
):
    """
    An object that represents the optical properties of a chemical.

    Uses the tabulated optical constants from :cite:t:`Windt1998`.

    Examples
    --------

    Plot the indices of refractions of silicon and silicon dioxide.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na
        import optika

        si = optika.chemicals.Chemical("Si")
        sio2 = optika.chemicals.Chemical("SiO2")

        n_si = si.index_refraction
        n_sio2 = sio2.index_refraction

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(n_si.inputs, n_si.outputs, label="silicon");
        na.plt.plot(n_sio2.inputs, n_sio2.outputs, label="silicon dioxide");
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({n_si.inputs.unit:latex_inline})");
        ax.set_ylabel("index of refraction");
        ax.legend();

    Plot the wavenumbers of silicon and silicon dioxide

    .. jupyter-execute::

        k_si = si.wavenumber
        k_sio2 = sio2.wavenumber

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(k_si.inputs, k_si.outputs, label="silicon");
        na.plt.plot(k_sio2.inputs, k_sio2.outputs, label="silicon dioxide");
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({k_si.inputs.unit:latex_inline})");
        ax.set_ylabel("wavenumber");
        ax.legend();
    """

    formula: str = dataclasses.MISSING
    """
    the `empirical formula <https://en.wikipedia.org/wiki/Empirical_formula>`_
    of the chemical compound.

    For example, water would be expressed as ``"H2O"``
    and hydrogen peroxide would be expressed as ``"H2O2"``.
    """

    is_amorphous: bool = False
    """
    Boolean flag controlling whether the chemical is amorphous
    or crystalline.
    """

    table: None | str = None
    """
    Name of the table of chemical constants.

    Common options are ``"palik"``, ``"llnl"``, and ``"windt"``.
    The database is the same as the IMD code :cite:p:`Windt1998`.
    The default value, :obj:`None`, usually means concatenating the
    tables in :cite:t:`Palik1997` and :cite:t:`Henke1993`.
    """
