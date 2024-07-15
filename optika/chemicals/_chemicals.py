import abc
import re
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
    optika.mixins.Shaped,
):
    """
    Interface defining the optical constants for a given chemical.
    """

    @property
    @abc.abstractmethod
    def formula(self) -> str | na.AbstractScalar:
        """
        the `empirical formula <https://en.wikipedia.org/wiki/Empirical_formula>`_
        of the chemical compound.

        For example, water would be expressed as ``"H2O"``
        and hydrogen peroxide would be expressed as ``"H2O2"``.
        """

    @property
    def formula_latex(self) -> str:
        """
        LaTeX representation of the chemical formula, with appropriate subscripts.
        """
        formula = self.formula
        pattern = r"\d+"
        repl = r"$_\g<0>$"
        if isinstance(formula, na.ScalarArray):
            formula = formula.copy()
            for index in formula.ndindex():
                formula[index] = re.sub(pattern, repl, formula[index].ndarray)
        else:
            formula = re.sub(pattern, repl, formula)
        return formula

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

    def n(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The complex index of refaction of this chemical for a given wavelength
        """
        file_nk = self.file_nk
        shape_base = file_nk.shape

        shape = na.broadcast_shapes(shape_base, wavelength.shape)

        wavelength = na.broadcast_to(wavelength, shape)
        result = np.empty_like(wavelength.value, dtype=complex)

        for i, index in enumerate(file_nk.ndindex()):
            file_nk_index = file_nk[index].ndarray

            skip_header = 0
            with open(file_nk_index, "r") as f:
                for line in f:
                    if line.startswith(";"):
                        skip_header += 1
                    else:
                        break

            wavelength_index, n_index, k_index = np.loadtxt(
                fname=file_nk_index,
                skiprows=skip_header,
                unpack=True,
            )

            wavelength_index = wavelength_index << u.AA
            wavelength_index = na.ScalarArray(wavelength_index, axes="wavelength")
            n_index = na.ScalarArray(n_index, axes="wavelength")
            k_index = na.ScalarArray(k_index, axes="wavelength")

            result[index] = na.interp(
                x=wavelength[index],
                xp=wavelength_index,
                fp=n_index + 1j * k_index,
            )

        return result

    def index_refraction(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The index of refraction of this chemical for the given wavelength.
        """
        return np.real(self.n(wavelength))

    def wavenumber(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ) -> na.AbstractScalar:
        """
        The wavenumber of this chemical for the given wavelength.
        """
        return np.imag(self.n(wavelength))

    def absorption(
        self,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        """
        The absorption coefficient of this chemical for the given wavelength.

        Parameters
        ----------
        wavelength
            The wavelength of light in vacuum for which to compute the
            absorption coefficient.
        """
        return 4 * np.pi * self.wavenumber(wavelength) / wavelength


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
        import astropy.units as u
        import named_arrays as na
        import optika

        si = optika.chemicals.Chemical("Si")
        sio2 = optika.chemicals.Chemical("SiO2")

        wavelength = na.geomspace(10, 10000, axis="wavelength", num=1001) * u.AA

        n_si = si.index_refraction(wavelength)
        n_sio2 = sio2.index_refraction(wavelength)

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, n_si, label="silicon");
        na.plt.plot(wavelength, n_sio2, label="silicon dioxide");
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("index of refraction");
        ax.legend();

    Plot the wavenumbers of silicon and silicon dioxide

    .. jupyter-execute::

        k_si = si.wavenumber(wavelength)
        k_sio2 = sio2.wavenumber(wavelength)

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(wavelength, k_si, label="silicon");
        na.plt.plot(wavelength, k_sio2, label="silicon dioxide");
        ax.set_xscale("log");
        ax.set_xlabel(f"wavelength ({wavelength.unit:latex_inline})");
        ax.set_ylabel("wavenumber");
        ax.legend();
    """

    formula: str | na.AbstractScalar = dataclasses.MISSING
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

    @property
    def shape(self) -> dict[str, int]:
        return na.broadcast_shapes(
            optika.shape(self.formula),
            optika.shape(self.is_amorphous),
            optika.shape(self.table),
        )
