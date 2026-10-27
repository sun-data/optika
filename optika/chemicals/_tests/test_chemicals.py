import pytest
import abc
import pathlib
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins

_wavelength = [
    500 * u.nm,
    na.geomspace(10, 10000, axis="wavelength", num=101) * u.AA,
    na.NormalUncertainScalarArray(500 * u.nm, width=10 * u.nm),
]


class AbstractTestAbstractChemical(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestReplaceable,
    test_mixins.AbstractTestShaped,
    abc.ABC,
):
    def test_formula(self, a: optika.chemicals.AbstractChemical):
        result = a.formula
        assert isinstance(result, (str, na.AbstractScalar))

    def test_formula_latex(self, a: optika.chemicals.AbstractChemical):
        result = a.formula_latex
        assert isinstance(result, (str, na.AbstractScalar))

    def test_is_amorphous(self, a: optika.chemicals.AbstractChemical):
        result = a.is_amorphous
        assert np.issubdtype(na.get_dtype(result), bool)

    def test_table(self, a: optika.chemicals.AbstractChemical):
        result = a.table
        if result is not None:
            assert isinstance(result, str)

    def test_file_nk(self, a: optika.chemicals.AbstractChemical):
        result = a.file_nk
        for index in result.ndindex():
            assert isinstance(result[index].ndarray, pathlib.Path)
            assert result[index].ndarray.exists()

    @pytest.mark.parametrize("wavelength", _wavelength)
    def test_index_refraction(
        self,
        a: optika.chemicals.AbstractChemical,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.index_refraction(wavelength)
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result > 0)

    @pytest.mark.parametrize("wavelength", _wavelength)
    def test_wavenumber(
        self,
        a: optika.chemicals.AbstractChemical,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.wavenumber(wavelength)
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result >= 0)

    @pytest.mark.parametrize("wavelength", _wavelength)
    def test_absorption(
        self,
        a: optika.chemicals.AbstractChemical,
        wavelength: u.Quantity | na.AbstractScalar,
    ):
        result = a.absorption(wavelength)
        assert isinstance(result, na.AbstractScalar)
        assert np.all(result >= 0)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.chemicals.Chemical(
            formula=formula,
            is_amorphous=is_amorphous,
            table=table,
        )
        for formula in [
            "Si",
            "SiO2",
            na.ScalarArray(np.array(["Si", "SiO2"]), axes="layer"),
        ]
        for is_amorphous in [
            False,
            True,
        ]
        for table in [None, "palik"]
    ]
    + [
        optika.chemicals.Chemical(
            formula=formula,
            table=table,
        )
        for formula, table in [
            ("Zr", "lynch"),
            ("Zr", "palm"),
            ("Zr", "querry"),
            ("ZrO2", "synowicki"),
        ]
    ],
)
class TestChemical(
    AbstractTestAbstractChemical,
):
    pass


@pytest.mark.parametrize(
    argnames="energy,epsilon",
    argvalues=[
        (0.10 * u.eV, -812.14 + 360.46j),
        (2.50 * u.eV, -5.24 + 9.12j),
        (10.50 * u.eV, 0.63 + 2.77j),
        (30.00 * u.eV, 0.55 + 0.59j),
    ],
)
def test_zr_lynch(
    energy: u.Quantity,
    epsilon: complex,
):
    """
    The table of polycrystalline Zr from Lynch et al. (1975) gives the
    dielectric function tabulated by Weaver et al. (1981), whose printed
    column of k does not agree with it.
    """
    wavelength = energy.to(u.AA, equivalencies=u.spectral())
    n = optika.chemicals.Chemical("Zr", table="lynch").n(wavelength)
    assert np.isclose(n.ndarray**2, epsilon, atol=1e-3)


@pytest.mark.parametrize(
    argnames="formula,table,wavelength",
    argvalues=[
        ("Zr", "llnl_cxro", 100 * u.AA),
        ("Zr", "windt", 470 * u.AA),
        ("Zr", "windt", 1000 * u.AA),
        ("Zr", "lynch", 1500 * u.AA),
        ("Zr", "querry", 500 * u.nm),
        ("Zr", "querry", 50 * u.um),
        ("ZrO2", "llnl_cxro", 100 * u.AA),
        ("ZrO2", "synowicki", 500 * u.nm),
    ],
)
def test_zr_default(
    formula: str,
    table: str,
    wavelength: u.Quantity,
):
    """
    The default tables of Zr and ZrO2 follow each of the tables they are
    concatenated from, over its range.
    """
    result = optika.chemicals.Chemical(formula).n(wavelength)
    expected = optika.chemicals.Chemical(formula, table=table).n(wavelength)
    assert np.isclose(result.ndarray, expected.ndarray)
