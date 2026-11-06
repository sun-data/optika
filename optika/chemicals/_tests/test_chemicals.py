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
            ("Sc", "larruquert"),
            ("Zr", "lynch"),
            ("Zr", "palm"),
            ("Zr", "querry"),
            ("ZrO2", "synowicki"),
            ("SiO2", "llnl_cxro_rodriguez"),
        ]
    ]
    + [
        optika.chemicals.Chemical(
            formula="SiO2",
            is_amorphous=True,
            table=table,
        )
        for table in ["franta", "franta300"]
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
    argnames="wavelength,index",
    argvalues=[
        (12.39842 * u.AA, 0.99947778 + 0.0001487j),
        (486.2125 * u.AA, 1.0165236 + 0.02671593j),
        (495.93675 * u.AA, 1.0052265 + 0.02799917j),
        (1897.81921 * u.AA, 1.1198012 + 1.154793j),
    ],
)
def test_sc_larruquert(
    wavelength: u.Quantity,
    index: complex,
):
    """
    The table of Sc from Larruquert et al. (2004) and Fernandez-Perea et al.
    (2006) gives the rows of the refractiveindex.info database, including
    those between 477 and 521 angstroms, where the table of
    Uspenskii et al. (2003) has no data.
    """
    n = optika.chemicals.Chemical("Sc", table="larruquert").n(wavelength)
    assert np.isclose(n.ndarray, index)


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


@pytest.mark.parametrize(
    argnames="formula,is_amorphous,expected,wavelength",
    argvalues=[
        ("Sc", False, dict(table="llnl_cxro"), 10 * u.AA),
        ("Sc", False, dict(table="larruquert"), 100 * u.AA),
        ("Sc", False, dict(table="larruquert"), 499.4 * u.AA),
        ("Sc", False, dict(table="larruquert"), 1500 * u.AA),
        ("Si", True, dict(is_amorphous=True, table="llnl_cxro"), 100 * u.AA),
        ("Si", True, dict(is_amorphous=True, table="palik"), 500 * u.nm),
        ("SiO2", True, dict(table="llnl_cxro_rodriguez"), 100 * u.AA),
        ("SiO2", True, dict(table="rodriguez"), 465.2 * u.AA),
        ("SiO2", True, dict(is_amorphous=True, table="palik"), 1216 * u.AA),
        ("SiO2", True, dict(is_amorphous=True, table="palik"), 500 * u.nm),
    ],
)
def test_default(
    formula: str,
    is_amorphous: bool,
    expected: dict,
    wavelength: u.Quantity,
):
    """
    The default tables of Sc, amorphous Si and amorphous SiO2 follow each of
    the tables they are concatenated from, over its range.
    """
    result = optika.chemicals.Chemical(formula, is_amorphous=is_amorphous)
    expected = optika.chemicals.Chemical(formula, **expected)
    assert np.isclose(result.n(wavelength).ndarray, expected.n(wavelength).ndarray)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[469.977 * u.AA, 600.097 * u.AA, 729.294 * u.AA],
)
def test_a_si_scaled(
    wavelength: u.Quantity,
):
    """
    From 430 to 729 angstroms the default table of amorphous Si is crystalline
    Si with epsilon - 1 scaled to the density of the CXRO table of amorphous
    Si, 2.189 g/cm3.
    """
    n = optika.chemicals.Chemical("Si").n(wavelength).ndarray
    expected = np.sqrt(1 + 2.189 / 2.33 * (np.square(n) - 1))
    result = optika.chemicals.Chemical("Si", is_amorphous=True).n(wavelength)
    assert np.isclose(result.ndarray, expected, rtol=1e-5)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(is_amorphous=True),
        dict(table="llnl_cxro_rodriguez"),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[100 * u.AA, 250 * u.AA, 297.3 * u.AA],
)
def test_sio2_scaled(
    kwargs: dict,
    wavelength: u.Quantity,
):
    """
    Below 300 angstroms the default table of amorphous SiO2 and the table of
    the oxide of silicon sensors are the CXRO table of amorphous SiO2 with
    1 - n and k scaled from 2.19 to 2.28 g/cm3, the density of the films of
    Rodriguez-de Marcos et al. (2016) whose table follows. Until this was so,
    the table of the oxide of silicon sensors held the CXRO table of
    crystalline SiO2, at 2.65 g/cm3.
    """
    cxro = optika.chemicals.Chemical("SiO2", is_amorphous=True, table="llnl_cxro")
    expected = 1 - (1 - cxro.n(wavelength).ndarray) * 2.28 / 2.19
    result = optika.chemicals.Chemical("SiO2", **kwargs).n(wavelength)
    assert np.isclose(result.ndarray, expected, rtol=1e-6)


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[299.714 * u.AA, 304 * u.AA, 1216 * u.AA, 1 * u.um],
)
def test_sio2_llnl_cxro_rodriguez(
    wavelength: u.Quantity,
):
    """
    From 300 angstroms up, the table of the oxide of silicon sensors is the
    table of Rodriguez-de Marcos et al. (2016).
    """
    result = optika.chemicals.Chemical("SiO2", table="llnl_cxro_rodriguez")
    expected = optika.chemicals.Chemical("SiO2", table="rodriguez")
    assert result.n(wavelength).ndarray == expected.n(wavelength).ndarray


@pytest.mark.parametrize(
    argnames="formula,wavelength_below,wavelength_above",
    argvalues=[
        ("Si", 423.15444 * u.AA, 430.038 * u.AA),
        ("SiO2", 297.30366 * u.AA, 299.714 * u.AA),
        ("SiO2", 549.902 * u.AA, 550.044 * u.AA),
    ],
)
def test_amorphous_default_joins(
    formula: str,
    wavelength_below: u.Quantity,
    wavelength_above: u.Quantity,
):
    """
    The default tables of amorphous Si and amorphous SiO2 have no step in k
    where their sources meet in the EUV. Until the tables were rebuilt, k of
    amorphous Si rose by a factor of 3.9 from 423 to 428 angstroms, and k of
    amorphous SiO2 rose by 6% from 297 to 300 angstroms, where its density
    changed from 2.19 to 2.28 g/cm3.
    """
    chemical = optika.chemicals.Chemical(formula, is_amorphous=True)
    below = chemical.n(wavelength_below).ndarray
    above = chemical.n(wavelength_above).ndarray
    assert np.isclose(above.real, below.real, rtol=0.02)
    assert np.isclose(above.imag, below.imag, rtol=0.05)
