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
    ],
)
class TestChemical(
    AbstractTestAbstractChemical,
):
    pass
