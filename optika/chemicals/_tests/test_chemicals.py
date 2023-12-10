import pathlib

import pytest
import abc
import numpy as np
import astropy.units as u
import named_arrays as na
import optika


class AbstractTestAbstractChemical(
    abc.ABC,
):
    def test_formula(self, a: optika.chemicals.AbstractChemical):
        result = a.formula
        assert isinstance(result, str)

    def test_density(self, a: optika.chemicals.AbstractChemical):
        result = a.density
        assert np.issubdtype(na.get_dtype(result), float)
        assert na.unit(result).is_equivalent(u.g / u.cm**3)

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

    def test_index_refraction(self, a: optika.chemicals.AbstractChemical):
        result = a.index_refraction
        assert isinstance(result, na.AbstractFunctionArray)

    def test_wavenumber(self, a: optika.chemicals.AbstractChemical):
        result = a.wavenumber
        assert isinstance(result, na.AbstractFunctionArray)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        optika.chemicals.Chemical(
            formula=formula,
            density=density,
            is_amorphous=is_amorphous,
            table=table,
        )
        for formula in ["Si", "SiO2"]
        for density in [None]
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
