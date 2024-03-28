#!/usr/bin/env python3
""" Unit tests for gfactor.py. 

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.gfactor import (
    gfactor,
    _lf,
    _largest_common_cube,
    _make_cube_free,
    _is_cube_free,
    _divide,
    _get_variables_in_formula,
    _most_common_condition,
)
from sympy import sympify, Add, Mul, Symbol
import unittest


class GetVariablesInFormulaTest(unittest.TestCase):

    def test_function(self):

        formula = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
        )

        var_list = sorted(_get_variables_in_formula(formula), key=lambda s: s.name)
        self.assertEqual(len(var_list), 5)
        self.assertEqual(
            var_list,
            [Symbol("v1"), Symbol("v1"), Symbol("v2"), Symbol("v3"), Symbol("v4")],
        )


class MostCommonConditionTest(unittest.TestCase):

    def test_function(self):
        formula = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
        )

        self.assertEqual(_most_common_condition(formula), Symbol("v1"))

        formula = Add(
            Symbol("v1"), Symbol("v2"), Symbol("v3"), Symbol("v4"), Symbol("v5")
        )

        self.assertEqual(_most_common_condition(formula), None)


class DivideTest(unittest.TestCase):

    def test_function(self):
        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 6)}

        formula = sympify("v1*v2*v3 + v1*v2*v4 + v5", locals=symbols)
        divisor = sympify("v1*v2")

        quotient, remainder = _divide(formula, divisor)

        ex_quotient = sympify("v3 + v4", locals=symbols)
        ex_remainder = sympify("v5", locals=symbols)

        self.assertEqual(quotient, ex_quotient)
        self.assertEqual(remainder, ex_remainder)


class IsCubeFreeTest(unittest.TestCase):

    def test_function(self):

        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 6)}

        # Test 1: No cubes
        formula = sympify("v1*v2 + v2*v3 + v3*v4 + v4*v5 + v5*v1", locals=symbols)
        self.assertEqual(_is_cube_free(formula), True)
        formula = sympify("v1 + v2 + v3 + v4 + v5", locals=symbols)
        self.assertEqual(_is_cube_free(formula), True)

        # Test 2: Single symbol as cube
        formula = sympify("v1*v2 + v1*v3 + v1*v4 + v1*v5", locals=symbols)
        self.assertEqual(_is_cube_free(formula), False)

        # Test 3: Multi-symbol cube
        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2", locals=symbols)
        self.assertEqual(_is_cube_free(formula), False)


class LargestCommonCubeTest(unittest.TestCase):

    def test_function(self):

        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 6)}

        # Test 1: No cubes
        formula = sympify("v1*v2 + v2*v3 + v3*v4 + v4*v5 + v5*v1", locals=symbols)
        self.assertEqual(_largest_common_cube(formula), None)
        formula = sympify("v1 + v2 + v3 + v4 + v5", locals=symbols)
        self.assertEqual(_largest_common_cube(formula), None)

        # Test 2: Single symbol as cube
        formula = sympify("v1*v2 + v1*v3 + v1*v4 + v1*v5 + v1", locals=symbols)
        self.assertEqual(_largest_common_cube(formula), sympify("v1", locals=symbols))

        # Test 3: Multi-symbol cube
        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2", locals=symbols)
        self.assertEqual(
            _largest_common_cube(formula), sympify("v1*v2", locals=symbols)
        )


class MakeCubeFreeTest(unittest.TestCase):

    def test_function(self):
        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 6)}

        # Test 1: There is a common cube
        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2", locals=symbols)
        cube_free = _make_cube_free(formula)
        self.assertEqual(cube_free, sympify("v3 + v4 + v5", locals=symbols))

        # Test 2: There is no common cube
        formula = sympify("v1*v2 + v2*v3 + v3*v4 + v4*v5 + v5*v1", locals=symbols)
        cube_free = _make_cube_free(formula)
        self.assertEqual(cube_free, formula)


class LFTest(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        self.fail()


class GFactorTest(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        self.fail()


if __name__ == "__main__":
    unittest.main()
