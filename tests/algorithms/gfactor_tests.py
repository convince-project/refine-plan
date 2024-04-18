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
    _get_variable_frequencies,
    _get_random_divisor,
    _quick_divisor,
    _one_zero_kernel,
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


class GetVariablesFrequenciesTest(unittest.TestCase):

    def test_function(self):
        formula = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
        )

        frequencies = _get_variable_frequencies(formula)

        self.assertEqual(len(frequencies), 4)
        self.assertEqual(frequencies[Symbol("v1")], 2)
        self.assertEqual(frequencies[Symbol("v2")], 1)
        self.assertEqual(frequencies[Symbol("v3")], 1)
        self.assertEqual(frequencies[Symbol("v4")], 1)


class GetRandomDivisorTest(unittest.TestCase):

    def test_function(self):

        formula_1 = Symbol("v1")
        formula_2 = Add(
            Mul(Symbol("v1"), Symbol("v2")), Mul(Symbol("v4"), Symbol("v3"))
        )
        formula_3 = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
        )

        divisor = _get_random_divisor(formula_1)
        self.assertEqual(divisor, formula_1)

        divisor = _get_random_divisor(formula_2)
        self.assertEqual(divisor, formula_2)

        symbols = [Symbol("v1"), Symbol("v2"), Symbol("v3"), Symbol("v4")]
        for i in range(100):
            divisor = _get_random_divisor(formula_3)
            self.assertTrue(divisor in symbols)


class OneZeroKernelTest(unittest.TestCase):

    def test_function(self):
        formula_1 = Symbol("v1")
        formula_2 = Add(
            Mul(Symbol("v1"), Symbol("v2")), Mul(Symbol("v4"), Symbol("v3"))
        )
        formula_3 = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
        )
        kernel = _one_zero_kernel(formula_1)
        self.assertEqual(kernel, formula_1)

        kernel = _one_zero_kernel(formula_2)
        self.assertEqual(kernel, formula_2)

        kernel = _one_zero_kernel(formula_3)
        self.assertEqual(kernel, Add(Symbol("v2"), Symbol("v3")))


class QuickDivisorTest(unittest.TestCase):

    def test_function(self):
        formula_1 = Symbol("v1")
        formula_2 = Add(
            Mul(Symbol("v1"), Symbol("v2")), Mul(Symbol("v4"), Symbol("v3"))
        )
        formula_3 = Add(
            Mul(Symbol("v1"), Symbol("v2")),
            Mul(Symbol("v1"), Symbol("v3")),
            Symbol("v4"),
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
        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 6)}

        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2 + v4*v5", locals=symbols)
        divisor = sympify("v1*v2", locals=symbols)

        reduced = _lf(formula, divisor)

        self.assertEqual(str(reduced), "v1*v2*(v3 + v4 + v5) + v4*v5")

        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2", locals=symbols)
        divisor = sympify("v1*v2", locals=symbols)

        reduced = _lf(formula, divisor)

        self.assertEqual(str(reduced), "v1*v2*(v3 + v4 + v5)")


class GFactorTest(unittest.TestCase):

    def test_function(self):
        symbols = {"v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 10)}

        formula = sympify("v3*v2*v1 + v1*v2*v4 + v1*v5*v2 + v4*v5", locals=symbols)

        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v1*v2*(v3 + v4 + v5) + v4*v5")

        # Using all the examples from the BT-Factor paper experiments
        formula = sympify("v5 + v7 + v8", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v5 + v7 + v8")

        formula = sympify("v8 + v2*v3*v5 + v2*v7 + v9", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v2*(v3*v5 + v7) + v8 + v9")

        formula = sympify("v2*v3*v4*v6 + v2*v7 + v8", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v2*(v3*v4*v6 + v7) + v8")

        formula = sympify("v3*v4*v5*v6 + v7", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v3*v4*v5*v6 + v7")

        formula = sympify("v4*v6 + v6*v9 + v7 + v8", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v6*(v4 + v9) + v7 + v8")

        formula = sympify("v10 + v2*v3*v4*v5 + v2*v7 + v8 + v9", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v10 + v2*(v3*v4*v5 + v7) + v8 + v9")

        formula = sympify("v1 + v2", locals=symbols)
        reduced = gfactor(formula)
        self.assertEqual(str(reduced), "v1 + v2")


if __name__ == "__main__":
    unittest.main()
