#!/usr/bin/env python3
""" Implementation of the GFactor factorisation algorithm.

GFactor was introduced in: 
Wang, A.R.R., 1990. Algorithms for multilevel logic optimization.

The implementation here is based on the implementation in:
https://github.com/SimonaGug/BT-from-planning-experts

Which is an implementation of the paper:

Gugliermo, S., Schaffernicht, E., Koniaris, C. and Pecora, F., 2023. 
Learning behavior trees from planning experts using decision tree and logic 
factorization. IEEE Robotics and Automation Letters.

It is not entirely clear if the version of GFactor in the paper is identical
to that in the original PhD thesis it came from.

Note the implementation I am referencing here does not have an associated license.

My reimplementation improves readability, makes bug fixes from the original, and makes
general quality of life improvements.

Author: Charlie Street
Owner: Charlie Street
"""

from sympy import sympify, reduced, Symbol, Add, Mul
import random

# Magic number for _quick_divisor to avoid sympy negative remainder problem
MAX_TRIES = 10


def _most_common_condition(formula):
    """Find the most condition in a formula.

    Finds the variable (which represents a logical condition) which
    appears in the most conjunctions.

    The assumption here I think is that the expression is DNF.
    And that a variable only appears once in a conjunction.
    Given our sympy expressions are logical expressions, this should always
    be satisfied.

    Args:
        formula: The formula to search through

    Returns:
        most_common_condition: The most common condition, or None if there isn't one
    """

    frequencies = _get_variable_frequencies(formula)
    mcc = max(frequencies, key=frequencies.get)
    if frequencies[mcc] > 1:
        return mcc
    else:
        return None


def gfactor(formula, divisor_fn=_most_common_condition):
    """Runs the GFactor factorisation algorithm.

    This code really just follows the pseudocode in the paper

    Args:
        formula: A sympy formula
        divisor_fn: Optional. Sets the function to use for the initial divisor.

    Returns:
        factorised: A factorised formula in the format p*q + r
    """
    divisor = divisor_fn(formula)

    if divisor is None:  # If no divisor is possible
        return formula  # No factorisation can be done

    quotient, remainder = _divide(formula, divisor)

    if isinstance(quotient, Symbol):  # If just a single symbol
        return _lf(formula, quotient, divisor_fn)

    quotient = _make_cube_free(quotient)
    divisor, remainder = _divide(formula, quotient)

    if divisor.is_zero:  # I.e. the division didn't work
        return formula

    if _is_cube_free(divisor):
        # Condition found in implementation I'm referencing - can't hurt
        if quotient != 1:
            quotient = gfactor(quotient, divisor_fn)
        divisor = gfactor(divisor, divisor_fn)
        remainder = gfactor(remainder, divisor_fn) if remainder != 0 else remainder

        return quotient * divisor + remainder

    largest_common_cube = _largest_common_cube(divisor)
    return _lf(formula, largest_common_cube, divisor_fn)


def _lf(formula, divisor, divisor_fn=_most_common_condition):
    """A more basic logical factorisation algorithm.

    This is a subroutine of GFactor and also makes recursive calls to g factor.

    Args:
        formula: The formula being fatorised.
        divisor: The divisor (what we're dividing by)
        divisor_fn: Optional. Sets the function to use for the initial divisor.

    Returns:
        factorised: The factorised formula
    """
    quotient, remainder = _divide(formula, divisor)

    quotient = gfactor(quotient, divisor_fn)

    if not remainder.is_zero:
        remainder = gfactor(remainder, divisor_fn)

    return divisor * quotient + remainder


def _largest_common_cube(formula):
    """Finds the most common cube in an formula.

    The formula should be in DNF.

    The most common cube is the largest cube (i.e. a conjunction) which
    appears in all components of a big discussion. This common cube can
    then be factored out.

    Args:
        formula: The formula we're searching through

    Returns:
        largest_common_cube: The largest common cube (a Sympy formula)
                             or None if no common cube exists
    """

    # if we have just a single cube (i.e. one conjunction)
    if isinstance(formula, Mul) or isinstance(formula, Symbol):
        return formula

    assert isinstance(formula, Add)  # Logical OR

    common_cube = None

    for cube in formula.args:
        assert isinstance(cube, Mul) or isinstance(cube, Symbol)
        if isinstance(cube, Mul):
            assert all(isinstance(a, Symbol) for a in cube.args)
            cube_vars = set(cube.args)
        elif isinstance(cube, Symbol):
            cube_vars = set([cube])

        if common_cube is None:
            common_cube = cube_vars
        else:
            common_cube = common_cube.intersection(cube_vars)

        if len(common_cube) == 0:
            return None

    return Mul(*common_cube)  # Make it a conjunction again


def _make_cube_free(formula):
    """Remove the common cubes from a formula.

    Assuming we can identify the largest common cube, I think that means
    there are no more common cubes across all parts of the expression.

    Args:
        formula: The formula to make cube free

    Returns:
        cube_free_formula: The formula with the largest common cube removed
    """
    if _is_cube_free(formula):
        return formula

    largest_common_cube = _largest_common_cube(formula)
    quotient, remainder = _divide(formula, largest_common_cube)
    # This should divide perfectly as its a common cube
    assert remainder.is_zero
    assert _is_cube_free(quotient)
    return quotient


def _is_cube_free(formula):
    """Checks if there are any common cubes within a formula.

    Args:
        formula: The formula to check

    Returns:
        is_cube_free: Are there no common cubes in the formula?
    """
    # I think it is more efficient to check the largest common cube
    return _largest_common_cube(formula) is None


def _divide(formula, divisor):
    """Divide a formula by a divisor.

    Args:
        formula: The formula to be divided
        divisor: What we're dividing formula by

    Returns:
        quotient: The quotient of the division
        remainder: The remainder of the division
    """
    quotient, remainder = reduced(formula, [divisor])

    if "-" in str(remainder):
        # Helpful tip from the implementation this is based on
        # This is to handle a bug in sympy apparently
        return sympify(0), formula

    # quotient is a single-item list
    return quotient[0], remainder


def _get_variables_in_formula(formula):
    """Gets a list of all variables in a sympy formula.

    This returns duplicates of variables (i.e. its a list not a set)

    Args:
        formula: The sympy expression

    Returns:
        variables: The list of variables (symbols) in the formula
    """

    if isinstance(formula, Add) or isinstance(formula, Mul):
        variables = []
        for sub_formula in formula.args:
            variables += _get_variables_in_formula(sub_formula)
        return variables
    elif isinstance(formula, Symbol):
        return [formula]


def _get_variable_frequencies(formula):
    """Returns the frequencies of each variable in a formula.

    Args:
        formula: An algebraic logical formula

    Returns:
        frequencies: A dict from variables (Symbols) to frequencies
    """
    variables = _get_variables_in_formula(formula)
    frequencies = {}

    for var in variables:
        if var not in frequencies:
            frequencies[var] = 0
        frequencies[var] += 1

    return frequencies


def _get_random_divisor(formula):
    """Get a random divisor for GFactor. The divisor is a single variable.

    Args:
        formula: The formula we're trying to factorise

    Returns:
        divisor: A random variable to choose as divisor
    """
    if isinstance(formula, Symbol):
        return None
    else:
        frequencies = _get_variable_frequencies(formula)
        # Can only occur if all variables occur once
        if sum(frequencies.values()) == len(frequencies):
            return None
        else:
            return random.choice(list(frequencies.keys()))


def _one_zero_kernel(formula):
    """Find a level-0 kernel to use as an initial divisor for GFactor.

    Kernels are cube-free primary divisions of formula.

    Implemented from:
    Wang, A.R.R., 1990. Algorithms for multilevel logic optimization.

    Args:
        formula: The formula we're trying to factorise

    Returns:
        kernel: The level-0 kernel.
    """
    frequencies = _get_variable_frequencies(formula)
    if sum(frequencies.values()) == len(frequencies):
        return formula

    more_than_once = {v: frequencies[v] for v in frequencies if frequencies[v] > 1}
    random_literal = random.choice(list(more_than_once.keys()))

    q, _ = _divide(formula, random_literal)
    cubeless_q = _make_cube_free(q)

    return _one_zero_kernel(cubeless_q)


def _quick_divisor(formula):
    """Implements the quick divisor function from the thesis below.

    Wang, A.R.R., 1990. Algorithms for multilevel logic optimization.

    Args:
        formula: The formula we're trying to factorise

    Returns:
        divisor: The initial divisor for gfactor
    """
    if isinstance(formula, Symbol):
        return None
    else:
        frequencies = _get_variable_frequencies(formula)
        # Can only occur if all variables occur once
        if sum(frequencies.values()) == len(frequencies):
            return None
        else:
            kernel = _one_zero_kernel(formula)

            # NOTE: This is a hacky solution to unfortunate behaviour in sympy
            # reduced in Sympy will sometimes return a negative remainder
            # which doesn't make any sense
            # To address this, I will exploit the randomness in
            # _one_zero_kernel and run it a number of times
            # If everything else fails, I'll just run _most_common_condition
            # which should never have this issue, as it returns one variable
            q, _ = _divide(formula, kernel)
            tries = 1
            while q.is_zero and tries < MAX_TRIES:
                kernel = _one_zero_kernel(formula)
                q, _ = _divide(formula, kernel)
                tries += 1

            if tries >= MAX_TRIES:
                return _most_common_condition(formula)

            return kernel
