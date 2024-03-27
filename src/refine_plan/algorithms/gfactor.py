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

from sympy import sympify, reduced, Symbol


# TODO: Do I even need the symbol map?
# TODO: Is every function which assumes DNF correct to assume this?
def gfactor(formula):
    """Runs the GFactor factorisation algorithm.

    This code really just follows the pseudocode in the paper

    Args:
        formula: A sympy formula

    Returns:
        factorised: A factorised formula in the format p*q + r
    """
    divisor = _most_common_condition(formula)

    if divisor is None:  # If there is no common condition
        return formula  # No factorisation can be done

    quotient, remainder = _divide(formula, divisor)

    if isinstance(quotient, Symbol):  # If just a single symbol
        return _lf(formula, quotient)

    quotient = _make_cube_free(quotient)
    divisor, remainder = _divide(formula, quotient)

    if divisor == 0:  # I.e. the division didn't work
        return formula

    if _is_cube_free(divisor):
        # NOTE: There was a condition here 'if "1" not in q" - not in the paper?
        quotient = gfactor(quotient)
        divisor = gfactor(divisor)
        remainder = gfactor(remainder) if remainder != 0 else remainder

        return quotient * divisor + remainder

    largest_common_cube = _largest_common_cube(divisor)
    return _lf(formula, largest_common_cube)


def _lf(formula, divisor):
    """A more basic logical factorisation algorithm.

    This is a subroutine of GFactor and also makes recursive calls to g factor.

    Args:
        formula: The formula being fatorised.
        divisor: The divisor (what we're dividing by)

    Returns:
        factorised: The factorised formula
    """
    quotient, remainder = _divide(formula, divisor)

    quotient = gfactor(quotient)

    if remainder != 0:
        remainder = gfactor(remainder)

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
        most_common_cube: The most common cube (a Sympy formula)
    """
    # TODO: Fill in
    # I think I can do this by just doing the intersection as I traverse
    # forward rather than the two step approach the original authors take
    pass


def _make_cube_free(formula):
    """Remove the common cubes from a formula.

    Assuming we can identify the largest common cube, I think that means
    there are no more common cubes across all parts of the expression.

    Args:
        formula: The formula to make cube free

    Returns:
        cube_free_formula: The formula with the largest common cube removed
    """
    # TODO: Fill in. Should be a quick check and a divide
    # Maybe add an assertion that the remainder from the division is zero?
    pass


def _is_cube_free(formula):
    """Checks if there are any common cubes within a formula.

    Args:
        formula: The formula to check

    Returns:
        is_cube_free: Are there no common cubes in the formula?
    """
    # TODO: Fill in - I wonder if just seeing if most common cube returns none
    # is an easier way to write this?
    pass


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
        return 0, formula

    # quotient is a single-item list
    return quotient[0], remainder


def _most_common_condition(formula):
    """Find the most condition in a formula.

    Finds the variable (which represents a logical condition) which
    appears in the most conjunctions.

    Args:
        formula: The formula to search through

    Returns:
        most_common_condition: The most common condition, or None if there isn't one
    """
    # TODO: Fill in
    pass
