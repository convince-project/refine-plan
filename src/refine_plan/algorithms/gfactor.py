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

from sympy import sympify, reduced


# TODO: Do I even need the symbol map?
# TODO: Is every function which assumes DNF correct to assume this?
def gfactor(formula, symbol_map):
    """Runs the GFactor factorisation algorithm.

    Args:
        formula: A sympy formula
        symbol_map: A mapping from variable names to symbols (for Sympy)

    Returns:
        factorised: A factorised formula in the format p*q + r
    """
    # TODO: Fill in
    pass


def _lf(formula, divisor):
    """A more basic logical factorisation algorithm.

    This is a subroutine of GFactor and also makes recursive calls to g factor.

    Args:
        formula: The formula being fatorised.
        divisor: The divisor (what we're dividing by)

    Returns:
        factorised: The factorised formula
    """
    # TODO: Fill in
    pass


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
    # TODO: Fill in - should be straightforward
    pass


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
