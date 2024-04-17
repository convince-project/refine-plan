""" Small test to evaluate logic minimisation behaviour.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.state_factor import IntStateFactor
from refine_plan.models.condition import EqCondition
from refine_plan.algorithms.gfactor import gfactor
from sympy import Symbol, sympify, factor
from pyeda.inter import expr

if __name__ == "__main__":

    # State factors for grid locations
    x = IntStateFactor("x", 0, 10)
    y = IntStateFactor("y", 0, 10)

    # Create rule
    rule = expr(
        "Or(And(xEQ4, yEQ1), And(yEQ0, xEQ2), And(xEQ1, yEQ3), And(yEQ0, xEQ4), "
        + "And(xEQ0, yEQ0),And(xEQ1, yEQ4), And(xEQ2, yEQ1), And(xEQ0, yEQ4), And(xEQ2, yEQ2), "
        + "And(xEQ3, yEQ1), And(xEQ1, yEQ1), And(yEQ0, xEQ3), And(xEQ2, yEQ4), And(xEQ4, yEQ3), "
        + "And(yEQ0, xEQ1), And(xEQ3, yEQ4), And(xEQ3, yEQ3), And(xEQ0, yEQ1), And(xEQ4, yEQ4), "
        + "And(xEQ1, yEQ2), And(xEQ0, yEQ3), And(xEQ3, yEQ2), And(xEQ1, yEQ5), And(xEQ0, yEQ2), "
        + "And(xEQ4, yEQ2), And(xEQ2, yEQ3))"
    )

    # Setup converter
    converter = PolicyBTConverter()
    for i in range(0, 11):
        x_name = "xEQ{}".format(i)
        y_name = "yEQ{}".format(i)
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}
        converter._vars_to_conds[x_name] = EqCondition(x, i)
        converter._vars_to_conds[y_name] = EqCondition(y, i)
        converter._vars_to_symbols[x_name] = Symbol(x_name)
        converter._vars_to_symbols[y_name] = Symbol(y_name)

    algebra_rule = sympify(
        converter._logic_to_algebra(rule), locals=converter._vars_to_symbols
    )

    print("Sympy version of rule")
    print(algebra_rule)

    factorised = gfactor(algebra_rule)
    print("Post GFactor")
    print(factorised)

    print("Pyeda")
    print(converter._minimise_with_espresso({"a": rule}))
