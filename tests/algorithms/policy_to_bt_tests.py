#!/usr/bin/env python3
""" Unit tests for the policy->BT converter.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, IntStateFactor
from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from pyeda.boolalg.expr import expr, And, Or, Not
from refine_plan.models.behaviour_tree import (
    ConditionNode,
    SequenceNode,
    FallbackNode,
    BehaviourTree,
    ActionNode,
)
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
from refine_plan.models.condition import (
    EqCondition,
    GtCondition,
    GeqCondition,
    NeqCondition,
    LeqCondition,
    LtCondition,
)
from pyeda.inter import espresso_exprs
from sympy import Symbol, Add, sympify
import unittest
import os


class ResetTest(unittest.TestCase):

    def test_function(self):
        # Reset called in constructor
        converter = PolicyBTConverter()
        self.assertEqual(converter._vars_to_conds, None)
        self.assertEqual(converter._vars_to_symbols, None)
        self.assertEqual(converter._default_action, "None")
        self.assertFalse(converter._default_needed)

        converter = PolicyBTConverter("test")
        self.assertEqual(converter._vars_to_conds, None)
        self.assertEqual(converter._vars_to_symbols, None)
        self.assertEqual(converter._default_action, "test")
        self.assertFalse(converter._default_needed)


class ExtractRulesFromPolicyTest(unittest.TestCase):

    def test_function(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = "a3"
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)

        with self.assertRaises(Exception):
            converter._extract_rules_from_policy("POLICY")

        act_to_rule, act_to_var_map = converter._extract_rules_from_policy(policy)

        self.assertFalse(converter._default_needed)

        self.assertEqual(len(act_to_var_map), 3)
        self.assertEqual(len(act_to_var_map["a1"]), 5)
        self.assertEqual(
            act_to_var_map["a1"],
            {
                "sf1EQa": EqCondition(sf1, "a"),
                "sf2EQd": EqCondition(sf2, "d"),
                "sf1EQb": EqCondition(sf1, "b"),
                "sf2EQf": EqCondition(sf2, "f"),
                "sf1EQc": EqCondition(sf1, "c"),
            },
        )
        self.assertEqual(len(act_to_var_map["a2"]), 5)
        self.assertEqual(
            act_to_var_map["a2"],
            {
                "sf1EQa": EqCondition(sf1, "a"),
                "sf2EQe": EqCondition(sf2, "e"),
                "sf1EQb": EqCondition(sf1, "b"),
                "sf2EQd": EqCondition(sf2, "d"),
                "sf1EQc": EqCondition(sf1, "c"),
            },
        )
        self.assertEqual(len(act_to_var_map["a3"]), 6)
        self.assertEqual(
            act_to_var_map["a3"],
            {
                "sf1EQa": EqCondition(sf1, "a"),
                "sf2EQf": EqCondition(sf2, "f"),
                "sf1EQb": EqCondition(sf1, "b"),
                "sf2EQe": EqCondition(sf2, "e"),
                "sf1EQc": EqCondition(sf1, "c"),
                "sf2EQd": EqCondition(sf2, "d"),
            },
        )

        self.assertEqual(len(act_to_rule), 3)

        self.assertTrue(
            act_to_rule["a1"].equivalent(
                Or(
                    And(expr("sf1EQa"), expr("sf2EQd")),
                    And(expr("sf1EQb"), expr("sf2EQf")),
                    And(expr("sf1EQc"), expr("sf2EQf")),
                )
            )
        )

        self.assertTrue(
            act_to_rule["a2"].equivalent(
                Or(
                    And(expr("sf1EQa"), expr("sf2EQe")),
                    And(expr("sf1EQb"), expr("sf2EQd")),
                    And(expr("sf1EQc"), expr("sf2EQe")),
                )
            )
        )

        self.assertTrue(
            act_to_rule["a3"].equivalent(
                Or(
                    And(expr("sf1EQa"), expr("sf2EQf")),
                    And(expr("sf1EQb"), expr("sf2EQe")),
                    And(expr("sf1EQc"), expr("sf2EQd")),
                )
            )
        )

    def test_default_action(self):

        converter = PolicyBTConverter("replacement")

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = None
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = None
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = None
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)

        with self.assertRaises(Exception):
            converter._extract_rules_from_policy("POLICY")

        act_to_rule, act_to_var_map = converter._extract_rules_from_policy(policy)

        self.assertTrue(converter._default_needed)

        self.assertEqual(len(act_to_var_map), 2)
        self.assertEqual(len(act_to_var_map["a1"]), 5)
        self.assertEqual(
            act_to_var_map["a1"],
            {
                "sf1EQa": EqCondition(sf1, "a"),
                "sf2EQd": EqCondition(sf2, "d"),
                "sf1EQb": EqCondition(sf1, "b"),
                "sf2EQf": EqCondition(sf2, "f"),
                "sf1EQc": EqCondition(sf1, "c"),
            },
        )
        self.assertEqual(len(act_to_var_map["a2"]), 5)
        self.assertEqual(
            act_to_var_map["a2"],
            {
                "sf1EQa": EqCondition(sf1, "a"),
                "sf2EQe": EqCondition(sf2, "e"),
                "sf1EQb": EqCondition(sf1, "b"),
                "sf2EQd": EqCondition(sf2, "d"),
                "sf1EQc": EqCondition(sf1, "c"),
            },
        )

        self.assertEqual(len(act_to_rule), 2)

        self.assertTrue(
            act_to_rule["a1"].equivalent(
                Or(
                    And(expr("sf1EQa"), expr("sf2EQd")),
                    And(expr("sf1EQb"), expr("sf2EQf")),
                    And(expr("sf1EQc"), expr("sf2EQf")),
                )
            )
        )

        self.assertTrue(
            act_to_rule["a2"].equivalent(
                Or(
                    And(expr("sf1EQa"), expr("sf2EQe")),
                    And(expr("sf1EQb"), expr("sf2EQd")),
                    And(expr("sf1EQc"), expr("sf2EQe")),
                )
            )
        )


class BuildInternalMappingsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = "a3"
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)
        _, act_to_var_map = converter._extract_rules_from_policy(policy)

        self.assertEqual(converter._vars_to_conds, None)
        self.assertEqual(converter._vars_to_symbols, None)

        converter._build_internal_mappings(act_to_var_map)

        self.assertEqual(len(converter._vars_to_conds), 6)
        self.assertEqual(len(converter._vars_to_symbols), 6)

        self.assertEqual(converter._vars_to_conds["sf1EQa"], EqCondition(sf1, "a"))
        self.assertEqual(converter._vars_to_conds["sf1EQb"], EqCondition(sf1, "b"))
        self.assertEqual(converter._vars_to_conds["sf1EQc"], EqCondition(sf1, "c"))
        self.assertEqual(converter._vars_to_conds["sf2EQd"], EqCondition(sf2, "d"))
        self.assertEqual(converter._vars_to_conds["sf2EQe"], EqCondition(sf2, "e"))
        self.assertEqual(converter._vars_to_conds["sf2EQf"], EqCondition(sf2, "f"))

        self.assertEqual(converter._vars_to_symbols["sf1EQa"], Symbol("sf1EQa"))
        self.assertEqual(converter._vars_to_symbols["sf1EQb"], Symbol("sf1EQb"))
        self.assertEqual(converter._vars_to_symbols["sf1EQc"], Symbol("sf1EQc"))
        self.assertEqual(converter._vars_to_symbols["sf2EQd"], Symbol("sf2EQd"))
        self.assertEqual(converter._vars_to_symbols["sf2EQe"], Symbol("sf2EQe"))
        self.assertEqual(converter._vars_to_symbols["sf2EQf"], Symbol("sf2EQf"))


class MinimiseWithEspresso(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = "a3"
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)
        act_to_rule, _ = converter._extract_rules_from_policy(policy)

        act_to_min_rule = converter._minimise_with_espresso(act_to_rule)

        self.assertEqual(len(act_to_min_rule), 3)

        expected_min_rules = {}

        expected_min_rules["a1"] = espresso_exprs(
            Or(
                And(expr("sf1EQa"), expr("sf2EQd")),
                And(expr("sf1EQb"), expr("sf2EQf")),
                And(expr("sf1EQc"), expr("sf2EQf")),
            ).to_dnf()
        )[0]

        expected_min_rules["a2"] = espresso_exprs(
            Or(
                And(expr("sf1EQa"), expr("sf2EQe")),
                And(expr("sf1EQb"), expr("sf2EQd")),
                And(expr("sf1EQc"), expr("sf2EQe")),
            ).to_dnf()
        )[0]

        expected_min_rules["a3"] = espresso_exprs(
            Or(
                And(expr("sf1EQa"), expr("sf2EQf")),
                And(expr("sf1EQb"), expr("sf2EQe")),
                And(expr("sf1EQc"), expr("sf2EQd")),
            ).to_dnf()
        )[0]

        for action in ["a1", "a2", "a3"]:
            min_rule = act_to_min_rule[action]
            self.assertTrue(min_rule.equivalent(expected_min_rules[action]))
            self.assertTrue(min_rule.is_dnf())


class ConvertToHornClausesTest(unittest.TestCase):
    def test_function(self):
        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = "a3"
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = "a3"
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)
        act_to_rule, _ = converter._extract_rules_from_policy(policy)

        act_to_horn = converter._convert_to_horn_clauses(act_to_rule)

        self.assertEqual(len(act_to_horn), 3)

        for action in ["a1", "a2", "a3"]:
            self.assertTrue(act_to_horn[action].is_dnf())
            self.assertTrue(act_to_horn[action].equivalent(Not(act_to_rule[action])))


class GetVariablesInDNFExprTest(unittest.TestCase):

    def test_function(self):

        rule = Or(
            And(expr("v1"), expr("v2")), And(expr("v3"), Not(expr("v4"))), expr("v5")
        )
        converter = PolicyBTConverter()
        var_list = converter._get_variables_in_dnf_expr(rule)

        self.assertEqual(len(var_list), 5)
        self.assertTrue(expr("v1") in var_list)
        self.assertTrue(expr("v2") in var_list)
        self.assertTrue(expr("v3") in var_list)
        self.assertTrue(expr("v4") in var_list)
        self.assertTrue(expr("v5") in var_list)

        bad_rule = And(expr("v1"), Or(expr("v2"), expr("v3")))
        with self.assertRaises(Exception):
            converter._get_variables_in_dnf_expr(bad_rule)


class ScoreAndSortRulesTest(unittest.TestCase):

    def test_function(self):
        act_to_horn = {
            "a1": Or(
                And(Not(expr("v1")), Not(expr("v3"))),
                And(Not(expr("v3")), Not(expr("v2"))),
            ),
            "a2": And(Not(expr("v2")), Not(expr("v4"))),
            "a3": And(Not(expr("v3")), Not(expr("v1"))),
        }

        converter = PolicyBTConverter()
        ordered_ra_pairs = converter._score_and_sort_rules(act_to_horn)

        self.assertEqual(len(ordered_ra_pairs), 3)

        self.assertEqual(ordered_ra_pairs[0][1], "a3")
        self.assertTrue(ordered_ra_pairs[0][0].equivalent(act_to_horn["a3"]))
        self.assertEqual(ordered_ra_pairs[1][1], "a1")
        self.assertTrue(ordered_ra_pairs[1][0].equivalent(act_to_horn["a1"]))
        self.assertEqual(ordered_ra_pairs[2][1], "a2")
        self.assertTrue(ordered_ra_pairs[2][0].equivalent(act_to_horn["a2"]))


class LogicToAlgebraTest(unittest.TestCase):

    def test_function(self):

        test_expr = Or(
            And(Not(expr("v1")), expr("v2")), expr("v3"), And(expr("v4"), expr("v5"))
        )
        converter = PolicyBTConverter()
        converter._vars_to_symbols = {}
        sympy_str = converter._logic_to_algebra(test_expr)
        self.assertEqual(sympy_str, "v3 + NOTv1*v2 + v4*v5")

        self.assertEqual(converter._vars_to_symbols, {"NOTv1": Symbol("NOTv1")})


class PyedaRulesToSympyAlgebraicTest(unittest.TestCase):

    def test_function(self):

        converter = PolicyBTConverter()
        converter._vars_to_symbols = {}
        for i in range(1, 6):
            var_name = "v{}".format(i)
            converter._vars_to_symbols[var_name] = Symbol(var_name)

        ordered_ra_pairs = []
        ordered_ra_pairs.append(
            (
                Or(And(Not(expr("v1")), expr("v2")), And(Not(expr("v2")), expr("v3"))),
                "a1",
            )
        )
        ordered_ra_pairs.append((Or(expr("v3"), expr("v4")), "a2"))
        ordered_ra_pairs.append((Or(And(expr("v3"), expr("v4")), expr("v5")), "a3"))

        ordered_alg_act_pairs = converter._pyeda_rules_to_sympy_algebraic(
            ordered_ra_pairs
        )

        self.assertEqual(len(ordered_alg_act_pairs), 3)
        self.assertTrue(isinstance(ordered_alg_act_pairs[0][0], Add))
        self.assertEqual(str(ordered_alg_act_pairs[0][0]), "NOTv1*v2 + NOTv2*v3")
        self.assertEqual(ordered_alg_act_pairs[0][1], "a1")
        self.assertEqual(str(ordered_alg_act_pairs[1][0]), "v3 + v4")
        self.assertTrue(isinstance(ordered_alg_act_pairs[1][0], Add))
        self.assertEqual(ordered_alg_act_pairs[1][1], "a2")
        self.assertEqual(str(ordered_alg_act_pairs[2][0]), "v3*v4 + v5")
        self.assertTrue(isinstance(ordered_alg_act_pairs[2][0], Add))
        self.assertEqual(ordered_alg_act_pairs[2][1], "a3")

        self.assertEqual(len(converter._vars_to_symbols), 7)
        self.assertTrue("NOTv1" in converter._vars_to_symbols)
        self.assertEqual(converter._vars_to_symbols["NOTv1"], Symbol("NOTv1"))
        self.assertTrue("NOTv2" in converter._vars_to_symbols)
        self.assertEqual(converter._vars_to_symbols["NOTv2"], Symbol("NOTv2"))


class ReduceSympyExpressionsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_symbols = {
            "v{}".format(i): Symbol("v{}".format(i)) for i in range(1, 10)
        }

        ordered_alg_act_pairs = []
        ordered_alg_act_pairs.append(
            (sympify("v5 + v7 + v8", locals=converter._vars_to_symbols), "a1")
        )
        ordered_alg_act_pairs.append(
            (
                sympify(
                    "v8 + v2*v3*v5 + v2*v7 + v9", locals=converter._vars_to_symbols
                ),
                "a2",
            )
        )
        ordered_alg_act_pairs.append(
            (
                sympify("v2*v3*v4*v6 + v2*v7 + v8", locals=converter._vars_to_symbols),
                "a3",
            )
        )
        ordered_alg_act_pairs.append(
            (sympify("v3*v4*v5*v6 + v7", locals=converter._vars_to_symbols), "a4")
        )
        ordered_alg_act_pairs.append(
            (
                sympify("v4*v6 + v6*v9 + v7 + v8", locals=converter._vars_to_symbols),
                "a5",
            )
        )
        ordered_alg_act_pairs.append(
            (
                sympify(
                    "v10 + v2*v3*v4*v5 + v2*v7 + v8 + v9",
                    locals=converter._vars_to_symbols,
                ),
                "a6",
            )
        )
        ordered_alg_act_pairs.append(
            (sympify("v1 + v2", locals=converter._vars_to_symbols), "a7")
        )

        min_alg_act_pairs = converter._reduce_sympy_expressions(ordered_alg_act_pairs)
        self.assertEqual(len(min_alg_act_pairs), 7)
        self.assertEqual(str(min_alg_act_pairs[0][0]), "v5 + v7 + v8")
        self.assertEqual(min_alg_act_pairs[0][1], "a1")
        self.assertEqual(str(min_alg_act_pairs[1][0]), "v2*(v3*v5 + v7) + v8 + v9")
        self.assertEqual(min_alg_act_pairs[1][1], "a2")
        self.assertEqual(str(min_alg_act_pairs[2][0]), "v2*(v3*v4*v6 + v7) + v8")
        self.assertEqual(min_alg_act_pairs[2][1], "a3")
        self.assertEqual(str(min_alg_act_pairs[3][0]), "v3*v4*v5*v6 + v7")
        self.assertEqual(min_alg_act_pairs[3][1], "a4")
        self.assertEqual(str(min_alg_act_pairs[4][0]), "v6*(v4 + v9) + v7 + v8")
        self.assertEqual(min_alg_act_pairs[4][1], "a5")
        self.assertEqual(
            str(min_alg_act_pairs[5][0]), "v10 + v2*(v3*v4*v5 + v7) + v8 + v9"
        )
        self.assertEqual(min_alg_act_pairs[5][1], "a6")
        self.assertEqual(str(min_alg_act_pairs[6][0]), "v1 + v2")
        self.assertEqual(min_alg_act_pairs[6][1], "a7")


class SfValsCoveredBySymbolsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)

        symbols = [
            Symbol("sf1EQa"),
            Symbol("NOTsf1EQc"),
            Symbol("sf2GT6"),
            Symbol("NOTsf2GT9"),
        ]

        converter._vars_to_conds["sf1EQa"] = EqCondition(sf1, "a")
        converter._vars_to_conds["sf1EQc"] = EqCondition(sf1, "c")
        converter._vars_to_conds["sf2GT6"] = GtCondition(sf2, 6)
        converter._vars_to_conds["sf2GT9"] = GtCondition(sf2, 9)

        converter._vars_to_symbols["sf1EQa"] = Symbol("sf1EQa")
        converter._vars_to_symbols["NOTsf1EQc"] = Symbol("NOTsf1EQc")
        converter._vars_to_symbols["sf2GT6"] = Symbol("sf2GT6")
        converter._vars_to_symbols["NOTsf2GT9"] = Symbol("NOTsf2GT9")

        sf_val_dict = converter._sf_vals_covered_by_symbols(symbols, True)
        self.assertEqual(len(sf_val_dict), 2)
        sf1_symbols, sf1_covered = sf_val_dict[sf1]
        self.assertEqual(len(sf1_symbols), 2)
        self.assertTrue(Symbol("sf1EQa") in sf1_symbols)
        self.assertTrue(Symbol("NOTsf1EQc") in sf1_symbols)
        self.assertEqual(sf1_covered, set(["a"]))

        sf2_symbols, sf2_covered = sf_val_dict[sf2]
        self.assertEqual(len(sf2_symbols), 2)
        self.assertTrue(Symbol("sf2GT6") in sf2_symbols)
        self.assertTrue(Symbol("NOTsf2GT9") in sf2_symbols)
        self.assertEqual(sf2_covered, set([7, 8, 9]))

        sf_val_dict = converter._sf_vals_covered_by_symbols(symbols, False)
        self.assertEqual(len(sf_val_dict), 2)
        sf1_symbols, sf1_covered = sf_val_dict[sf1]
        self.assertEqual(len(sf1_symbols), 2)
        self.assertTrue(Symbol("sf1EQa") in sf1_symbols)
        self.assertTrue(Symbol("NOTsf1EQc") in sf1_symbols)
        self.assertEqual(sf1_covered, set(["a", "b"]))

        sf2_symbols, sf2_covered = sf_val_dict[sf2]
        self.assertEqual(len(sf2_symbols), 2)
        self.assertTrue(Symbol("sf2GT6") in sf2_symbols)
        self.assertTrue(Symbol("NOTsf2GT9") in sf2_symbols)
        self.assertEqual(sf2_covered, set([5, 6, 7, 8, 9, 10]))


class CondToSymbolTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}

        cond = EqCondition(StateFactor("sf", ["a", "b", "c"]), "b")
        s = converter._cond_to_symbol(cond)
        self.assertEqual(s, Symbol("sfEQb"))
        self.assertEqual(converter._vars_to_symbols, {"sfEQb": Symbol("sfEQb")})
        self.assertEqual(converter._vars_to_conds, {"sfEQb": cond})

        cond_two = NeqCondition(StateFactor("sf", ["a", "b", "c"]), "b")
        s = converter._cond_to_symbol(cond_two)
        self.assertEqual(s, Symbol("NOTsfEQb"))
        self.assertEqual(
            converter._vars_to_symbols,
            {"sfEQb": Symbol("sfEQb"), "NOTsfEQb": Symbol("NOTsfEQb")},
        )
        self.assertEqual(converter._vars_to_conds, {"sfEQb": cond})

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}
        s = converter._cond_to_symbol(cond_two)
        self.assertEqual(s, Symbol("NOTsfEQb"))
        self.assertEqual(
            converter._vars_to_symbols,
            {"sfEQb": Symbol("sfEQb"), "NOTsfEQb": Symbol("NOTsfEQb")},
        )
        self.assertEqual(converter._vars_to_conds, {"sfEQb": cond})

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}
        gt_cond = GtCondition(IntStateFactor("sf", 5, 10), 7)
        s = converter._cond_to_symbol(gt_cond)
        self.assertEqual(s, Symbol("sfGT7"))
        self.assertEqual(converter._vars_to_symbols, {"sfGT7": Symbol("sfGT7")})
        self.assertEqual(converter._vars_to_conds, {"sfGT7": gt_cond})

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}
        geq_cond = GeqCondition(IntStateFactor("sf", 5, 10), 7)
        s = converter._cond_to_symbol(geq_cond)
        self.assertEqual(s, Symbol("sfGEQ7"))
        self.assertEqual(converter._vars_to_symbols, {"sfGEQ7": Symbol("sfGEQ7")})
        self.assertEqual(converter._vars_to_conds, {"sfGEQ7": geq_cond})

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}
        lt_cond = LtCondition(IntStateFactor("sf", 5, 10), 7)
        s = converter._cond_to_symbol(lt_cond)
        self.assertEqual(s, Symbol("NOTsfGEQ7"))
        self.assertEqual(
            converter._vars_to_symbols,
            {"sfGEQ7": Symbol("sfGEQ7"), "NOTsfGEQ7": Symbol("NOTsfGEQ7")},
        )
        self.assertEqual(converter._vars_to_conds, {"sfGEQ7": geq_cond})

        converter._vars_to_symbols = {}
        converter._vars_to_conds = {}
        leq_cond = LeqCondition(IntStateFactor("sf", 5, 10), 7)
        s = converter._cond_to_symbol(leq_cond)
        self.assertEqual(s, Symbol("NOTsfGT7"))
        self.assertEqual(
            converter._vars_to_symbols,
            {"sfGT7": Symbol("sfGT7"), "NOTsfGT7": Symbol("NOTsfGT7")},
        )
        self.assertEqual(converter._vars_to_conds, {"sfGT7": gt_cond})

        converter._vars_to_conds = {"sfEQb": "WHAT"}
        converter._vars_to_symbols = {}
        cond = EqCondition(StateFactor("sf", ["a", "b", "c"]), "b")
        with self.assertRaises(Exception):
            converter._cond_to_symbol(cond)

        converter._vars_to_conds = {}
        converter._vars_to_symbols = {"sfEQb": Symbol("WHAT")}
        cond = EqCondition(StateFactor("sf", ["a", "b", "c"]), "b")
        with self.assertRaises(Exception):
            converter._cond_to_symbol(cond)

        converter._vars_to_conds = {}
        converter._vars_to_symbols = {"NOTsfEQb": Symbol("WHAT")}
        cond = NeqCondition(StateFactor("sf", ["a", "b", "c"]), "b")
        with self.assertRaises(Exception):
            converter._cond_to_symbol(cond)


class SimplifyIntValsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        sf = IntStateFactor("sf", 5, 10)
        symbs = [Symbol("sfEQ6"), Symbol("sfEQ8"), Symbol("sfEQ10")]
        # self.assertEqual(converter._simplify_int_vals(sf, symbs, [6, 8, 10]), None)

        symbs = [Symbol("sfEQ6"), Symbol("sfEQ7"), Symbol("sfEQ9"), Symbol("sfEQ10")]
        int_simple = converter._simplify_int_vals(sf, symbs, [6, 7, 10, 9])
        self.assertEqual(
            int_simple,
            sympify("sfGEQ9 + (NOTsfGT7*sfGEQ6)", locals=converter._vars_to_symbols),
        )

        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        sf = IntStateFactor("sf", 5, 14)
        symbs = [
            Symbol("sfEQ5"),
            Symbol("sfEQ6"),
            Symbol("sfEQ7"),
            Symbol("sfEQ9"),
            Symbol("sfEQ11"),
            Symbol("sfEQ12"),
            Symbol("sfEQ14"),
        ]

        int_simple = converter._simplify_int_vals(sf, symbs, [14, 12, 11, 9, 7, 6, 5])
        expected = sympify(
            "NOTsfGT7 + sfEQ9 + (sfGEQ11 * NOTsfGT12) + sfEQ14",
            locals=converter._vars_to_symbols,
        )
        self.assertEqual(int_simple, expected)
        self.assertEqual(
            converter._vars_to_conds,
            {
                "sfGT7": GtCondition(sf, 7),
                "sfEQ9": EqCondition(sf, 9),
                "sfGEQ11": GeqCondition(sf, 11),
                "sfGT12": GtCondition(sf, 12),
                "sfEQ14": EqCondition(sf, 14),
            },
        )

        self.assertEqual(
            converter._vars_to_symbols,
            {
                "sfGT7": Symbol("sfGT7"),
                "sfEQ9": Symbol("sfEQ9"),
                "sfGEQ11": Symbol("sfGEQ11"),
                "sfGT12": Symbol("sfGT12"),
                "sfEQ14": Symbol("sfEQ14"),
                "NOTsfGT7": Symbol("NOTsfGT7"),
                "NOTsfGT12": Symbol("NOTsfGT12"),
            },
        )


class BuildOrConditionTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        sf = StateFactor("sf", ["a", "b", "c", "d", "e"])

        sympy_add = converter._build_or_condition(sf, ["a", "c", "e"])
        self.assertEqual(
            sympy_add,
            sympify("sfEQa + sfEQc + sfEQe", locals=converter._vars_to_symbols),
        )

        self.assertEqual(
            converter._vars_to_conds,
            {
                "sfEQa": EqCondition(sf, "a"),
                "sfEQc": EqCondition(sf, "c"),
                "sfEQe": EqCondition(sf, "e"),
            },
        )
        self.assertEqual(
            converter._vars_to_symbols,
            {
                "sfEQa": Symbol("sfEQa"),
                "sfEQc": Symbol("sfEQc"),
                "sfEQe": Symbol("sfEQe"),
            },
        )


class BuildAndNotConditionTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        sf = StateFactor("sf", ["a", "b", "c", "d", "e"])

        sympy_add = converter._build_and_not_condition(sf, ["a", "c", "e"])
        self.assertEqual(
            sympy_add,
            sympify("NOTsfEQb * NOTsfEQd", locals=converter._vars_to_symbols),
        )

        self.assertEqual(
            converter._vars_to_conds,
            {
                "sfEQb": EqCondition(sf, "b"),
                "sfEQd": EqCondition(sf, "d"),
            },
        )
        self.assertEqual(
            converter._vars_to_symbols,
            {
                "sfEQb": Symbol("sfEQb"),
                "NOTsfEQb": Symbol("NOTsfEQb"),
                "sfEQd": Symbol("sfEQd"),
                "NOTsfEQd": Symbol("NOTsfEQd"),
            },
        )


class SimplifySymbolsTest(unittest.TestCase):

    def test_one(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)

        converter._vars_to_conds = {
            "sf1EQa": EqCondition(sf1, "a"),
            "sf1EQb": EqCondition(sf1, "b"),
            "sf1EQc": EqCondition(sf1, "c"),
            "sf2EQ5": EqCondition(sf2, 5),
            "sf2EQ7": EqCondition(sf2, 7),
            "sf2EQ9": EqCondition(sf2, 9),
        }
        converter._vars_to_symbols = {
            "sf1EQa": Symbol("sf1EQa"),
            "sf1EQb": Symbol("sf1EQb"),
            "sf1EQc": Symbol("sf1EQc"),
            "sf2EQ5": Symbol("sf2EQ5"),
            "sf2EQ7": Symbol("sf2EQ7"),
            "sf2EQ9": Symbol("sf2EQ9"),
        }
        symbols = list(converter._vars_to_symbols.values())

        simplified = converter._simplify_symbols(symbols, False)
        self.assertEqual(len(simplified), 4)
        self.assertTrue(sympify(1) in simplified)
        self.assertTrue(Symbol("sf2EQ5") in simplified)
        self.assertTrue(Symbol("sf2EQ7") in simplified)
        self.assertTrue(Symbol("sf2EQ9") in simplified)

    def test_two(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)

        converter._vars_to_conds = {
            "sf1EQa": EqCondition(sf1, "a"),
            "sf1EQb": EqCondition(sf1, "b"),
            "sf1EQc": EqCondition(sf1, "c"),
            "sf2EQ5": EqCondition(sf2, 5),
            "sf2EQ7": EqCondition(sf2, 7),
            "sf2EQ9": EqCondition(sf2, 9),
        }
        converter._vars_to_symbols = {
            "sf1EQa": Symbol("sf1EQa"),
            "sf1EQb": Symbol("sf1EQb"),
            "sf1EQc": Symbol("sf1EQc"),
            "sf2EQ5": Symbol("sf2EQ5"),
            "sf2EQ7": Symbol("sf2EQ7"),
            "sf2EQ9": Symbol("sf2EQ9"),
        }
        symbols = list(converter._vars_to_symbols.values())

        simplified = converter._simplify_symbols(symbols, True)
        self.assertEqual(len(simplified), 2)
        self.assertEqual(simplified, [sympify(0), sympify(0)])

    def test_three(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = IntStateFactor("sf2", 5, 10)

        converter._vars_to_conds = {
            "sf1EQa": EqCondition(sf1, "a"),
            "sf1EQb": EqCondition(sf1, "b"),
            "sf1EQc": EqCondition(sf1, "c"),
            "sf2EQ5": EqCondition(sf2, 5),
            "sf2EQ6": EqCondition(sf2, 6),
            "sf2EQ7": EqCondition(sf2, 7),
        }
        converter._vars_to_symbols = {
            "sf1EQa": Symbol("sf1EQa"),
            "sf1EQb": Symbol("sf1EQb"),
            "sf1EQc": Symbol("sf1EQc"),
            "sf2EQ5": Symbol("sf2EQ5"),
            "sf2EQ6": Symbol("sf2EQ6"),
            "sf2EQ7": Symbol("sf2EQ7"),
        }
        symbols = list(converter._vars_to_symbols.values())

        simplified = converter._simplify_symbols(symbols, False)
        self.assertEqual(len(simplified), 2)
        self.assertTrue(sympify(1) in simplified)
        self.assertTrue(Symbol("NOTsf2GT7") in simplified)

        self.assertEqual(converter._vars_to_conds["sf2GT7"], GtCondition(sf2, 7))
        self.assertEqual(converter._vars_to_symbols["sf2GT7"], Symbol("sf2GT7"))
        self.assertEqual(converter._vars_to_symbols["NOTsf2GT7"], Symbol("NOTsf2GT7"))

    def test_four(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c"])

        converter._vars_to_conds = {
            "sf1EQa": EqCondition(sf1, "a"),
            "sf1EQb": EqCondition(sf1, "b"),
        }
        converter._vars_to_symbols = {
            "sf1EQa": Symbol("sf1EQa"),
            "sf1EQb": Symbol("sf1EQb"),
            "NOTsf1EQa": Symbol("NOTsf1EQa"),
            "NOTsf1EQb": Symbol("NOTsf1EQb"),
        }
        symbols = [Symbol("NOTsf1EQa"), Symbol("NOTsf1EQb")]

        simplified = converter._simplify_symbols(symbols, True)
        self.assertEqual(simplified, [Symbol("sf1EQc")])

        self.assertEqual(converter._vars_to_conds["sf1EQc"], EqCondition(sf1, "c"))
        self.assertEqual(converter._vars_to_symbols["sf1EQc"], Symbol("sf1EQc"))

    def test_five(self):

        converter = PolicyBTConverter()

        sf1 = StateFactor("sf1", ["a", "b", "c", "d", "e"])

        converter._vars_to_conds = {
            "sf1EQa": EqCondition(sf1, "a"),
            "sf1EQc": EqCondition(sf1, "c"),
            "sf1EQd": EqCondition(sf1, "d"),
            "sf1EQe": EqCondition(sf1, "e"),
        }
        converter._vars_to_symbols = {
            "sf1EQa": Symbol("sf1EQa"),
            "sf1EQc": Symbol("sf1EQc"),
            "sf1EQd": Symbol("sf1EQd"),
            "sf1EQe": Symbol("sf1EQe"),
        }
        symbols = list(converter._vars_to_symbols.values())

        simplified = converter._simplify_symbols(symbols, False)
        self.assertEqual(simplified, [Symbol("NOTsf1EQb")])

        self.assertEqual(converter._vars_to_conds["sf1EQb"], EqCondition(sf1, "b"))
        self.assertEqual(converter._vars_to_symbols["sf1EQb"], Symbol("sf1EQb"))
        self.assertEqual(converter._vars_to_symbols["NOTsf1EQb"], Symbol("NOTsf1EQb"))


class SimplifyUsingStateFactorInfoTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()

        self.assertEqual(
            converter._simplify_using_state_factor_info(Symbol("v1")), Symbol("v1")
        )

        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        loc = StateFactor("loc", ["v1", "v2", "v3", "v4", "v5"])
        busy = StateFactor("busy", ["yes", "no"])

        converter._vars_to_conds["locEQv1"] = EqCondition(loc, "v1")
        converter._vars_to_conds["locEQv2"] = EqCondition(loc, "v2")
        converter._vars_to_conds["locEQv3"] = EqCondition(loc, "v3")
        converter._vars_to_conds["busyEQyes"] = EqCondition(busy, "yes")
        converter._vars_to_conds["busyEQno"] = EqCondition(busy, "no")

        converter._vars_to_symbols["locEQv1"] = Symbol("locEQv1")
        converter._vars_to_symbols["locEQv2"] = Symbol("locEQv2")
        converter._vars_to_symbols["locEQv3"] = Symbol("locEQv3")
        converter._vars_to_symbols["busyEQyes"] = Symbol("busyEQyes")
        converter._vars_to_symbols["busyEQno"] = Symbol("busyEQno")

        expr_str = (
            "(busyEQyes + busyEQno) * "
            + "(locEQv1 + locEQv2 + locEQv3) * "
            + "((locEQv1 * busyEQyes) + locEQv2)"
        )

        expression = sympify(expr_str, locals=converter._vars_to_symbols)

        simple_expression = converter._simplify_using_state_factor_info(expression)

        expected = sympify(
            "(NOTlocEQv4 * NOTlocEQv5) * ((locEQv1 * busyEQyes) + locEQv2)",
            locals=converter._vars_to_symbols,
        )

        self.assertEqual(simple_expression, expected)

        self.assertEqual(len(converter._vars_to_conds), 7)
        self.assertEqual(converter._vars_to_conds["locEQv4"], EqCondition(loc, "v4"))
        self.assertEqual(converter._vars_to_conds["locEQv5"], EqCondition(loc, "v5"))

        self.assertEqual(len(converter._vars_to_symbols), 9)
        self.assertEqual(converter._vars_to_symbols["locEQv4"], Symbol("locEQv4"))
        self.assertEqual(converter._vars_to_symbols["locEQv5"], Symbol("locEQv5"))
        self.assertEqual(converter._vars_to_symbols["NOTlocEQv4"], Symbol("NOTlocEQv4"))
        self.assertEqual(converter._vars_to_symbols["NOTlocEQv5"], Symbol("NOTlocEQv5"))


class SimplifyUsingStateFactorsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()

        sf = StateFactor("sf", ["a", "b", "c"])

        converter._vars_to_conds = {
            "sfEQa": EqCondition(sf, "a"),
            "sfEQb": EqCondition(sf, "b"),
            "sfEQc": EqCondition(sf, "c"),
        }
        converter._vars_to_symbols = {
            "sfEQa": Symbol("sfEQa"),
            "sfEQb": Symbol("sfEQb"),
            "sfEQc": Symbol("sfEQc"),
        }

        expr_1 = sympify("sfEQa + sfEQb + sfEQc", locals=converter._vars_to_symbols)
        expr_2 = sympify("sfEQa * sfEQb * sfEQc", locals=converter._vars_to_symbols)
        min_alg_act_pairs = [(expr_1, "a1"), (expr_2, "a2")]

        simple_alg_act_pairs = converter._simplify_using_state_factors(
            min_alg_act_pairs
        )

        self.assertEqual(simple_alg_act_pairs, [(sympify(1), "a1"), (sympify(0), "a2")])


class AlgebraToLogicTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        sf = StateFactor("sf", ["a", "b", "c", "d"])

        converter._vars_to_conds = {
            "a": EqCondition(sf, "a"),
            "b": EqCondition(sf, "b"),
            "c": EqCondition(sf, "c"),
            "d": EqCondition(sf, "d"),
        }

        converter._vars_to_symbols = {
            "a": Symbol("a"),
            "b": Symbol("b"),
            "c": Symbol("c"),
            "d": Symbol("d"),
        }

        e = sympify("(a + b) * (NOTc + d)", locals=converter._vars_to_symbols)

        rule = converter._algebra_to_logic(e)

        self.assertTrue(rule.equivalent(expr("(a | b) & (~c | d)")))

        converter._vars_to_symbols = {
            "a": Symbol("a"),
            "b": Symbol("b"),
            "d": Symbol("d"),
        }

        with self.assertRaises(AssertionError):
            converter._algebra_to_logic(e)


class SympyAlgebraicToPyedaRules(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        sf = StateFactor("sf", ["a", "b", "c", "d"])

        converter._vars_to_conds = {
            "a": EqCondition(sf, "a"),
            "b": EqCondition(sf, "b"),
            "c": EqCondition(sf, "c"),
            "d": EqCondition(sf, "d"),
        }

        converter._vars_to_symbols = {
            "a": Symbol("a"),
            "b": Symbol("b"),
            "c": Symbol("c"),
            "d": Symbol("d"),
        }

        e1 = sympify("(a + b) * (NOTc + d)", locals=converter._vars_to_symbols)
        e2 = sympify("(NOTa * NOTb) + (c * NOTd)", locals=converter._vars_to_symbols)

        sympy_act_pairs = [(e1, "a1"), (e2, "a2")]

        pyeda_act_pairs = converter._sympy_algebraic_to_pyeda_rules(sympy_act_pairs)

        r1 = expr("(a | b) & (~c | d)")
        r2 = expr("(~a & ~b) | (c & ~d)")

        self.assertEqual(len(pyeda_act_pairs), 2)
        self.assertEqual(pyeda_act_pairs[0][1], "a1")
        self.assertEqual(pyeda_act_pairs[1][1], "a2")
        self.assertTrue(pyeda_act_pairs[0][0].equivalent(r1))
        self.assertTrue(pyeda_act_pairs[1][0].equivalent(r2))


class MinimiseRuleActPairsTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}
        converter._vars_to_symbols = {}

        loc = StateFactor("loc", ["v1", "v2", "v3", "v4", "v5"])
        busy = StateFactor("busy", ["yes", "no"])

        converter._vars_to_conds["locEQv1"] = EqCondition(loc, "v1")
        converter._vars_to_conds["locEQv2"] = EqCondition(loc, "v2")
        converter._vars_to_conds["locEQv3"] = EqCondition(loc, "v3")
        converter._vars_to_conds["busyEQyes"] = EqCondition(busy, "yes")
        converter._vars_to_conds["busyEQno"] = EqCondition(busy, "no")

        converter._vars_to_symbols["locEQv1"] = Symbol("locEQv1")
        converter._vars_to_symbols["locEQv2"] = Symbol("locEQv2")
        converter._vars_to_symbols["locEQv3"] = Symbol("locEQv3")
        converter._vars_to_symbols["busyEQyes"] = Symbol("busyEQyes")
        converter._vars_to_symbols["busyEQno"] = Symbol("busyEQno")

        rule_1 = expr("(locEQv1 & busyEQyes) | (locEQv1 & busyEQno)")

        rule_2 = expr("locEQv1 | locEQv2")

        ra_pairs = [(rule_1, "a1"), (rule_2, "a2")]

        new_ra_pairs = converter._minimise_rule_act_pairs(ra_pairs)
        self.assertEqual(len(ra_pairs), 2)
        self.assertEqual(new_ra_pairs[0][1], "a1")
        self.assertEqual(new_ra_pairs[1][1], "a2")

        expected_1 = sympify(
            "locEQv1",
            locals=converter._vars_to_symbols,
        )
        expected_2 = sympify("locEQv1 + locEQv2", locals=converter._vars_to_symbols)

        self.assertEqual(new_ra_pairs[0][0], expected_1)
        self.assertEqual(new_ra_pairs[1][0], expected_2)

        new_ra_pairs = converter._minimise_rule_act_pairs(ra_pairs, True)
        self.assertEqual(len(ra_pairs), 2)
        self.assertEqual(new_ra_pairs[0][1], "a1")
        self.assertEqual(new_ra_pairs[1][1], "a2")
        self.assertTrue(new_ra_pairs[0][0].equivalent(expr("locEQv1")))
        self.assertTrue(new_ra_pairs[1][0].equivalent(expr("locEQv1 | locEQv2")))


class BuildConditionNodeTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        converter._vars_to_conds = {}

        sf = IntStateFactor("sf", 0, 5)

        converter._vars_to_conds["sfEQ1"] = EqCondition(sf, 1)
        converter._vars_to_conds["sfGT1"] = GtCondition(sf, 1)
        converter._vars_to_conds["sfGEQ1"] = GeqCondition(sf, 1)

        node = converter._build_condition_node("sfEQ1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfEQ1")
        self.assertEqual(node.get_cond(), EqCondition(sf, 1))

        node = converter._build_condition_node("sfGT1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfGT1")
        self.assertEqual(node.get_cond(), GtCondition(sf, 1))

        node = converter._build_condition_node("sfGEQ1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfGEQ1")
        self.assertEqual(node.get_cond(), GeqCondition(sf, 1))

        node = converter._build_condition_node("NOTsfEQ1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfNEQ1")
        self.assertEqual(node.get_cond(), NeqCondition(sf, 1))

        node = converter._build_condition_node("NOTsfGT1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfLEQ1")
        self.assertEqual(node.get_cond(), LeqCondition(sf, 1))

        node = converter._build_condition_node("NOTsfGEQ1")
        self.assertTrue(isinstance(node, ConditionNode))
        self.assertEqual(node.get_name(), "sfLT1")
        self.assertEqual(node.get_cond(), LtCondition(sf, 1))


class SubBTForRuleTest(unittest.TestCase):

    def test_function(self):

        converter = PolicyBTConverter()
        symbols = ["sfEQ1", "sfGT3", "sfEQ2", "sfGEQ4", "sfEQ4", "NOTsfGT3"]
        converter._vars_to_symbols = {s: Symbol(s) for s in symbols}

        sf = IntStateFactor("sf", 1, 5)

        converter._vars_to_conds = {
            "sfEQ1": EqCondition(sf, 1),
            "sfGT3": GtCondition(sf, 3),
            "sfEQ2": EqCondition(sf, 2),
            "sfGEQ4": GeqCondition(sf, 4),
            "sfEQ4": EqCondition(sf, 4),
        }

        rule = sympify(
            "sfEQ1 + NOTsfGT3*sfEQ2 + sfGEQ4*sfEQ4", locals=converter._vars_to_symbols
        )
        bt = converter._sub_bt_for_rule(rule)

        self.assertTrue(isinstance(bt, FallbackNode))
        self.assertEqual(len(bt._children), 3)

        for i in range(3):
            if "sfEQ1" in str(rule.args[i]):
                self.assertTrue(isinstance(bt._children[i], ConditionNode))
                self.assertEqual(bt._children[i].get_name(), "sfEQ1")
                self.assertEqual(bt._children[i].get_cond(), EqCondition(sf, 1))
            elif "NOTsfGT3" in str(rule.args[i]):
                self.assertTrue(isinstance(bt._children[i], SequenceNode))
                self.assertEqual(len(bt._children[i]._children), 2)
                self.assertTrue(isinstance(bt._children[i]._children[0], ConditionNode))
                self.assertTrue(isinstance(bt._children[i]._children[1], ConditionNode))
                if str(rule.args[i].args[0])[:3] == "NOT":
                    leq_node = 0
                    eq_node = 1
                else:
                    leq_node = 1
                    eq_node = 0
                self.assertEqual(
                    bt._children[i]._children[leq_node].get_name(), "sfLEQ3"
                )
                self.assertEqual(
                    bt._children[i]._children[leq_node].get_cond(), LeqCondition(sf, 3)
                )
                self.assertEqual(bt._children[i]._children[eq_node].get_name(), "sfEQ2")
                self.assertEqual(
                    bt._children[i]._children[eq_node].get_cond(), EqCondition(sf, 2)
                )
            elif "sfGEQ4" in str(rule.args[i]):
                self.assertTrue(isinstance(bt._children[i], SequenceNode))
                self.assertEqual(len(bt._children[i]._children), 2)
                self.assertTrue(isinstance(bt._children[i]._children[0], ConditionNode))
                self.assertTrue(isinstance(bt._children[i]._children[1], ConditionNode))
                if str(rule.args[i].args[0]) == "sfGEQ4":
                    geq_node = 0
                    eq_node = 1
                else:
                    geq_node = 1
                    eq_node = 0
                self.assertEqual(
                    bt._children[i]._children[geq_node].get_name(), "sfGEQ4"
                )
                self.assertEqual(
                    bt._children[i]._children[geq_node].get_cond(), GeqCondition(sf, 4)
                )
                self.assertEqual(bt._children[i]._children[eq_node].get_name(), "sfEQ4")
                self.assertEqual(
                    bt._children[i]._children[eq_node].get_cond(), EqCondition(sf, 4)
                )


class ConvertRulesToBTTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter()
        symbols = [
            "sfEQ1",
            "sfGT3",
            "sfEQ2",
            "sfGEQ4",
            "sfEQ4",
            "NOTsfGT3",
            "sfGEQ0",
            "sfEQ5",
        ]
        converter._vars_to_symbols = {s: Symbol(s) for s in symbols}

        sf = IntStateFactor("sf", 0, 5)

        converter._vars_to_conds = {
            "sfEQ1": EqCondition(sf, 1),
            "sfGT3": GtCondition(sf, 3),
            "sfEQ2": EqCondition(sf, 2),
            "sfGEQ4": GeqCondition(sf, 4),
            "sfEQ4": EqCondition(sf, 4),
            "sfGEQ0": GeqCondition(sf, 0),
            "sfEQ5": EqCondition(sf, 5),
        }

        rule_1 = sympify(
            "sfEQ1 + NOTsfGT3*sfEQ2 + sfGEQ4*sfEQ4", locals=converter._vars_to_symbols
        )
        rule_2 = sympify("sfGEQ0*sfEQ5", locals=converter._vars_to_symbols)

        min_alg_act_pairs = [(rule_1, "a1"), (rule_2, "a2")]

        bt = converter._convert_rules_to_bt(min_alg_act_pairs)
        self.assertTrue(isinstance(bt, BehaviourTree))

        root = bt.get_root_node()
        self.assertTrue(isinstance(root, SequenceNode))
        self.assertEqual(len(root._children), 2)

        sub_1 = root._children[0]

        self.assertTrue(isinstance(sub_1, FallbackNode))
        self.assertEqual(len(sub_1._children), 4)

        for i in range(3):
            if "sfEQ1" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], ConditionNode))
                self.assertEqual(sub_1._children[i].get_name(), "sfEQ1")
                self.assertEqual(sub_1._children[i].get_cond(), EqCondition(sf, 1))
            elif "NOTsfGT3" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], SequenceNode))
                self.assertEqual(len(sub_1._children[i]._children), 2)
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[0], ConditionNode)
                )
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[1], ConditionNode)
                )
                if str(rule_1.args[i].args[0])[:3] == "NOT":
                    leq_node = 0
                    eq_node = 1
                else:
                    leq_node = 1
                    eq_node = 0
                self.assertEqual(
                    sub_1._children[i]._children[leq_node].get_name(), "sfLEQ3"
                )
                self.assertEqual(
                    sub_1._children[i]._children[leq_node].get_cond(),
                    LeqCondition(sf, 3),
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_name(), "sfEQ2"
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_cond(), EqCondition(sf, 2)
                )
            elif "sfGEQ4" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], SequenceNode))
                self.assertEqual(len(sub_1._children[i]._children), 2)
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[0], ConditionNode)
                )
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[1], ConditionNode)
                )
                if str(rule_1.args[i].args[0]) == "sfGEQ4":
                    geq_node = 0
                    eq_node = 1
                else:
                    geq_node = 1
                    eq_node = 0
                self.assertEqual(
                    sub_1._children[i]._children[geq_node].get_name(), "sfGEQ4"
                )
                self.assertEqual(
                    sub_1._children[i]._children[geq_node].get_cond(),
                    GeqCondition(sf, 4),
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_name(), "sfEQ4"
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_cond(), EqCondition(sf, 4)
                )

        self.assertTrue(isinstance(sub_1._children[3], ActionNode))
        self.assertEqual(sub_1._children[3].get_name(), "a1")

        sub_2 = root._children[1]

        self.assertTrue(isinstance(sub_2, FallbackNode))
        self.assertEqual(len(sub_2._children), 2)

        self.assertTrue(isinstance(sub_2._children[0], SequenceNode))
        self.assertEqual(len(sub_2._children[0]._children), 2)
        geq_node = 0
        eq_node = 1
        if str(rule_2.args[0]) == "sfEQ5":
            eq_node = 0
            geq_node = 1

        self.assertTrue(
            isinstance(sub_2._children[0]._children[geq_node], ConditionNode)
        )
        self.assertEqual(sub_2._children[0]._children[geq_node].get_name(), "sfGEQ0")
        self.assertEqual(
            sub_2._children[0]._children[geq_node].get_cond(), GeqCondition(sf, 0)
        )

        self.assertTrue(
            isinstance(sub_2._children[0]._children[eq_node], ConditionNode)
        )
        self.assertEqual(sub_2._children[0]._children[eq_node].get_name(), "sfEQ5")
        self.assertEqual(
            sub_2._children[0]._children[eq_node].get_cond(), EqCondition(sf, 5)
        )

        self.assertTrue(isinstance(sub_2._children[1], ActionNode))
        self.assertEqual(sub_2._children[1].get_name(), "a2")

    def test_default_action(self):
        converter = PolicyBTConverter("default_action")
        symbols = [
            "sfEQ1",
            "sfGT3",
            "sfEQ2",
            "sfGEQ4",
            "sfEQ4",
            "NOTsfGT3",
            "sfGEQ0",
            "sfEQ5",
        ]
        converter._vars_to_symbols = {s: Symbol(s) for s in symbols}

        sf = IntStateFactor("sf", 0, 5)

        converter._vars_to_conds = {
            "sfEQ1": EqCondition(sf, 1),
            "sfGT3": GtCondition(sf, 3),
            "sfEQ2": EqCondition(sf, 2),
            "sfGEQ4": GeqCondition(sf, 4),
            "sfEQ4": EqCondition(sf, 4),
            "sfGEQ0": GeqCondition(sf, 0),
            "sfEQ5": EqCondition(sf, 5),
        }

        rule_1 = sympify(
            "sfEQ1 + NOTsfGT3*sfEQ2 + sfGEQ4*sfEQ4", locals=converter._vars_to_symbols
        )
        rule_2 = sympify("sfGEQ0*sfEQ5", locals=converter._vars_to_symbols)

        min_alg_act_pairs = [(rule_1, "a1"), (rule_2, "a2")]

        converter._default_needed = True
        bt = converter._convert_rules_to_bt(min_alg_act_pairs)
        self.assertTrue(isinstance(bt, BehaviourTree))

        root = bt.get_root_node()
        self.assertTrue(isinstance(root, SequenceNode))
        self.assertEqual(len(root._children), 3)

        sub_1 = root._children[0]

        self.assertTrue(isinstance(sub_1, FallbackNode))
        self.assertEqual(len(sub_1._children), 4)

        for i in range(3):
            if "sfEQ1" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], ConditionNode))
                self.assertEqual(sub_1._children[i].get_name(), "sfEQ1")
                self.assertEqual(sub_1._children[i].get_cond(), EqCondition(sf, 1))
            elif "NOTsfGT3" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], SequenceNode))
                self.assertEqual(len(sub_1._children[i]._children), 2)
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[0], ConditionNode)
                )
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[1], ConditionNode)
                )
                if str(rule_1.args[i].args[0])[:3] == "NOT":
                    leq_node = 0
                    eq_node = 1
                else:
                    leq_node = 1
                    eq_node = 0
                self.assertEqual(
                    sub_1._children[i]._children[leq_node].get_name(), "sfLEQ3"
                )
                self.assertEqual(
                    sub_1._children[i]._children[leq_node].get_cond(),
                    LeqCondition(sf, 3),
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_name(), "sfEQ2"
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_cond(), EqCondition(sf, 2)
                )
            elif "sfGEQ4" in str(rule_1.args[i]):
                self.assertTrue(isinstance(sub_1._children[i], SequenceNode))
                self.assertEqual(len(sub_1._children[i]._children), 2)
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[0], ConditionNode)
                )
                self.assertTrue(
                    isinstance(sub_1._children[i]._children[1], ConditionNode)
                )
                if str(rule_1.args[i].args[0]) == "sfGEQ4":
                    geq_node = 0
                    eq_node = 1
                else:
                    geq_node = 1
                    eq_node = 0
                self.assertEqual(
                    sub_1._children[i]._children[geq_node].get_name(), "sfGEQ4"
                )
                self.assertEqual(
                    sub_1._children[i]._children[geq_node].get_cond(),
                    GeqCondition(sf, 4),
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_name(), "sfEQ4"
                )
                self.assertEqual(
                    sub_1._children[i]._children[eq_node].get_cond(), EqCondition(sf, 4)
                )

        self.assertTrue(isinstance(sub_1._children[3], ActionNode))
        self.assertEqual(sub_1._children[3].get_name(), "a1")

        sub_2 = root._children[1]

        self.assertTrue(isinstance(sub_2, FallbackNode))
        self.assertEqual(len(sub_2._children), 2)

        self.assertTrue(isinstance(sub_2._children[0], SequenceNode))
        self.assertEqual(len(sub_2._children[0]._children), 2)
        geq_node = 0
        eq_node = 1
        if str(rule_2.args[0]) == "sfEQ5":
            eq_node = 0
            geq_node = 1

        self.assertTrue(
            isinstance(sub_2._children[0]._children[geq_node], ConditionNode)
        )
        self.assertEqual(sub_2._children[0]._children[geq_node].get_name(), "sfGEQ0")
        self.assertEqual(
            sub_2._children[0]._children[geq_node].get_cond(), GeqCondition(sf, 0)
        )

        self.assertTrue(
            isinstance(sub_2._children[0]._children[eq_node], ConditionNode)
        )
        self.assertEqual(sub_2._children[0]._children[eq_node].get_name(), "sfEQ5")
        self.assertEqual(
            sub_2._children[0]._children[eq_node].get_cond(), EqCondition(sf, 5)
        )

        self.assertTrue(isinstance(sub_2._children[1], ActionNode))
        self.assertEqual(sub_2._children[1].get_name(), "a2")

        sub_3 = root._children[2]
        self.assertTrue(isinstance(sub_3, ActionNode))
        self.assertEqual(sub_3.get_name(), "default_action")


class ConvertPolicyTest(unittest.TestCase):

    def test_function(self):
        converter = PolicyBTConverter(default_action="a3")

        sf1 = StateFactor("sf1", ["a", "b", "c"])
        sf2 = StateFactor("sf2", ["d", "e", "f"])

        state_action_map = {}
        state_action_map[State({sf1: "a", sf2: "d"})] = "a1"
        state_action_map[State({sf1: "a", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "a", sf2: "f"})] = None
        state_action_map[State({sf1: "b", sf2: "d"})] = "a2"
        state_action_map[State({sf1: "b", sf2: "e"})] = None
        state_action_map[State({sf1: "b", sf2: "f"})] = "a1"
        state_action_map[State({sf1: "c", sf2: "d"})] = None
        state_action_map[State({sf1: "c", sf2: "e"})] = "a2"
        state_action_map[State({sf1: "c", sf2: "f"})] = "a1"

        policy = Policy(state_action_map)
        out_file = "/tmp/test_bt.xml"

        bt = converter.convert_policy(policy, out_file)

        self.assertTrue(os.path.exists("/tmp/test_bt.xml"))
        os.unlink("/tmp/test_bt.xml")

        # Now test the BT bt checking which action is returned for each state
        self.assertEqual(bt.tick_at_state(State({sf1: "a", sf2: "d"})), "a1")
        self.assertEqual(bt.tick_at_state(State({sf1: "a", sf2: "e"})), "a2")
        self.assertEqual(bt.tick_at_state(State({sf1: "a", sf2: "f"})), "a3")
        self.assertEqual(bt.tick_at_state(State({sf1: "b", sf2: "d"})), "a2")
        self.assertEqual(bt.tick_at_state(State({sf1: "b", sf2: "e"})), "a3")
        self.assertEqual(bt.tick_at_state(State({sf1: "b", sf2: "f"})), "a1")
        self.assertEqual(bt.tick_at_state(State({sf1: "c", sf2: "d"})), "a3")
        self.assertEqual(bt.tick_at_state(State({sf1: "c", sf2: "e"})), "a2")
        self.assertEqual(bt.tick_at_state(State({sf1: "c", sf2: "f"})), "a1")


if __name__ == "__main__":
    unittest.main()
