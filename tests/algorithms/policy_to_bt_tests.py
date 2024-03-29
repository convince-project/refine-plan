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


class ConvertPolicyTest(unittest.TestCase):

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
