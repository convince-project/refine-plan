#!/usr/bin/env python3
""" Unit tests for the policy->BT converter.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.condition import EqCondition
from pyeda.boolalg.expr import expr, And, Or, Not
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
from pyeda.inter import espresso_exprs
from sympy import Symbol
import unittest


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


if __name__ == "__main__":
    unittest.main()
