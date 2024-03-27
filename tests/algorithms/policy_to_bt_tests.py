#!/usr/bin/env python3
""" Unit tests for the policy->BT converter.

Author: Charlie Street
Owner: Charlie Street
"""

from pyeda.boolalg.expr import expr, And, Or
from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.state_factor import StateFactor
from refine_plan.models.condition import EqCondition
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
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


if __name__ == "__main__":
    unittest.main()
