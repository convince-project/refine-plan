#!/usr/bin/env python3
""" Unit tests for policy.py. 

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import unittest


class PolicyTest(unittest.TestCase):

    def test_function(self):
        loc = StateFactor("loc", ["v1", "v2", "v3"])
        battery = StateFactor("battery", ["low", "med", "high"])

        state_action_dict = {
            State({loc: "v1", battery: "high"}): "move",
            State({loc: "v2", battery: "med"}): "charge",
            State({loc: "v3", battery: "low"}): "stop",
        }

        policy = Policy(state_action_dict)
        with self.assertRaises(Exception):
            policy.get_value(State({loc: "v1", battery: "high"}))

        value_dict = {
            State({loc: "v1", battery: "high"}): 3,
            State({loc: "v2", battery: "med"}): 1,
            State({loc: "v3", battery: "low"}): 2,
        }

        policy = Policy(state_action_dict, value_dict=value_dict)
        for l in loc.get_valid_values():
            for b in battery.get_valid_values():
                state = State({loc: l, battery: b})
                if l == "v1" and b == "high":
                    self.assertEqual(policy.get_action(state), "move")
                    self.assertEqual(policy[state], "move")
                    self.assertEqual(policy.get_value(state), 3)
                elif l == "v2" and b == "med":
                    self.assertEqual(policy.get_action(state), "charge")
                    self.assertEqual(policy[state], "charge")
                    self.assertEqual(policy.get_value(state), 1)
                elif l == "v3" and b == "low":
                    self.assertEqual(policy.get_action(state), "stop")
                    self.assertEqual(policy[state], "stop")
                    self.assertEqual(policy.get_value(state), 2)
                else:
                    self.assertEqual(policy.get_action(state), None)
                    self.assertEqual(policy[state], None)
                    self.assertEqual(policy.get_value(state), None)


if __name__ == "__main__":
    unittest.main()
