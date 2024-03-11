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
        for l in loc.get_valid_values():
            for b in battery.get_valid_values():
                state = State({loc: l, battery: b})
                if l == "v1" and b == "high":
                    self.assertEqual(policy.get_action(state), "move")
                    self.assertEqual(policy[state], "move")
                elif l == "v2" and b == "med":
                    self.assertEqual(policy.get_action(state), "charge")
                    self.assertEqual(policy[state], "charge")
                elif l == "v3" and b == "low":
                    self.assertEqual(policy.get_action(state), "stop")
                    self.assertEqual(policy[state], "stop")
                else:
                    self.assertEqual(policy.get_action(state), None)
                    self.assertEqual(policy[state], None)


if __name__ == "__main__":
    unittest.main()
