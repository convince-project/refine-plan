#!/usr/bin/env python3
""" Unit tests for policy.py. 

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, IntStateFactor, BoolStateFactor
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import unittest
import tempfile


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


class ReadWriteTest(unittest.TestCase):

    def test_function(self):
        loc = StateFactor("loc", ["v1", "v2", "v3"])
        door = BoolStateFactor("door")
        battery = IntStateFactor("battery", 0, 10)

        state_action_dict = {
            State({loc: "v1", door: False, battery: 5}): "a1",
            State({loc: "v2", door: True, battery: 9}): "a2",
            State({loc: "v3", door: False, battery: 1}): None,
        }

        value_dict = {
            State({loc: "v1", door: False, battery: 5}): 1.0,
            State({loc: "v3", door: False, battery: 1}): 5.0,
        }
        policy = Policy(state_action_dict, value_dict=value_dict)

        tmp = tempfile.NamedTemporaryFile()
        policy.write_policy(tmp.name)
        read_policy = Policy({}, policy_file=tmp.name)

        sf_dict = list(read_policy._state_action_dict.keys())[0]._sf_dict
        self.assertEqual(len(sf_dict), 3)
        self.assertEqual(sf_dict["loc"], loc)
        self.assertEqual(sf_dict["door"], door)
        self.assertEqual(sf_dict["battery"], battery)

        self.assertEqual(
            set(policy._state_action_dict.keys()),
            set(read_policy._state_action_dict.keys()),
        )

        self.assertEqual(
            set(policy._value_dict.keys()),
            set(read_policy._value_dict.keys()),
        )

        for state in policy._state_action_dict:
            self.assertEqual(policy[state], read_policy[state])

        for state in policy._value_dict:
            self.assertEqual(policy.get_value(state), read_policy.get_value(state))

        # Now test without the value function
        policy = Policy(state_action_dict)

        tmp = tempfile.NamedTemporaryFile()
        policy.write_policy(tmp.name)
        read_policy = Policy({}, policy_file=tmp.name)

        self.assertEqual(
            set(policy._state_action_dict.keys()),
            set(read_policy._state_action_dict.keys()),
        )

        self.assertEqual(read_policy._value_dict, None)

        for state in policy._state_action_dict:
            self.assertEqual(policy[state], read_policy[state])


if __name__ == "__main__":
    unittest.main()
