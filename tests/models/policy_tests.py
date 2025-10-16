#!/usr/bin/env python3
"""Unit tests for policy.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, IntStateFactor, BoolStateFactor
from refine_plan.models.policy import Policy, TimeDependentPolicy
from refine_plan.models.state import State
import unittest
import tempfile
import os


def generate_scxml_test_string():

    expected = "<?xml version='1.0' encoding='UTF-8'?>\n"
    expected += '<scxml initial="init" version="1.0" name="my_policy" model_src="" xmlns="http://www.w3.org/2005/07/scxml">\n'
    expected += "\t<datamodel>\n"
    expected += '\t\t<data id="sf1" expr="0" type="int32" />\n'
    expected += '\t\t<data id="sf2" expr="1" type="int32" />\n'
    expected += '\t\t<data id="sf3" expr="2" type="int32" />\n'
    expected += "\t</datamodel>\n"
    expected += '\t<state id="init">\n'
    expected += "\t\t<onentry>\n"
    expected += '\t\t\t<if cond="sf1==0">\n'
    expected += '\t\t\t\t<if cond="sf2==0">\n'
    expected += '\t\t\t\t\t<if cond="sf3==0">\n'
    expected += '\t\t\t\t\t\t<send event="a1" target="mdp" />\n'
    expected += '\t\t\t\t\t\t<elseif cond="sf3==2" />\n'
    expected += '\t\t\t\t\t\t<send event="a3" target="mdp" />\n'
    expected += "\t\t\t\t\t</if>\n"
    expected += '\t\t\t\t\t<elseif cond="sf2==1" />\n'
    expected += '\t\t\t\t\t<if cond="sf3==1">\n'
    expected += '\t\t\t\t\t\t<send event="a2" target="mdp" />\n'
    expected += "\t\t\t\t\t</if>\n"
    expected += "\t\t\t\t</if>\n"
    expected += '\t\t\t\t<elseif cond="sf1==1" />\n'
    expected += '\t\t\t\t<if cond="sf2==0">\n'
    expected += '\t\t\t\t\t<if cond="sf3==2">\n'
    expected += '\t\t\t\t\t\t<send event="a4" target="mdp" />\n'
    expected += "\t\t\t\t\t</if>\n"
    expected += "\t\t\t\t</if>\n"
    expected += "\t\t\t</if>\n"
    expected += "\t\t</onentry>\n"
    expected += '\t\t<transition target="init" event="update_datamodel">\n'
    expected += '\t\t\t<assign location="sf1" expr="_event.data.sf1" />\n'
    expected += '\t\t\t<assign location="sf2" expr="_event.data.sf2" />\n'
    expected += '\t\t\t<assign location="sf3" expr="_event.data.sf3" />\n'
    expected += "\t\t</transition>\n"
    expected += "\t</state>\n"
    expected += "</scxml>"

    return expected


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


class TimeDependentPolicyTest(unittest.TestCase):

    def test_function(self):
        loc = StateFactor("loc", ["v1", "v2", "v3"])
        battery = StateFactor("battery", ["low", "med", "high"])

        state_action_dict = {
            State({loc: "v1", battery: "high"}): "move",
            State({loc: "v2", battery: "med"}): "charge",
            State({loc: "v3", battery: "low"}): "stop",
        }

        state_action_dict_2 = {
            State({loc: "v1", battery: "high"}): "charge",
            State({loc: "v2", battery: "med"}): "stop",
            State({loc: "v3", battery: "low"}): "move",
        }

        policy = TimeDependentPolicy([state_action_dict, state_action_dict_2])
        with self.assertRaises(Exception):
            policy.get_value(State({loc: "v1", battery: "high"}), 0)

        value_dict = {
            State({loc: "v1", battery: "high"}): 3,
            State({loc: "v2", battery: "med"}): 1,
            State({loc: "v3", battery: "low"}): 2,
        }

        value_dict_2 = {
            State({loc: "v1", battery: "high"}): 1,
            State({loc: "v2", battery: "med"}): 2,
            State({loc: "v3", battery: "low"}): 3,
        }

        policy = TimeDependentPolicy(
            [state_action_dict, state_action_dict_2],
            value_dicts=[value_dict, value_dict_2],
        )
        for l in loc.get_valid_values():
            for b in battery.get_valid_values():
                state = State({loc: l, battery: b})
                if l == "v1" and b == "high":
                    self.assertEqual(policy.get_action(state, 0), "move")
                    self.assertEqual(policy[state, 0], "move")
                    self.assertEqual(policy.get_value(state, 0), 3)
                    self.assertEqual(policy.get_action(state, 1), "charge")
                    self.assertEqual(policy[state, 1], "charge")
                    self.assertEqual(policy.get_value(state, 1), 1)
                elif l == "v2" and b == "med":
                    self.assertEqual(policy.get_action(state, 0), "charge")
                    self.assertEqual(policy[state, 0], "charge")
                    self.assertEqual(policy.get_value(state, 0), 1)
                    self.assertEqual(policy.get_action(state, 1), "stop")
                    self.assertEqual(policy[state, 1], "stop")
                    self.assertEqual(policy.get_value(state, 1), 2)
                elif l == "v3" and b == "low":
                    self.assertEqual(policy.get_action(state, 0), "stop")
                    self.assertEqual(policy[state, 0], "stop")
                    self.assertEqual(policy.get_value(state, 0), 2)
                    self.assertEqual(policy.get_action(state, 1), "move")
                    self.assertEqual(policy[state, 1], "move")
                    self.assertEqual(policy.get_value(state, 1), 3)
                else:
                    self.assertEqual(policy.get_action(state, 0), None)
                    self.assertEqual(policy[state, 0], None)
                    self.assertEqual(policy.get_value(state, 0), None)
                    self.assertEqual(policy.get_action(state, 1), None)
                    self.assertEqual(policy[state, 1], None)
                    self.assertEqual(policy.get_value(state, 1), None)


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


class ReadWriteTimedTest(unittest.TestCase):

    def test_function(self):
        loc = StateFactor("loc", ["v1", "v2", "v3"])
        door = BoolStateFactor("door")
        battery = IntStateFactor("battery", 0, 10)

        state_action_dict_1 = {
            State({loc: "v1", door: False, battery: 5}): "a1",
            State({loc: "v2", door: True, battery: 9}): "a2",
            State({loc: "v3", door: False, battery: 1}): None,
        }
        state_action_dict_2 = {
            State({loc: "v1", door: False, battery: 5}): "a2",
            State({loc: "v2", door: True, battery: 9}): "a3",
            State({loc: "v3", door: False, battery: 1}): "a4",
        }

        value_dict_1 = {
            State({loc: "v1", door: False, battery: 5}): 1.0,
            State({loc: "v3", door: False, battery: 1}): 5.0,
        }

        value_dict_2 = {
            State({loc: "v1", door: False, battery: 5}): 4.0,
            State({loc: "v3", door: False, battery: 1}): 6.0,
        }

        policy = TimeDependentPolicy(
            [state_action_dict_1, state_action_dict_2], [value_dict_1, value_dict_2]
        )

        tmp = tempfile.NamedTemporaryFile()
        policy.write_policy(tmp.name)
        read_policy = TimeDependentPolicy({}, policy_file=tmp.name)

        self.assertEqual(len(read_policy._state_action_dicts), 2)
        self.assertEqual(len(read_policy._value_dicts), 2)
        sf_dict = list(read_policy._state_action_dicts[0].keys())[0]._sf_dict
        self.assertEqual(len(sf_dict), 3)
        self.assertEqual(sf_dict["loc"], loc)
        self.assertEqual(sf_dict["door"], door)
        self.assertEqual(sf_dict["battery"], battery)

        self.assertEqual(
            set(policy._state_action_dicts[0].keys()),
            set(read_policy._state_action_dicts[0].keys()),
        )

        self.assertEqual(
            set(policy._state_action_dicts[1].keys()),
            set(read_policy._state_action_dicts[1].keys()),
        )

        self.assertEqual(
            set(policy._value_dicts[0].keys()),
            set(read_policy._value_dicts[0].keys()),
        )

        self.assertEqual(
            set(policy._value_dicts[1].keys()),
            set(read_policy._value_dicts[1].keys()),
        )

        for state in policy._state_action_dicts[0]:
            self.assertEqual(policy[state, 0], read_policy[state, 0])

        for state in policy._state_action_dicts[1]:
            self.assertEqual(policy[state, 1], read_policy[state, 1])

        for state in policy._value_dicts[0]:
            self.assertEqual(
                policy.get_value(state, 0), read_policy.get_value(state, 0)
            )

        for state in policy._value_dicts[1]:
            self.assertEqual(
                policy.get_value(state, 1), read_policy.get_value(state, 1)
            )

        # Now test without the value function
        policy = TimeDependentPolicy([state_action_dict_1, state_action_dict_2])

        tmp = tempfile.NamedTemporaryFile()
        policy.write_policy(tmp.name)
        read_policy = TimeDependentPolicy({}, policy_file=tmp.name)

        self.assertEqual(
            set(policy._state_action_dicts[0].keys()),
            set(read_policy._state_action_dicts[0].keys()),
        )

        self.assertEqual(
            set(policy._state_action_dicts[1].keys()),
            set(read_policy._state_action_dicts[1].keys()),
        )

        self.assertEqual(read_policy._value_dicts, None)

        for state in policy._state_action_dicts[0]:
            self.assertEqual(policy[state, 0], read_policy[state, 0])

        for state in policy._state_action_dicts[1]:
            self.assertEqual(policy[state, 1], read_policy[state, 1])


class HierarchicalRepTest(unittest.TestCase):

    def test_function(self):
        sf1 = StateFactor("sf1", ["x1", "x2", "x3"])
        sf2 = StateFactor("sf2", ["y1", "y2", "y3"])
        sf3 = StateFactor("sf3", ["z1", "z2", "z3"])

        state_act_dict = {
            State({sf1: "x1", sf2: "y1", sf3: "z1"}): "a1",
            State({sf1: "x1", sf2: "y1", sf3: "z2"}): None,
            State({sf1: "x1", sf2: "y2", sf3: "z2"}): "a2",
            State({sf1: "x1", sf2: "y1", sf3: "z3"}): "a3",
            State({sf1: "x2", sf2: "y1", sf3: "z3"}): "a4",
        }

        policy = Policy(state_act_dict)

        hier_policy = policy._hierarchical_rep()

        expected = {
            "sf1==0": {
                "sf2==0": {"sf3==0": "a1", "sf3==2": "a3"},
                "sf2==1": {"sf3==1": "a2"},
            },
            "sf1==1": {"sf2==0": {"sf3==2": "a4"}},
        }

        self.assertEqual(hier_policy, expected)


class ToSCXMLTest(unittest.TestCase):
    def test_function(self):
        sf1 = StateFactor("sf1", ["x1", "x2", "x3"])
        sf2 = StateFactor("sf2", ["y1", "y2", "y3"])
        sf3 = StateFactor("sf3", ["z1", "z2", "z3"])

        state_act_dict = {
            State({sf1: "x1", sf2: "y1", sf3: "z1"}): "a1",
            State({sf1: "x1", sf2: "y1", sf3: "z2"}): None,
            State({sf1: "x1", sf2: "y2", sf3: "z2"}): "a2",
            State({sf1: "x1", sf2: "y1", sf3: "z3"}): "a3",
            State({sf1: "x2", sf2: "y1", sf3: "z3"}): "a4",
        }

        policy = Policy(state_act_dict)
        initial_state = State({sf1: "x1", sf2: "y2", sf3: "z3"})
        policy.to_scxml("test_policy.scxml", "mdp", initial_state, "my_policy")

        with open("test_policy.scxml", "r") as in_file:
            read_scxml = in_file.read()
            self.assertEqual(read_scxml, generate_scxml_test_string())

        os.remove("test_policy.scxml")


if __name__ == "__main__":
    unittest.main()
