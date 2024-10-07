#!/usr/bin/env python3
""" Unit tests for the simple policy to BT converter.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.simple_policy_to_bt import SimplePolicyBTConverter
from refine_plan.models.state_factor import StateFactor, BoolStateFactor
from refine_plan.models.policy import Policy
from refine_plan.models.state import State
import unittest
import os


class SimplePolicyBTConverterTest(unittest.TestCase):

    def test_conversion(self):

        loc_sf = StateFactor("location", ["v1", "v2", "v3"])
        v1_door_sf = BoolStateFactor("v1_door")
        v2_door_sf = BoolStateFactor("v2_door")

        states = [
            State({loc_sf: "v1", v1_door_sf: False, v2_door_sf: False}),
            State({loc_sf: "v1", v1_door_sf: False, v2_door_sf: True}),
            State({loc_sf: "v1", v1_door_sf: True, v2_door_sf: False}),
            State({loc_sf: "v1", v1_door_sf: True, v2_door_sf: True}),
            State({loc_sf: "v2", v1_door_sf: False, v2_door_sf: False}),
            State({loc_sf: "v2", v1_door_sf: False, v2_door_sf: True}),
            State({loc_sf: "v2", v1_door_sf: True, v2_door_sf: False}),
            State({loc_sf: "v2", v1_door_sf: True, v2_door_sf: True}),
            State({loc_sf: "v3", v1_door_sf: False, v2_door_sf: False}),
            State({loc_sf: "v3", v1_door_sf: False, v2_door_sf: True}),
            State({loc_sf: "v3", v1_door_sf: True, v2_door_sf: False}),
            State({loc_sf: "v3", v1_door_sf: True, v2_door_sf: True}),
        ]

        sa_dict = {
            states[0]: "a1",
            states[1]: "a2",
            states[2]: "a3",
            states[3]: "a4",
            states[4]: "a5",
            states[5]: "a6",
            states[6]: "a6",
            states[7]: "a5",
            states[8]: "a4",
            states[9]: "a3",
            states[10]: "a2",
            states[11]: "a1",
        }

        policy = Policy(sa_dict)

        converter = SimplePolicyBTConverter()
        bt = converter.convert_policy(policy, "/tmp/simple_bt.xml")

        self.assertTrue(os.path.exists("/tmp/simple_bt.xml"))
        os.unlink("/tmp/simple_bt.xml")

        for state in states:
            self.assertEqual(policy[state], bt.tick_at_state(state))


if __name__ == "__main__":
    unittest.main()
