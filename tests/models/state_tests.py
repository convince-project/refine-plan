#!/usr/bin/env python3
""" Unit tests for state.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import AndCondition, EqCondition
from refine_plan.models.state_factor import StateFactor, IntStateFactor
from refine_plan.models.state import State
import unittest


class StateTest(unittest.TestCase):

    def test_function(self):
        loc = StateFactor("loc", ["v1", "v2", "v3"])
        time = IntStateFactor("time", 0, 100)

        with self.assertRaises(Exception):
            State({loc: "v4", time: 17})

        state = State({loc: "v2", time: 5})
        self.assertEqual(state._state_dict, {"loc": "v2", "time": 5})
        self.assertEqual(state._sf_dict, {"loc": loc, "time": time})
        self.assertEqual(state._hash_val, None)
        self.assertEqual(repr(state), "State(loc: v2, time: 5)")
        self.assertEqual(str(state), "State(loc: v2, time: 5)")

        with self.assertRaises(Exception):
            state["battery"]

        self.assertEqual(state["loc"], "v2")
        self.assertEqual(state["time"], 5)

        self.assertTrue("loc" in state)
        self.assertTrue("time" in state)
        self.assertFalse("battery" in state)

        self.assertEqual(state, state)
        self.assertEqual(state, State({loc: "v2", time: 5}))
        self.assertNotEqual(state, State({loc: "v1", time: 5}))
        battery = StateFactor("battery", ["low", "med", "high"])
        self.assertNotEqual(state, State({loc: "v2", time: 5, battery: "med"}))

        self.assertEqual(state._hash_val, None)
        self.assertEqual(hash(state), hash(State({loc: "v2", time: 5})))
        self.assertEqual(state._hash_val, hash(State({loc: "v2", time: 5})))
        self.assertNotEqual(hash(state), hash(State({loc: "v2", time: 6})))
        self.assertNotEqual(state._hash_val, hash(State({loc: "v2", time: 6})))
        self.assertEqual(hash(state), hash(State({loc: "v2", time: 5})))
        self.assertEqual(state._hash_val, hash(State({loc: "v2", time: 5})))
        self.assertEqual(state._hash_val, hash(state))

        and_cond = state.to_and_cond()
        self.assertTrue(isinstance(and_cond, AndCondition))
        self.assertEqual(len(and_cond._cond_list), 2)
        if and_cond._cond_list[0]._sf == loc:
            self.assertEqual(and_cond._cond_list[0]._sf, loc)
            self.assertEqual(and_cond._cond_list[0]._value, "v2")
            self.assertEqual(and_cond._cond_list[1]._sf, time)
            self.assertEqual(and_cond._cond_list[1]._value, 5)
        elif and_cond._cond_list[0]._sf == time:
            self.assertEqual(and_cond._cond_list[0]._sf, time)
            self.assertEqual(and_cond._cond_list[0]._value, 5)
            self.assertEqual(and_cond._cond_list[1]._sf, loc)
            self.assertEqual(and_cond._cond_list[1]._value, "v2")
        else:
            self.fail()


if __name__ == "__main__":
    unittest.main()
