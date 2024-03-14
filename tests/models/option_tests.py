#!/usr/bin/env python3
""" Unit tests for option.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor
from refine_plan.models.condition import EqCondition
from refine_plan.models.option import Option
from refine_plan.models.state import State
import unittest


class ConstructorTest(unittest.TestCase):

    def test_function(self):
        # This also tests check valid probs and get_name
        sf = StateFactor("sf", ["a", "b", "c"])

        pre_cond = EqCondition(sf, "a")
        prob_post_conds = {EqCondition(sf, "b"): 0.5, EqCondition(sf, "c"): 0.4}

        with self.assertRaises(Exception):
            Option("opt", [(pre_cond, prob_post_conds)], [])

        prob_post_conds = {EqCondition(sf, "b"): 0.5, EqCondition(sf, "c"): 0.5}
        opt = Option("opt", [(pre_cond, prob_post_conds)], [(pre_cond, 5)])

        self.assertEqual(opt._name, "opt")
        self.assertEqual(opt._transition_list, [(pre_cond, prob_post_conds)])
        self.assertEqual(opt._reward_list, [(pre_cond, 5)])

        self.assertEqual(opt.get_name(), "opt")


class TestTransAndRewards(unittest.TestCase):

    def test_function(self):
        # This also tests check valid probs and get_name
        sf = StateFactor("sf", ["a", "b", "c"])

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        trans = [
            (a_cond, {b_cond: 0.6, c_cond: 0.4}),
            (b_cond, {a_cond: 0.3, c_cond: 0.7}),
        ]

        rewards = [(a_cond, 6), (a_cond, 4), (b_cond, 5)]

        opt = Option("opt", trans, rewards)

        self.assertEqual(opt.get_name(), "opt")

        a_state = State({sf: "a"})
        b_state = State({sf: "b"})
        c_state = State({sf: "c"})

        self.assertEqual(opt.get_transition_prob(a_state, a_state), 0.0)
        self.assertEqual(opt.get_transition_prob(a_state, b_state), 0.6)
        self.assertEqual(opt.get_transition_prob(a_state, c_state), 0.4)
        self.assertEqual(opt.get_transition_prob(b_state, a_state), 0.3)
        self.assertEqual(opt.get_transition_prob(b_state, b_state), 0.0)
        self.assertEqual(opt.get_transition_prob(b_state, c_state), 0.7)
        self.assertEqual(opt.get_transition_prob(c_state, a_state), 0.0)
        self.assertEqual(opt.get_transition_prob(c_state, b_state), 0.0)
        self.assertEqual(opt.get_transition_prob(c_state, c_state), 0.0)

        self.assertEqual(opt.get_reward(a_state), 10)
        self.assertEqual(opt.get_reward(b_state), 5)
        self.assertEqual(opt.get_reward(c_state), 0)

        trans_str = (
            "[opt] (sf = 0) -> 0.6:(sf' = 1) + 0.4:(sf' = 2);\n"
            + "[opt] (sf = 1) -> 0.3:(sf' = 0) + 0.7:(sf' = 2);\n"
        )
        self.assertEqual(opt.get_transition_prism_string(), trans_str)

        reward_str = "[opt] (sf = 0): 6;\n[opt] (sf = 0): 4;\n[opt] (sf = 1): 5;\n"
        self.assertEqual(opt.get_reward_prism_string(), reward_str)


if __name__ == "__main__":
    unittest.main()
