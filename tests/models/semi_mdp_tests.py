#!/usr/bin/env python
""" Unit tests for semi_mdp.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor
from refine_plan.models.condition import EqCondition, Label
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.option import Option
from refine_plan.models.state import State
from datetime import datetime
import numpy as np
import unittest


def generate_prism_test_string(add_initial_state=False):
    now = datetime.now()
    expected = "// Auto-generated semi-MDP for REFINE-PLAN\n"
    expected += "// Date generated: {}/{}/{}\n\n".format(now.day, now.month, now.year)
    if add_initial_state:
        expected += (
            "mdp\n\nmodule semimdp\n\nsf: [0..2] init 1;\nbool_sf: [0..1] init 1;\n\n"
        )
    else:
        expected += "mdp\n\nmodule semimdp\n\nsf: [0..2];\nbool_sf: [0..1];\n\n"
    expected += (
        "[opt_1] ((sf = 0) & (bool_sf = 0)) -> "
        + "0.6:(sf' = 1) & (bool_sf' = 0) + 0.4:(sf' = 2) & (bool_sf' = 0);\n"
    )
    expected += (
        "[opt_1] ((sf = 1) & (bool_sf = 0)) -> "
        + "0.3:(sf' = 0) & (bool_sf' = 0) + 0.7:(sf' = 2) & (bool_sf' = 0);\n"
    )
    expected += (
        "[opt_1] ((sf = 2) & (bool_sf = 0)) -> "
        + "0.5:(sf' = 0) & (bool_sf' = 0) + 0.5:(sf' = 1) & (bool_sf' = 0);\n"
    )
    expected += (
        "[opt_2] ((sf = 0) & (bool_sf = 0)) -> 1.0:(sf' = 0) & (bool_sf' = 1);\n"
    )
    expected += (
        "[opt_2] ((sf = 1) & (bool_sf = 0)) -> 1.0:(sf' = 1) & (bool_sf' = 1);\n"
    )
    expected += (
        "[opt_2] ((sf = 2) & (bool_sf = 0)) -> 1.0:(sf' = 2) & (bool_sf' = 1);\n"
    )
    expected += "\nendmodule\n\n"
    expected += 'label "label_1" = (bool_sf = 1);\n'
    expected += 'label "label_2" = (sf = 2);\n'
    expected += "\nrewards\n"
    expected += "[opt_1] ((sf = 0) & (bool_sf = 0)): 1;\n"
    expected += "[opt_1] ((sf = 1) & (bool_sf = 0)): 3;\n"
    expected += "[opt_1] ((sf = 2) & (bool_sf = 0)): 5;\n"
    expected += "[opt_2] ((sf = 0) & (bool_sf = 0)): 10;\n"
    expected += "[opt_2] ((sf = 1) & (bool_sf = 0)): 15;\n"
    expected += "[opt_2] ((sf = 2) & (bool_sf = 0)): 20;\n"
    expected += "endrewards\n"

    if not add_initial_state:
        expected += "\ninit true endinit\n"

    return expected


class ConstructorTest(unittest.TestCase):

    def test_function(self):

        sf_1 = StateFactor("sf", ["a", "b", "c"])
        sf_2 = BoolStateFactor("bool_sf")

        opt_1 = Option("opt_1", [], [])
        opt_2 = Option("opt_2", [], [])

        label_1 = Label("label_1", EqCondition(sf_2, True))
        label_2 = Label("label_2", EqCondition(sf_1, "c"))

        initial_state = State({sf_1: "b", sf_2: False})

        semi_mdp = SemiMDP(
            [sf_1, sf_2], [opt_1, opt_2], [label_1, label_2], initial_state
        )

        self.assertEqual(semi_mdp._state_factors, {"sf": sf_1, "bool_sf": sf_2})
        self.assertEqual(semi_mdp._options, {"opt_1": opt_1, "opt_2": opt_2})
        self.assertEqual(semi_mdp._labels, [label_1, label_2])
        self.assertEqual(semi_mdp._initial_state, initial_state)

        self.assertEqual(semi_mdp.get_det_initial_state(), initial_state)
        self.assertEqual(semi_mdp.get_labels(), [label_1, label_2])
        self.assertEqual(semi_mdp.get_state_factors(), {"sf": sf_1, "bool_sf": sf_2})


class TransitionRewardTest(unittest.TestCase):

    def test_function(self):

        sf_1 = StateFactor("sf", ["a", "b", "c"])
        sf_2 = BoolStateFactor("bool_sf")

        state_1 = State({sf_1: "a", sf_2: False})
        state_2 = State({sf_1: "a", sf_2: True})
        state_3 = State({sf_1: "b", sf_2: False})
        state_4 = State({sf_1: "b", sf_2: True})
        state_5 = State({sf_1: "c", sf_2: False})
        state_6 = State({sf_1: "c", sf_2: True})

        opt_1 = Option(
            "opt_1",
            [
                (
                    state_1.to_and_cond(),
                    {state_3.to_and_cond(): 0.6, state_5.to_and_cond(): 0.4},
                ),
                (
                    state_3.to_and_cond(),
                    {state_1.to_and_cond(): 0.3, state_5.to_and_cond(): 0.7},
                ),
                (
                    state_5.to_and_cond(),
                    {state_1.to_and_cond(): 0.5, state_3.to_and_cond(): 0.5},
                ),
            ],
            [
                (state_1.to_and_cond(), 1),
                (state_3.to_and_cond(), 3),
                (state_5.to_and_cond(), 5),
            ],
        )

        opt_2 = Option(
            "opt_2",
            [
                (
                    state_1.to_and_cond(),
                    {state_2.to_and_cond(): 1.0},
                ),
                (
                    state_3.to_and_cond(),
                    {state_4.to_and_cond(): 1.0},
                ),
                (
                    state_5.to_and_cond(),
                    {state_6.to_and_cond(): 1.0},
                ),
            ],
            [
                (state_1.to_and_cond(), 10),
                (state_3.to_and_cond(), 15),
                (state_5.to_and_cond(), 20),
            ],
        )

        label_1 = Label("label_1", EqCondition(sf_2, True))
        label_2 = Label("label_2", EqCondition(sf_1, "c"))

        semi_mdp = SemiMDP(
            [sf_1, sf_2],
            [opt_1, opt_2],
            [label_1, label_2],
        )

        states = [state_1, state_2, state_3, state_4, state_5, state_6]
        opt_1_mat = np.zeros((6, 6))
        opt_1_mat[0, 2] = 0.6
        opt_1_mat[0, 4] = 0.4
        opt_1_mat[2, 0] = 0.3
        opt_1_mat[2, 4] = 0.7
        opt_1_mat[4, 0] = 0.5
        opt_1_mat[4, 2] = 0.5

        opt_2_mat = np.zeros((6, 6))
        opt_2_mat[0, 1] = 1.0
        opt_2_mat[2, 3] = 1.0
        opt_2_mat[4, 5] = 1.0

        opt_1_reward = [1, 0, 3, 0, 5, 0]
        opt_2_reward = [10, 0, 15, 0, 20, 0]

        for s in states:
            for next_s in states:
                self.assertEqual(
                    semi_mdp.get_transition_prob(s, "opt_1", next_s),
                    opt_1_mat[states.index(s), states.index(next_s)],
                )
                self.assertEqual(
                    semi_mdp.get_transition_prob(s, "opt_2", next_s),
                    opt_2_mat[states.index(s), states.index(next_s)],
                )

            self.assertEqual(
                semi_mdp.get_reward(s, "opt_1"), opt_1_reward[states.index(s)]
            )
            self.assertEqual(
                semi_mdp.get_reward(s, "opt_2"), opt_2_reward[states.index(s)]
            )


class PRISMStringTest(unittest.TestCase):

    def test_function(self):

        sf_1 = StateFactor("sf", ["a", "b", "c"])
        sf_2 = BoolStateFactor("bool_sf")

        state_1 = State({sf_1: "a", sf_2: False})
        state_2 = State({sf_1: "a", sf_2: True})
        state_3 = State({sf_1: "b", sf_2: False})
        state_4 = State({sf_1: "b", sf_2: True})
        state_5 = State({sf_1: "c", sf_2: False})
        state_6 = State({sf_1: "c", sf_2: True})

        opt_1 = Option(
            "opt_1",
            [
                (
                    state_1.to_and_cond(),
                    {state_3.to_and_cond(): 0.6, state_5.to_and_cond(): 0.4},
                ),
                (
                    state_3.to_and_cond(),
                    {state_1.to_and_cond(): 0.3, state_5.to_and_cond(): 0.7},
                ),
                (
                    state_5.to_and_cond(),
                    {state_1.to_and_cond(): 0.5, state_3.to_and_cond(): 0.5},
                ),
            ],
            [
                (state_1.to_and_cond(), 1),
                (state_3.to_and_cond(), 3),
                (state_5.to_and_cond(), 5),
            ],
        )

        opt_2 = Option(
            "opt_2",
            [
                (
                    state_1.to_and_cond(),
                    {state_2.to_and_cond(): 1.0},
                ),
                (
                    state_3.to_and_cond(),
                    {state_4.to_and_cond(): 1.0},
                ),
                (
                    state_5.to_and_cond(),
                    {state_6.to_and_cond(): 1.0},
                ),
            ],
            [
                (state_1.to_and_cond(), 10),
                (state_3.to_and_cond(), 15),
                (state_5.to_and_cond(), 20),
            ],
        )

        label_1 = Label("label_1", EqCondition(sf_2, True))
        label_2 = Label("label_2", EqCondition(sf_1, "c"))

        semi_mdp = SemiMDP(
            [sf_1, sf_2],
            [opt_1, opt_2],
            [label_1, label_2],
        )

        prism_str = semi_mdp.to_prism_string()
        expected = generate_prism_test_string()

        self.assertEqual(prism_str, expected)

        semi_mdp.to_prism_string("/tmp/semi_mdp_prism_test.nm")

        with open("/tmp/semi_mdp_prism_test.nm", "r") as in_file:
            read_prism_str = in_file.read()
            self.assertEqual(read_prism_str, expected)

        initial_state = State({sf_1: "b", sf_2: True})

        semi_mdp = SemiMDP(
            [sf_1, sf_2],
            [opt_1, opt_2],
            [label_1, label_2],
            initial_state=initial_state,
        )

        prism_str = semi_mdp.to_prism_string()
        expected = generate_prism_test_string(add_initial_state=True)
        self.assertEqual(prism_str, expected)


if __name__ == "__main__":
    unittest.main()
