#!/usr/bin/env python3
""" Unit tests for refine.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, IntStateFactor
from refine_plan.algorithms.refine import synthesise_bt_from_options
from refine_plan.models.condition import EqCondition, Label
from refine_plan.models.option import Option
from refine_plan.models.state import State
import unittest
import os


def build_toy_semi_mdp_components():
    sf = IntStateFactor("t", 0, 2)

    state_0 = State({sf: 0})
    state_1 = State({sf: 1})
    state_2 = State({sf: 2})

    a1 = Option(
        "a1",
        [
            (
                state_0.to_and_cond(),
                {state_1.to_and_cond(): 1.0},
            ),
        ],
        [],
    )

    a2 = Option(
        "a2",
        [
            (
                state_0.to_and_cond(),
                {state_2.to_and_cond(): 1.0},
            ),
        ],
        [],
    )

    loop = Option(
        "loop",
        [
            (
                state_1.to_and_cond(),
                {state_1.to_and_cond(): 1.0},
            ),
            (
                state_2.to_and_cond(),
                {state_2.to_and_cond(): 1.0},
            ),
        ],
        [(state_2.to_and_cond(), 1)],
    )

    goal = Label("goal", EqCondition(sf, 2))

    return [sf], [a1, a2, loop], [goal]


def build_graph_semi_mdp_components():
    sf = StateFactor("loc", ["v1", "v2", "v3", "v4", "v5"])

    v1 = State({sf: "v1"})
    v2 = State({sf: "v2"})
    v3 = State({sf: "v3"})
    v4 = State({sf: "v4"})
    v5 = State({sf: "v5"})

    e12 = Option(
        "e12",
        [
            (
                v1.to_and_cond(),
                {v2.to_and_cond(): 1.0},
            ),
        ],
        [(v1.to_and_cond(), 3)],
    )

    e13 = Option(
        "e13",
        [
            (
                v1.to_and_cond(),
                {v3.to_and_cond(): 1.0},
            ),
        ],
        [(v1.to_and_cond(), 2)],
    )

    e14 = Option(
        "e14",
        [
            (
                v1.to_and_cond(),
                {v4.to_and_cond(): 1.0},
            ),
        ],
        [(v1.to_and_cond(), 1)],
    )

    e21 = Option(
        "e21",
        [
            (
                v2.to_and_cond(),
                {v1.to_and_cond(): 1.0},
            ),
        ],
        [(v2.to_and_cond(), 3)],
    )

    e25 = Option(
        "e25",
        [
            (
                v2.to_and_cond(),
                {v5.to_and_cond(): 1.0},
            ),
        ],
        [(v2.to_and_cond(), 2)],
    )

    e31 = Option(
        "e31",
        [
            (
                v3.to_and_cond(),
                {v1.to_and_cond(): 1.0},
            ),
        ],
        [(v3.to_and_cond(), 2)],
    )

    e35 = Option(
        "e35",
        [
            (
                v3.to_and_cond(),
                {v5.to_and_cond(): 1.0},
            ),
        ],
        [(v3.to_and_cond(), 4)],
    )

    e41 = Option(
        "e41",
        [
            (
                v4.to_and_cond(),
                {v1.to_and_cond(): 1.0},
            ),
        ],
        [(v4.to_and_cond(), 1)],
    )

    e45 = Option(
        "e45",
        [
            (
                v4.to_and_cond(),
                {v5.to_and_cond(): 1.0},
            ),
        ],
        [(v4.to_and_cond(), 6)],
    )

    e52 = Option(
        "e52",
        [
            (
                v5.to_and_cond(),
                {v2.to_and_cond(): 1.0},
            ),
        ],
        [(v5.to_and_cond(), 2)],
    )

    e53 = Option(
        "e53",
        [
            (
                v5.to_and_cond(),
                {v3.to_and_cond(): 1.0},
            ),
        ],
        [(v5.to_and_cond(), 4)],
    )

    e54 = Option(
        "e54",
        [
            (
                v5.to_and_cond(),
                {v4.to_and_cond(): 1.0},
            ),
        ],
        [(v5.to_and_cond(), 6)],
    )

    goal = Label("goal", EqCondition(sf, "v5"))

    return [sf], [e12, e13, e14, e21, e25, e31, e35, e41, e45, e52, e53, e54], [goal]


class SynthesiseBTFromOptionsTest(unittest.TestCase):
    """The tests in this class are almost identical to those in semi_mdp_solver_tests.py."""

    def test_toy_semi_mdp(self):
        # The tests in this class also test _extract_policy
        sf_list, option_list, labels = build_toy_semi_mdp_components()
        prop = 'Pmax=?[ F "goal" ]'

        bt = synthesise_bt_from_options(
            sf_list,
            option_list,
            labels,
            prism_prop=prop,
            none_replacer="dummy",
            out_file="/tmp/toy_bt.xml",
        )

        self.assertEqual(bt.tick_at_state(State({sf_list[0]: 0})), "a2")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: 1})), "loop")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: 2})), "dummy")

        self.assertTrue(os.path.exists("/tmp/toy_bt.xml"))

    def test_graph_semi_mdp(self):
        sf_list, option_list, labels = build_graph_semi_mdp_components()
        prop = 'Rmin=?[ F "goal" ]'

        bt = synthesise_bt_from_options(
            sf_list,
            option_list,
            labels,
            prism_prop=prop,
            none_replacer="dummy",
            out_file="/tmp/graph_bt.xml",
        )

        self.assertEqual(bt.tick_at_state(State({sf_list[0]: "v1"})), "e12")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: "v2"})), "e25")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: "v3"})), "e35")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: "v4"})), "e41")
        self.assertEqual(bt.tick_at_state(State({sf_list[0]: "v5"})), "dummy")

        self.assertTrue(os.path.exists("/tmp/graph_bt.xml"))


if __name__ == "__main__":
    unittest.main()
