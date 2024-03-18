#!/usr/bin/env python3
""" Unit tests for semi_mdp_solver.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor
from refine_plan.models.condition import EqCondition, Label
from refine_plan.solver.semi_mdp_solver import (
    _build_prism_program,
    _build_storm_model,
    _check_result,
    _extract_policy,
    synthesise_policy,
)
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.option import Option
from refine_plan.models.state import State
from datetime import datetime
import unittest


def build_test_semi_mdp():
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

    return semi_mdp


class BuildPrismProgramTest(unittest.TestCase):

    def test_function(self):
        semi_mdp = build_test_semi_mdp()
        prog = _build_prism_program(semi_mdp)

        self.assertEqual(len(prog.labels), 2)
        self.assertEqual(prog.labels[0].name, "label_1")
        self.assertEqual(prog.labels[1].name, "label_2")
        self.assertEqual(prog.nr_modules, 1)
        self.assertEqual(prog.model_type.name, "MDP")


if __name__ == "__main__":
    unittest.main()
