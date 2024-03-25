#!/usr/bin/env python3
""" Unit tests for semi_mdp_solver.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor, IntStateFactor
from refine_plan.models.condition import EqCondition, Label
from refine_plan.algorithms.semi_mdp_solver import (
    _build_prism_program,
    _build_storm_model,
    _check_result,
    synthesise_policy,
)
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.option import Option
from refine_plan.models.state import State
import unittest
import stormpy


# Dummy classes for _check_result test
class DummyScheduler(object):

    def __init__(self, memoryless, deterministic):
        self.memoryless = memoryless
        self.deterministic = deterministic


class DummyResult(object):

    def __init__(self, all_states, has_scheduler, memoryless, deterministic):
        self.result_for_all_states = all_states
        self.has_scheduler = has_scheduler
        self.scheduler = DummyScheduler(memoryless, deterministic)


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


def build_toy_semi_mdp():
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

    semi_mdp = SemiMDP(
        [sf],
        [a1, a2, loop],
        [goal],
    )

    return semi_mdp


def build_graph_semi_mdp():
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

    semi_mdp = SemiMDP(
        [sf],
        [e12, e13, e14, e21, e25, e31, e35, e41, e45, e52, e53, e54],
        [goal],
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


class BuildStormModelTest(unittest.TestCase):

    def test_function(self):

        semi_mdp = build_test_semi_mdp()
        prog = _build_prism_program(semi_mdp)
        formula = stormpy.parse_properties('Rmax=?[F "label_1"]', prog)[0]

        model = _build_storm_model(prog, formula)
        self.assertEqual(model.nr_states, 6)
        self.assertEqual(model.nr_transitions, 12)  # This includes added self loops
        self.assertEqual(len(model.labeling.get_labels()), 4)
        self.assertTrue("label_1" in model.labeling.get_labels())
        self.assertTrue("label_2" in model.labeling.get_labels())
        self.assertTrue("deadlock" in model.labeling.get_labels())
        self.assertTrue("init" in model.labeling.get_labels())
        self.assertEqual(model.model_type.name, "MDP")


class CheckResultTest(unittest.TestCase):

    def test_function(self):

        for all_states in [False, True]:
            for has_scheduler in [False, True]:
                for memoryless in [False, True]:
                    for deterministic in [False, True]:
                        result = DummyResult(
                            all_states, has_scheduler, memoryless, deterministic
                        )

                        if (
                            not all_states
                            or not has_scheduler
                            or not memoryless
                            or not deterministic
                        ):
                            with self.assertRaises(Exception):
                                _check_result(result)
                        else:
                            _check_result(result)


class SynthesisePolicyTest(unittest.TestCase):

    def test_toy_semi_mdp(self):
        # The tests in this class also test _extract_policy
        semi_mdp = build_toy_semi_mdp()
        prop = 'Pmax=?[ F "goal" ]'

        policy = synthesise_policy(semi_mdp, prop)
        self.assertEqual(len(policy._state_action_dict), 3)
        self.assertEqual(len(policy._value_dict), 3)

        sfs = semi_mdp.get_state_factors()
        self.assertEqual(policy[State({sfs["t"]: 0})], "a2")
        self.assertEqual(policy[State({sfs["t"]: 1})], "loop")
        self.assertEqual(policy[State({sfs["t"]: 2})], None)

        self.assertEqual(policy.get_value(State({sfs["t"]: 0})), 1.0)
        self.assertEqual(policy.get_value(State({sfs["t"]: 1})), 0.0)
        self.assertEqual(policy.get_value(State({sfs["t"]: 2})), 1.0)

    def test_graph_semi_mdp(self):
        semi_mdp = build_graph_semi_mdp()
        prop = 'Rmin=?[ F "goal" ]'

        policy = synthesise_policy(semi_mdp, prop)
        self.assertEqual(len(policy._state_action_dict), 5)
        self.assertEqual(len(policy._value_dict), 5)

        sfs = semi_mdp.get_state_factors()
        self.assertEqual(policy[State({sfs["loc"]: "v1"})], "e12")
        self.assertEqual(policy[State({sfs["loc"]: "v2"})], "e25")
        self.assertEqual(policy[State({sfs["loc"]: "v3"})], "e35")
        self.assertEqual(policy[State({sfs["loc"]: "v4"})], "e41")
        self.assertEqual(policy[State({sfs["loc"]: "v5"})], None)

        self.assertEqual(policy.get_value(State({sfs["loc"]: "v1"})), 5.0)
        self.assertEqual(policy.get_value(State({sfs["loc"]: "v2"})), 2.0)
        self.assertEqual(policy.get_value(State({sfs["loc"]: "v3"})), 4.0)
        self.assertEqual(policy.get_value(State({sfs["loc"]: "v4"})), 6.0)
        self.assertEqual(policy.get_value(State({sfs["loc"]: "v5"})), 0.0)


if __name__ == "__main__":
    unittest.main()
