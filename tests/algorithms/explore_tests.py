#!/usr/bin/env python3
"""Unit tests for explore.py.

Note that synthesise_exploration_policy is not unit tested here,
and is instead integration tested in bin/.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import OrCondition, AndCondition, EqCondition
from refine_plan.models.state_factor import StateFactor, IntStateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.semi_mdp import SemiMDP
from refine_plan.models.state import State
import refine_plan.algorithms.explore
import numpy as np
import unittest
import queue
import yaml


class BuildStateIdxMapTest(unittest.TestCase):

    def test_function(self):
        sf_1 = StateFactor("sf1", ["a", "b", "c"])
        sf_2 = StateFactor("sf2", [1, 2, 3])

        state_idx_map = refine_plan.algorithms.explore._build_state_idx_map(
            [sf_1, sf_2]
        )

        expected = {
            State({sf_1: "a", sf_2: 1}): 0,
            State({sf_1: "a", sf_2: 2}): 1,
            State({sf_1: "a", sf_2: 3}): 2,
            State({sf_1: "b", sf_2: 1}): 3,
            State({sf_1: "b", sf_2: 2}): 4,
            State({sf_1: "b", sf_2: 3}): 5,
            State({sf_1: "c", sf_2: 1}): 6,
            State({sf_1: "c", sf_2: 2}): 7,
            State({sf_1: "c", sf_2: 3}): 8,
        }

        self.assertEqual(state_idx_map, expected)


class CreateEnsembleForOptionTest(unittest.TestCase):
    def test_function(self):
        bookstore_data = "../../data/bookstore/dataset.yaml"

        with open(bookstore_data, "r") as yaml_in:
            data = yaml.load(yaml_in, Loader=yaml.FullLoader)["check_door"]

        loc_sf = StateFactor("location", ["v{}".format(i) for i in range(1, 9)])
        door_sfs = [
            StateFactor("v2_door", ["unknown", "closed", "open"]),
            StateFactor("v3_door", ["unknown", "closed", "open"]),
            StateFactor("v4_door", ["unknown", "closed", "open"]),
            StateFactor("v5_door", ["unknown", "closed", "open"]),
            StateFactor("v6_door", ["unknown", "closed", "open"]),
            StateFactor("v7_door", ["unknown", "closed", "open"]),
        ]
        sf_list = [loc_sf] + door_sfs
        door_locs = ["v{}".format(i) for i in range(2, 8)]
        enabled_cond = OrCondition()
        for i in range(len(door_locs)):
            enabled_cond.add_cond(
                AndCondition(
                    EqCondition(loc_sf, door_locs[i]),
                    EqCondition(door_sfs[i], "unknown"),
                )
            )

        state_idx_map = refine_plan.algorithms.explore._build_state_idx_map(sf_list)
        q = queue.Queue()
        refine_plan.algorithms.explore._create_ensemble_for_option(
            "check_door", data, 3, 100, sf_list, enabled_cond, state_idx_map, q
        )
        ensemble = q.get()

        self.assertEqual(ensemble._ensemble_size, 3)
        self.assertEqual(ensemble._horizon, 100)
        self.assertEqual(ensemble._sf_list, sf_list)
        self.assertEqual(ensemble._enabled_cond, enabled_cond)

        self.assertEqual(len(ensemble._dbns), 3)
        for i in range(3):
            self.assertTrue(isinstance(ensemble._dbns[i], DBNOption))

        self.assertEqual(len(ensemble._transition_dicts), 3)
        for i in range(3):
            self.assertEqual(
                len(ensemble._sampled_transition_dict),
                len(ensemble._transition_dicts[i]),
            )
            for state in ensemble._transition_dicts[i]:
                self.assertTrue(enabled_cond.is_satisfied(state))

        self.assertEqual(
            len(ensemble._reward_dict), len(ensemble._sampled_transition_dict)
        )
        for state in ensemble._reward_dict:
            self.assertTrue(enabled_cond.is_satisfied(state))

        self.assertTrue(ensemble._transition_prism_str is not None)
        self.assertTrue(ensemble._reward_prism_str is not None)
        self.assertTrue(ensemble._sampled_transition_mat is not None)
        self.assertTrue(ensemble._reward_mat is not None)


class BuildOptionsTest(unittest.TestCase):

    def test_function(self):
        holder = refine_plan.algorithms.explore._create_ensemble_for_option

        def just_name(opt, data, en_size, horizon, sf_list, e_cond, s_map, queue):
            queue.put(opt)

        refine_plan.algorithms.explore._create_ensemble_for_option = just_name

        option_names = ["opt_1", "opt_2", "opt_3"]
        dataset = {"opt_1": None, "opt_2": None, "opt_3": None}
        enabled_conds = {"opt_1": None, "opt_2": None, "opt_3": None}
        option_list = refine_plan.algorithms.explore._build_options(
            option_names, dataset, 3, 100, [], enabled_conds, {}
        )

        self.assertEqual(len(option_list), 3)
        self.assertTrue("opt_1" in option_list)
        self.assertTrue("opt_2" in option_list)
        self.assertTrue("opt_3" in option_list)

        refine_plan.algorithms.explore._create_ensemble_for_option = holder


class DummyOption(object):

    def __init__(self, name, trans_mat, reward_mat):
        self._sampled_transition_mat = trans_mat
        self._reward_mat = reward_mat
        self._name = name

    def get_name(self):
        return self._name


class SolveFiniteHorizonMDPTest(unittest.TestCase):

    def test_function(self):

        trans_mat_0 = np.zeros((3, 3))
        trans_mat_0[0, 1] = 0.6
        trans_mat_0[0, 2] = 0.4
        reward_mat_0 = np.zeros(3)
        reward_mat_0[0] = 5
        opt_0 = DummyOption("opt_0", trans_mat_0, reward_mat_0)

        trans_mat_1 = np.zeros((3, 3))
        trans_mat_1[0, 1] = 0.7
        trans_mat_1[0, 2] = 0.3
        reward_mat_1 = np.zeros(3)
        reward_mat_1[0] = 4
        opt_1 = DummyOption("opt_1", trans_mat_1, reward_mat_1)

        trans_mat_2 = np.zeros((3, 3))
        trans_mat_2[1, 0] = 1.0
        reward_mat_2 = np.zeros(3)
        reward_mat_2[1] = 3
        opt_2 = DummyOption("opt_2", trans_mat_2, reward_mat_2)

        trans_mat_3 = np.zeros((3, 3))
        trans_mat_3[2, 0] = 1.0
        reward_mat_3 = np.zeros(3)
        reward_mat_3[2] = 7
        opt_3 = DummyOption("opt_3", trans_mat_3, reward_mat_3)

        sf = IntStateFactor("s", 0, 2)
        mdp = SemiMDP([sf], [opt_0, opt_1, opt_2, opt_3], [], None)

        state_idx_map = {State({sf: i}): i for i in range(3)}
        horizon = 2

        policy = refine_plan.algorithms.explore.solve_finite_horizon_mdp(
            mdp, state_idx_map, horizon
        )

        self.assertEqual(policy[State({sf: 0}), 1], "opt_0")
        self.assertEqual(policy[State({sf: 1}), 1], "opt_2")
        self.assertEqual(policy[State({sf: 2}), 1], "opt_3")
        self.assertEqual(policy[State({sf: 0}), 0], "opt_0")
        self.assertEqual(policy[State({sf: 1}), 0], "opt_2")
        self.assertEqual(policy[State({sf: 2}), 0], "opt_3")
        self.assertAlmostEqual(policy.get_value(State({sf: 0}), 1), 5.0, 6)
        self.assertAlmostEqual(policy.get_value(State({sf: 1}), 1), 3, 6)
        self.assertAlmostEqual(policy.get_value(State({sf: 2}), 1), 7, 6)
        self.assertAlmostEqual(policy.get_value(State({sf: 0}), 0), 9.6, 6)
        self.assertAlmostEqual(policy.get_value(State({sf: 1}), 0), 8, 6)
        self.assertAlmostEqual(policy.get_value(State({sf: 2}), 0), 12, 6)


if __name__ == "__main__":
    unittest.main()
