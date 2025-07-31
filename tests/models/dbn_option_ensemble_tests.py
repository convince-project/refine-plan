#!/usr/bin/env python3
"""Unit tests for dbn_option_ensemble.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor, BoolStateFactor, StateFactor
from refine_plan.models.dbn_option_ensemble import DBNOptionEnsemble
from refine_plan.models.condition import EqCondition, AddCondition
from refine_plan.models.state import State
import unittest


class ConstructorTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        self.assertEqual(ensemble._ensemble_size, 32)
        self.assertEqual(ensemble._horizon, 100)
        self.assertEqual(ensemble._sf_list, "sf_list")
        self.assertEqual(ensemble._enabled_cond, "enabled_cond")
        self.assertEqual(ensemble._dbns, [None] * 32)
        self.assertEqual(ensemble._transition_dicts, [None] * 32)
        self.assertEqual(ensemble._sampled_transition_dict, {})
        self.assertEqual(ensemble._reward_dict, {})
        self.assertEqual(ensemble._transition_prism_str, None)
        self.assertEqual(ensemble._reward_prism_str, None)


class GetTransitionProbTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        sf = IntStateFactor("sf", 0, 10)
        state = State({sf: 1})

        trans_dict = {state: {EqCondition(sf, 3): 0.6, AddCondition(sf, 5): 0.4}}

        ensemble._sampled_transition_dict = trans_dict

        next_state = State({sf: 3})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.6)
        next_state = State({sf: 6})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.4)
        next_state = State({sf: 1})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.0)


class GetRewardTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        sf = IntStateFactor("sf", 0, 10)
        state = State({sf: 1})
        ensemble._reward_dict[state] = 7

        self.assertEqual(ensemble.get_reward(state), 7)

        state = State({sf: 4})
        self.assertEqual(ensemble.get_reward(state), 0.0)


class GetSCXMLTransitionsTest(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        pass


class GetTransitionPRISMStringTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        with self.assertRaises(Exception):
            ensemble.get_transition_prism_string()

        ensemble._transition_prism_str = "test"

        self.assertEqual(ensemble.get_transition_prism_string(), "test")


class GetRewardPRISMStringTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        with self.assertRaises(Exception):
            ensemble.get_reward_prism_string()

        ensemble._reward_prism_str = "test"

        self.assertEqual(ensemble.get_reward_prism_string(), "test")


class ComputeEntropyTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 32, 100, "sf_list", "enabled_cond")

        dist = {"a": 0.7, "b": 0.2, "c": 0.1}

        entropy = ensemble._compute_entropy(dist)

        self.assertAlmostEqual(entropy, 1.15677964945)


class ComputeAvgDist(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 2, 100, "sf_list", "enabled_cond")

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        avg_dist = ensemble._compute_avg_dist("s")
        expected = {"n1": 0.35, "n2": 0.35, "n3": 0.3}
        self.assertEqual(avg_dist, expected)


class ComputeInfoGain(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 2, 100, "sf_list", "enabled_cond")

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        info_gain = ensemble._compute_info_gain("s")
        self.assertAlmostEqual(info_gain, 0.65517015239)


class CreateDatasetsTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        ensemble = DBNOptionEnsemble("option", [], 2, 100, sf_list, "enabled_cond")

        data = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, True, False],
                "yt": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, True, False], "r": [5, 5, 7]},
        }

        datasets = ensemble._create_datasets(data)

        self.assertEqual(len(datasets), 2)

        ds1 = {
            "transition": {
                "x0": [1, 3],
                "xt": [2, 1],
                "y0": [False, False],
                "yt": [False, True],
            },
            "reward": {"x": [1, 3], "y": [False, False], "r": [5, 7]},
        }

        ds2 = {
            "transition": {
                "x0": [2],
                "xt": [3],
                "y0": [True],
                "yt": [False],
            },
            "reward": {"x": [2], "y": [True], "r": [5]},
        }

        self.assertEqual(datasets[0], ds1)
        self.assertEqual(datasets[1], ds2)


class LearnDBNOptionsTest(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        pass


class BuildTransitionDictForDBNTest(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        pass


class BuildTransitionDictsTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        def set_dbn_idx(s, i, q):
            q.put((i, {i: i + 1}))

        DBNOptionEnsemble._build_transition_dict_for_dbn = set_dbn_idx

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        ensemble = DBNOptionEnsemble("option", [], 2, 100, sf_list, "enabled_cond")

        ensemble._build_transition_dicts()

        self.assertEqual(ensemble._transition_dicts, [{0: 1}, {1: 2}])


class ComputeSampledTransitionsAndInfoGainTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble("option", [], 2, 100, "sf_list", "enabled_cond")

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        ensemble._compute_sampled_transitions_and_info_gain()

        self.assertTrue(
            ensemble._sampled_transition_dict["s"] == {"n1": 0.7, "n2": 0.3}
            or ensemble._sampled_transition_dict["s"] == {"n2": 0.4, "n3": 0.6}
        )

        self.assertEqual(len(ensemble._reward_dict), 1)
        self.assertAlmostEqual(ensemble._reward_dict["s"], 0.65517015239)


class PrecomputePRISMStringsTest(unittest.TestCase):

    def test_function(self):
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None
        sf = StateFactor("sf", ["a", "b", "c"])

        ensemble = DBNOptionEnsemble("opt", [], 2, 100, [sf], "enabled_cond")

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        ensemble._sampled_transition_dict = {
            State({sf: "a"}): {b_cond: 0.6, c_cond: 0.4},
            State({sf: "b"}): {a_cond: 0.3, c_cond: 0.7},
        }

        ensemble._reward_dict = {
            State({sf: "a"}): 7.5,
            State({sf: "b"}): 1.3,
        }

        ensemble._precompute_prism_strings()

        trans_str = (
            "[opt] ((sf = 0) & (time < 100)) -> 0.6:(sf' = 1) & (time' = time + 1) + "
        )
        trans_str += "0.4:(sf' = 2) & (time' = time + 1);\n"
        trans_str += (
            "[opt] ((sf = 1) & (time < 100)) -> 0.3:(sf' = 0) & (time' = time + 1) + "
        )
        trans_str += "0.7:(sf' = 2) & (time' = time + 1);\n"
        self.assertEqual(ensemble._transition_prism_str, trans_str)

        reward_str = "[opt] ((sf = 0) & (time < 100)): 7.5;\n"
        reward_str += "[opt] ((sf = 1) & (time < 100)): 1.3;\n"

        self.assertEqual(ensemble._reward_prism_str, reward_str)


class SetupEnsembleTests(unittest.TestCase):

    def test_function(self):
        # TODO: Fill in
        pass


if __name__ == "__main__":
    unittest.main()
