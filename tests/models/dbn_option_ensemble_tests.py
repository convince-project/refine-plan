#!/usr/bin/env python3
"""Unit tests for dbn_option_ensemble.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.dbn_option_ensemble import DBNOptionEnsemble
from refine_plan.models.condition import EqCondition, AddCondition
from refine_plan.models.state_factor import IntStateFactor
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


if __name__ == "__main__":
    unittest.main()
