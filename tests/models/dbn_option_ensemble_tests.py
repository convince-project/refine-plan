#!/usr/bin/env python3
"""Unit tests for dbn_option_ensemble.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor, BoolStateFactor, StateFactor
from refine_plan.models.dbn_option_ensemble import DBNOptionEnsemble
from refine_plan.algorithms.explore import _build_state_idx_map
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.state import State
from refine_plan.models.condition import (
    EqCondition,
    AddCondition,
    GeqCondition,
    TrueCondition,
    OrCondition,
    AndCondition,
)
from multiprocessing import SimpleQueue
import xml.etree.ElementTree as et
import pyAgrum as gum
import numpy as np
import unittest
import itertools
import yaml


class ConstructorTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", "map"
        )

        self.assertEqual(ensemble._ensemble_size, 32)
        self.assertEqual(ensemble._horizon, 100)
        self.assertEqual(ensemble._sf_list, "sf_list")
        self.assertEqual(ensemble._enabled_cond, "enabled_cond")
        self.assertEqual(ensemble._enabled_states, None)
        self.assertEqual(ensemble._dbns, [None] * 32)
        self.assertEqual(ensemble._transition_dicts, [None] * 32)
        self.assertEqual(ensemble._sampled_transition_dict, {})
        self.assertEqual(ensemble._reward_dict, {})
        self.assertEqual(ensemble._transition_prism_str, None)
        self.assertEqual(ensemble._reward_prism_str, None)
        self.assertEqual(ensemble._state_idx_map, "map")
        self.assertEqual(ensemble._sampled_transition_mat, None)
        self.assertEqual(ensemble._reward_mat, None)

        DBNOptionEnsemble._setup_ensemble = setup


class GetTransitionProbTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf = IntStateFactor("sf", 0, 10)
        state = State({sf: 1})
        state_2 = State({sf: 2})

        state_idx_map = {}
        for i in range(11):
            state_idx_map[State({sf: i})] = i

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", state_idx_map
        )

        trans_dict = {
            state: {EqCondition(sf, 3): 0.6, AddCondition(sf, 5): 0.4},
            state_2: None,
        }

        ensemble._sampled_transition_dict = trans_dict

        next_state = State({sf: 3})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.6)
        next_state = State({sf: 6})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.4)
        next_state = State({sf: 1})
        self.assertEqual(ensemble.get_transition_prob(state, next_state), 0.0)

        self.assertEqual(ensemble.get_transition_prob(state_2, state), 1.0 / 11)
        next_state = State({sf: 3})
        self.assertEqual(ensemble.get_transition_prob(state_2, next_state), 1.0 / 11)
        next_state = State({sf: 6})
        self.assertEqual(ensemble.get_transition_prob(state_2, next_state), 1.0 / 11)

        DBNOptionEnsemble._setup_ensemble = setup


class GetRewardTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", {}
        )

        sf = IntStateFactor("sf", 0, 10)
        state = State({sf: 1})
        ensemble._reward_dict[state] = 7

        self.assertEqual(ensemble.get_reward(state), 7)

        state = State({sf: 4})
        self.assertEqual(ensemble.get_reward(state), 0.0)

        DBNOptionEnsemble._setup_ensemble = setup


class GetSCXMLTransitionsTest(unittest.TestCase):

    def test_function(self):
        # This also tests check valid probs and get_name
        sf = StateFactor("sf", ["a", "b", "c"])

        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        state_idx_map = {State({sf: "a"}): 0, State({sf: "b"}): 1, State({sf: "c"}): 2}
        ensemble = DBNOptionEnsemble(
            "opt", [], 32, 100, [sf], "enabled_cond", state_idx_map
        )
        ensemble._enabled_states = [
            State({sf: "a"}),
            State({sf: "b"}),
            State({sf: "c"}),
        ]

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        ensemble._sampled_transition_dict = {
            State({sf: "a"}): {b_cond: 0.6, c_cond: 0.4},
            State({sf: "b"}): {a_cond: 0.3, c_cond: 0.7},
            State({sf: "c"}): None,
        }

        scxml_transitions = ensemble.get_scxml_transitions(["sf"], "policy")
        xml_string_1 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_1 += '<transition target="init" event="opt" cond="sf==0">'
        xml_string_1 += '<assign location="rand" expr="Math.random()" />'
        xml_string_1 += '<if cond="rand &lt;= 0.6">'
        xml_string_1 += '<assign location="sf" expr="1" />'
        xml_string_1 += "<else />"
        xml_string_1 += '<assign location="sf" expr="2" />'
        xml_string_1 += "</if>"
        xml_string_1 += '<send event="update_datamodel" target="policy">'
        xml_string_1 += '<param name="sf" expr="sf" />'
        xml_string_1 += "</send>"
        xml_string_1 += "</transition>"

        trans_str = et.tostring(scxml_transitions[0], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_1)

        xml_string_2 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_2 += '<transition target="init" event="opt" cond="sf==1">'
        xml_string_2 += '<assign location="rand" expr="Math.random()" />'
        xml_string_2 += '<if cond="rand &lt;= 0.3">'
        xml_string_2 += '<assign location="sf" expr="0" />'
        xml_string_2 += "<else />"
        xml_string_2 += '<assign location="sf" expr="2" />'
        xml_string_2 += "</if>"
        xml_string_2 += '<send event="update_datamodel" target="policy">'
        xml_string_2 += '<param name="sf" expr="sf" />'
        xml_string_2 += "</send>"
        xml_string_2 += "</transition>"

        trans_str = et.tostring(scxml_transitions[1], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_2)

        xml_string_3 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_3 += '<transition target="init" event="opt" cond="sf==2">'
        xml_string_3 += '<assign location="rand" expr="Math.random()" />'
        xml_string_3 += '<if cond="rand &lt;= {}">'.format(1.0 / 3)
        xml_string_3 += '<assign location="sf" expr="0" />'
        xml_string_3 += '<elseif cond="rand &lt;= {}" />'.format(1.0 / 3 + 1.0 / 3)
        xml_string_3 += '<assign location="sf" expr="1" />'
        xml_string_3 += "<else />"
        xml_string_3 += '<assign location="sf" expr="2" />'
        xml_string_3 += "</if>"
        xml_string_3 += '<send event="update_datamodel" target="policy">'
        xml_string_3 += '<param name="sf" expr="sf" />'
        xml_string_3 += "</send>"
        xml_string_3 += "</transition>"

        trans_str = et.tostring(scxml_transitions[2], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_3)

        ensemble._sampled_transition_dict = {State({sf: "a"}): {b_cond: 1.0}}
        ensemble._enabled_states = [State({sf: "a"})]
        xml_string_4 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_4 += '<transition target="init" event="opt" cond="sf==0">'
        xml_string_4 += '<assign location="sf" expr="1" />'
        xml_string_4 += '<send event="update_datamodel" target="policy">'
        xml_string_4 += '<param name="sf" expr="sf" />'
        xml_string_4 += "</send>"
        xml_string_4 += "</transition>"
        scxml_transitions = ensemble.get_scxml_transitions(["sf"], "policy")
        trans_str = et.tostring(scxml_transitions[0], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_4)

        ensemble._sampled_transition_dict = {
            State({sf: "a"}): {a_cond: 0.2, b_cond: 0.3, c_cond: 0.5}
        }
        xml_string_5 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_5 += '<transition target="init" event="opt" cond="sf==0">'
        xml_string_5 += '<assign location="rand" expr="Math.random()" />'
        xml_string_5 += '<if cond="rand &lt;= 0.2">'
        xml_string_5 += '<assign location="sf" expr="0" />'
        xml_string_5 += '<elseif cond="rand &lt;= 0.5" />'
        xml_string_5 += '<assign location="sf" expr="1" />'
        xml_string_5 += "<else />"
        xml_string_5 += '<assign location="sf" expr="2" />'
        xml_string_5 += "</if>"
        xml_string_5 += '<send event="update_datamodel" target="policy">'
        xml_string_5 += '<param name="sf" expr="sf" />'
        xml_string_5 += "</send>"
        xml_string_5 += "</transition>"
        scxml_transitions = ensemble.get_scxml_transitions(["sf"], "policy")
        trans_str = et.tostring(scxml_transitions[0], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_5)

        DBNOptionEnsemble._setup_ensemble = setup


class GetTransitionPRISMStringTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", {}
        )

        with self.assertRaises(Exception):
            ensemble.get_transition_prism_string()

        ensemble._transition_prism_str = "test"

        self.assertEqual(ensemble.get_transition_prism_string(), "test")

        DBNOptionEnsemble._setup_ensemble = setup


class GetRewardPRISMStringTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", {}
        )

        with self.assertRaises(Exception):
            ensemble.get_reward_prism_string()

        ensemble._reward_prism_str = "test"

        self.assertEqual(ensemble.get_reward_prism_string(), "test")

        DBNOptionEnsemble._setup_ensemble = setup


class ComputeEntropyTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 32, 100, "sf_list", "enabled_cond", {}
        )

        dist = {"a": 0.7, "b": 0.2, "c": 0.1}

        entropy = ensemble._compute_entropy(dist)

        self.assertAlmostEqual(entropy, 1.15677964945)

        DBNOptionEnsemble._setup_ensemble = setup

    def test_uniform(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        state_idx_map = {"s": 0, "t": 1, "n1": 2, "n2": 3, "n3": 4}
        ensemble = DBNOptionEnsemble(
            "option", [], 4, 100, "sf_list", "enabled_cond", state_idx_map
        )

        entropy = ensemble._compute_entropy(None)
        self.assertAlmostEqual(entropy, 2.32192809489)

        dist = {"n1": 0.275, "n2": 0.275, "n3": 0.25, None: 0.1}
        entropy = ensemble._compute_entropy(dist)
        self.assertAlmostEqual(entropy, 2.18875868092)

        DBNOptionEnsemble._setup_ensemble = setup


class ComputeAvgDistTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 2, 100, "sf_list", "enabled_cond", {}
        )

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        avg_dist = ensemble._compute_avg_dist("s")
        expected = {"n1": 0.35, "n2": 0.35, "n3": 0.3}
        self.assertEqual(avg_dist, expected)

        DBNOptionEnsemble._setup_ensemble = setup

    def test_uniform(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        state_idx_map = {"s": 0, "t": 1, "n1": 2, "n2": 3, "n3": 4}
        ensemble = DBNOptionEnsemble(
            "option", [], 4, 100, "sf_list", "enabled_cond", state_idx_map
        )

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": None}
        ensemble._transition_dicts[2] = {"s": {"n2": 0.4, "n3": 0.6}}
        ensemble._transition_dicts[3] = {"s": None}

        avg_dist = ensemble._compute_avg_dist("s")
        expected = {"n1": 0.275, "n2": 0.275, "n3": 0.25, None: 0.1}
        self.assertEqual(avg_dist, expected)

        DBNOptionEnsemble._setup_ensemble = setup


class ComputeInfoGainTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 2, 100, "sf_list", "enabled_cond", {}
        )

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        info_gain = ensemble._compute_info_gain("s")
        self.assertAlmostEqual(info_gain, 0.65517015239)

        DBNOptionEnsemble._setup_ensemble = setup

    def test_uniform(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        state_idx_map = {"s": 0, "t": 1, "n1": 2, "n2": 3, "n3": 4}
        ensemble = DBNOptionEnsemble(
            "option", [], 4, 100, "sf_list", "enabled_cond", state_idx_map
        )

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": None}
        ensemble._transition_dicts[2] = {"s": {"n2": 0.4, "n3": 0.6}}
        ensemble._transition_dicts[3] = {"s": None}

        expected = {"n1": 0.275, "n2": 0.275, "n3": 0.25, None: 0.1}

        info_gain = ensemble._compute_info_gain("s")

        self.assertAlmostEqual(info_gain, 0.56473426005)

        DBNOptionEnsemble._setup_ensemble = setup


class IdentifyEnabledStatesTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        enabled_cond = EqCondition(sf_list[0], True)
        state_map = {
            State({sf_list[0]: False, sf_list[1]: False}): 0,
            State({sf_list[0]: False, sf_list[1]: True}): 1,
            State({sf_list[0]: True, sf_list[1]: False}): 2,
            State({sf_list[0]: True, sf_list[1]: True}): 3,
        }
        ensemble = DBNOptionEnsemble(
            "option", [], 2, 100, sf_list, enabled_cond, state_map
        )
        ensemble._identify_enabled_states()
        self.assertEqual(
            ensemble._enabled_states,
            [
                State({sf_list[0]: True, sf_list[1]: False}),
                State({sf_list[0]: True, sf_list[1]: True}),
            ],
        )

        DBNOptionEnsemble._setup_ensemble = setup


class CreateDatasetsTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        ensemble = DBNOptionEnsemble("option", [], 2, 100, sf_list, "enabled_cond", {})

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

        DBNOptionEnsemble._setup_ensemble = setup


class LearnDBNOptionsTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [IntStateFactor("x", 1, 3), BoolStateFactor("y")]
        enabled_cond = GeqCondition(sf_list[0], 1)
        ensemble = DBNOptionEnsemble("option", [], 2, 100, sf_list, enabled_cond, {})
        dataset = {
            "transition": {
                "x0": [1, 2, 3],
                "xt": [2, 3, 1],
                "y0": [False, True, False],
                "yt": [False, False, True],
            },
            "reward": {"x": [1, 2, 3], "y": [False, False, True], "r": [5, 5, 7]},
        }

        datasets = [dataset, dataset]
        ensemble._learn_dbn_options(datasets)

        self.assertTrue(isinstance(ensemble._dbns[0], DBNOption))
        self.assertTrue(isinstance(ensemble._dbns[1], DBNOption))

        self.assertEqual(ensemble._dbns[0]._sf_list, sf_list)
        self.assertEqual(ensemble._dbns[1]._sf_list, sf_list)

        self.assertEqual(ensemble._dbns[0]._enabled_cond, ensemble._enabled_cond)
        self.assertEqual(ensemble._dbns[1]._enabled_cond, ensemble._enabled_cond)

        self.assertEqual(
            sorted(list(ensemble._dbns[0]._reward_dbn.names())), sorted(["y", "r"])
        )
        self.assertEqual(
            sorted(list(ensemble._dbns[1]._reward_dbn.names())), sorted(["y", "r"])
        )

        self.assertEqual(
            sorted(list(ensemble._dbns[0]._transition_dbn.names())),
            sorted(["x0", "xt", "y0", "yt"]),
        )
        self.assertEqual(
            sorted(list(ensemble._dbns[1]._transition_dbn.names())),
            sorted(["x0", "xt", "y0", "yt"]),
        )

        DBNOptionEnsemble._setup_ensemble = setup


class BuildTransitionDictForDBNTest(unittest.TestCase):

    def test_function(self):

        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y"), BoolStateFactor("z")]
        state_idx_map = {
            State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: False}): 0,
            State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: True}): 1,
            State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: False}): 2,
            State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: True}): 3,
            State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: False}): 4,
            State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: True}): 5,
            State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: False}): 6,
            State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: True}): 7,
        }
        ensemble = DBNOptionEnsemble(
            "test", [], 2, 100, sf_list, TrueCondition(), state_idx_map
        )
        ensemble._identify_enabled_states()
        t_bn = gum.BayesNet()
        _ = t_bn.add(gum.LabelizedVariable("x0", "x0?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("xt", "xt?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("y0", "y0?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("yt", "yt?", ["False", "True"]))
        t_bn.addArc("x0", "xt")
        t_bn.addArc("x0", "yt")
        t_bn.addArc("y0", "xt")
        t_bn.addArc("y0", "yt")

        t_bn.cpt("xt")[{"x0": "False", "y0": "False"}] = [0.4, 0.6]
        t_bn.cpt("xt")[{"x0": "False", "y0": "True"}] = [0.5, 0.5]
        t_bn.cpt("xt")[{"x0": "True", "y0": "False"}] = [0.6, 0.4]
        t_bn.cpt("xt")[{"x0": "True", "y0": "True"}] = [0.3, 0.7]

        t_bn.cpt("yt")[{"x0": "False", "y0": "False"}] = [0.8, 0.2]
        t_bn.cpt("yt")[{"x0": "False", "y0": "True"}] = [0.1, 0.9]
        t_bn.cpt("yt")[{"x0": "True", "y0": "False"}] = [0.7, 0.3]
        t_bn.cpt("yt")[{"x0": "True", "y0": "True"}] = [0.2, 0.8]

        # Create the reward DBN
        r_bn = gum.BayesNet()
        _ = r_bn.add(gum.LabelizedVariable("x", "x?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("y", "y?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("r", "r?", ["0", "1", "2", "3"]))
        r_bn.addArc("x", "r")
        r_bn.addArc("y", "r")

        r_bn.cpt("x").fillWith([0.5, 0.5])
        r_bn.cpt("y").fillWith([0.5, 0.5])
        r_bn.cpt("r")[{"x": "False", "y": "False"}] = [0.0, 0.2, 0.3, 0.5]
        r_bn.cpt("r")[{"x": "False", "y": "True"}] = [0.0, 0.6, 0.1, 0.3]
        r_bn.cpt("r")[{"x": "True", "y": "False"}] = [0.0, 0.4, 0.4, 0.2]
        r_bn.cpt("r")[{"x": "True", "y": "True"}] = [0.0, 0.3, 0.3, 0.4]

        ensemble._dbns[0] = DBNOption(
            "test",
            None,
            None,
            sf_list,
            TrueCondition(),
            transition_dbn=t_bn,
            reward_dbn=r_bn,
        )

        queue = SimpleQueue()

        ensemble._build_transition_dict_for_dbn(0, queue)

        dbn_idx, transition_dict = queue.get()

        self.assertEqual(dbn_idx, 0)
        self.assertEqual(len(transition_dict), 8)

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        state_fff = State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: False})
        state_fft = State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: True})
        state_ftf = State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: False})
        state_ftt = State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: True})
        state_tff = State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: False})
        state_tft = State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: True})
        state_ttf = State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: False})
        state_ttt = State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: True})

        ff_ff = ensemble._dbns[0].get_transition_prob(state_fff, state_fff)
        ff_ft = ensemble._dbns[0].get_transition_prob(state_fff, state_ftf)
        ff_tf = ensemble._dbns[0].get_transition_prob(state_fff, state_tff)
        ff_tt = ensemble._dbns[0].get_transition_prob(state_fff, state_ttf)

        ff_post_conds = {
            state_ff.to_and_cond(): ff_ff,
            state_ft.to_and_cond(): ff_ft,
            state_tf.to_and_cond(): ff_tf,
            state_tt.to_and_cond(): ff_tt,
        }

        self.assertEqual(transition_dict[state_fff], ff_post_conds)
        self.assertEqual(transition_dict[state_fft], ff_post_conds)

        ft_ff = ensemble._dbns[0].get_transition_prob(state_ftf, state_fff)
        ft_ft = ensemble._dbns[0].get_transition_prob(state_ftf, state_ftf)
        ft_tf = ensemble._dbns[0].get_transition_prob(state_ftf, state_tff)
        ft_tt = ensemble._dbns[0].get_transition_prob(state_ftf, state_ttf)

        ft_post_conds = {
            state_ff.to_and_cond(): ft_ff,
            state_ft.to_and_cond(): ft_ft,
            state_tf.to_and_cond(): ft_tf,
            state_tt.to_and_cond(): ft_tt,
        }

        self.assertEqual(transition_dict[state_ftf], ft_post_conds)
        self.assertEqual(transition_dict[state_ftt], ft_post_conds)

        tf_ff = ensemble._dbns[0].get_transition_prob(state_tff, state_fff)
        tf_ft = ensemble._dbns[0].get_transition_prob(state_tff, state_ftf)
        tf_tf = ensemble._dbns[0].get_transition_prob(state_tff, state_tff)
        tf_tt = ensemble._dbns[0].get_transition_prob(state_tff, state_ttf)

        tf_post_conds = {
            state_ff.to_and_cond(): tf_ff,
            state_ft.to_and_cond(): tf_ft,
            state_tf.to_and_cond(): tf_tf,
            state_tt.to_and_cond(): tf_tt,
        }

        self.assertEqual(transition_dict[state_tff], tf_post_conds)
        self.assertEqual(transition_dict[state_tft], tf_post_conds)

        tt_ff = ensemble._dbns[0].get_transition_prob(state_ttf, state_fff)
        tt_ft = ensemble._dbns[0].get_transition_prob(state_ttf, state_ftf)
        tt_tf = ensemble._dbns[0].get_transition_prob(state_ttf, state_tff)
        tt_tt = ensemble._dbns[0].get_transition_prob(state_ttf, state_ttf)

        tt_post_conds = {
            state_ff.to_and_cond(): tt_ff,
            state_ft.to_and_cond(): tt_ft,
            state_tf.to_and_cond(): tt_tf,
            state_tt.to_and_cond(): tt_tt,
        }

        self.assertEqual(transition_dict[state_ttf], tt_post_conds)
        self.assertEqual(transition_dict[state_ttt], tt_post_conds)

        DBNOptionEnsemble._setup_ensemble = setup

    def test_uniform_dists(self):

        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y"), BoolStateFactor("z")]
        state_idx_map = {
            State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: False}): 0,
            State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: True}): 1,
            State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: False}): 2,
            State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: True}): 3,
            State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: False}): 4,
            State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: True}): 5,
            State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: False}): 6,
            State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: True}): 7,
        }
        ensemble = DBNOptionEnsemble(
            "test", [], 2, 100, sf_list, TrueCondition(), state_idx_map
        )
        ensemble._identify_enabled_states()
        t_bn = gum.BayesNet()
        _ = t_bn.add(gum.LabelizedVariable("x0", "x0?", ["False"]))
        _ = t_bn.add(gum.LabelizedVariable("xt", "xt?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("y0", "y0?", ["False", "True"]))
        _ = t_bn.add(gum.LabelizedVariable("yt", "yt?", ["False", "True"]))
        t_bn.addArc("x0", "xt")
        t_bn.addArc("x0", "yt")
        t_bn.addArc("y0", "xt")
        t_bn.addArc("y0", "yt")

        t_bn.cpt("xt")[{"x0": "False", "y0": "False"}] = [0.4, 0.6]
        t_bn.cpt("xt")[{"x0": "False", "y0": "True"}] = [0.5, 0.5]

        t_bn.cpt("yt")[{"x0": "False", "y0": "False"}] = [0.8, 0.2]
        t_bn.cpt("yt")[{"x0": "False", "y0": "True"}] = [0.1, 0.9]

        # Create the reward DBN
        r_bn = gum.BayesNet()
        _ = r_bn.add(gum.LabelizedVariable("x", "x?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("y", "y?", ["False", "True"]))
        _ = r_bn.add(gum.LabelizedVariable("r", "r?", ["0", "1", "2", "3"]))
        r_bn.addArc("x", "r")
        r_bn.addArc("y", "r")

        r_bn.cpt("x").fillWith([0.5, 0.5])
        r_bn.cpt("y").fillWith([0.5, 0.5])
        r_bn.cpt("r")[{"x": "False", "y": "False"}] = [0.0, 0.2, 0.3, 0.5]
        r_bn.cpt("r")[{"x": "False", "y": "True"}] = [0.0, 0.6, 0.1, 0.3]
        r_bn.cpt("r")[{"x": "True", "y": "False"}] = [0.0, 0.4, 0.4, 0.2]
        r_bn.cpt("r")[{"x": "True", "y": "True"}] = [0.0, 0.3, 0.3, 0.4]

        ensemble._dbns[0] = DBNOption(
            "test",
            None,
            None,
            sf_list,
            TrueCondition(),
            transition_dbn=t_bn,
            reward_dbn=r_bn,
        )

        queue = SimpleQueue()

        ensemble._build_transition_dict_for_dbn(0, queue)

        dbn_idx, transition_dict = queue.get()

        self.assertEqual(dbn_idx, 0)
        self.assertEqual(len(transition_dict), 8)

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        state_fff = State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: False})
        state_fft = State({sf_list[0]: False, sf_list[1]: False, sf_list[2]: True})
        state_ftf = State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: False})
        state_ftt = State({sf_list[0]: False, sf_list[1]: True, sf_list[2]: True})
        state_tff = State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: False})
        state_tft = State({sf_list[0]: True, sf_list[1]: False, sf_list[2]: True})
        state_ttf = State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: False})
        state_ttt = State({sf_list[0]: True, sf_list[1]: True, sf_list[2]: True})

        ff_ff = ensemble._dbns[0].get_transition_prob(state_fff, state_fff)
        ff_ft = ensemble._dbns[0].get_transition_prob(state_fff, state_ftf)
        ff_tf = ensemble._dbns[0].get_transition_prob(state_fff, state_tff)
        ff_tt = ensemble._dbns[0].get_transition_prob(state_fff, state_ttf)

        ff_post_conds = {
            state_ff.to_and_cond(): ff_ff,
            state_ft.to_and_cond(): ff_ft,
            state_tf.to_and_cond(): ff_tf,
            state_tt.to_and_cond(): ff_tt,
        }

        self.assertEqual(transition_dict[state_fff], ff_post_conds)
        self.assertEqual(transition_dict[state_fft], ff_post_conds)

        ft_ff = ensemble._dbns[0].get_transition_prob(state_ftf, state_fff)
        ft_ft = ensemble._dbns[0].get_transition_prob(state_ftf, state_ftf)
        ft_tf = ensemble._dbns[0].get_transition_prob(state_ftf, state_tff)
        ft_tt = ensemble._dbns[0].get_transition_prob(state_ftf, state_ttf)

        ft_post_conds = {
            state_ff.to_and_cond(): ft_ff,
            state_ft.to_and_cond(): ft_ft,
            state_tf.to_and_cond(): ft_tf,
            state_tt.to_and_cond(): ft_tt,
        }

        self.assertEqual(transition_dict[state_ftf], ft_post_conds)
        self.assertEqual(transition_dict[state_ftt], ft_post_conds)

        self.assertEqual(transition_dict[state_tff], None)
        self.assertEqual(transition_dict[state_tft], None)
        self.assertEqual(transition_dict[state_ttf], None)
        self.assertEqual(transition_dict[state_ttt], None)

        DBNOptionEnsemble._setup_ensemble = setup


class BuildTransitionDictsTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        def set_dbn_idx(s, i, q):
            q.put((i, {i: i + 1}))

        build_trans_dict = DBNOptionEnsemble._build_transition_dict_for_dbn
        DBNOptionEnsemble._build_transition_dict_for_dbn = set_dbn_idx

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        ensemble = DBNOptionEnsemble("option", [], 2, 100, sf_list, "enabled_cond", {})

        ensemble._build_transition_dicts()

        self.assertEqual(ensemble._transition_dicts, [{0: 1}, {1: 2}])

        DBNOptionEnsemble._setup_ensemble = setup
        DBNOptionEnsemble._build_transition_dict_for_dbn = build_trans_dict


class ComputeSampledTransitionsAndInfoGainTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        ensemble = DBNOptionEnsemble(
            "option", [], 2, 100, "sf_list", "enabled_cond", {}
        )
        ensemble._enabled_states = ["s"]
        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n2": 0.4, "n3": 0.6}}

        ensemble._compute_sampled_transitions_and_info_gain()

        self.assertTrue(
            ensemble._sampled_transition_dict["s"] == {"n1": 0.7, "n2": 0.3}
            or ensemble._sampled_transition_dict["s"] == {"n2": 0.4, "n3": 0.6}
        )

        self.assertEqual(len(ensemble._reward_dict), 1)
        self.assertAlmostEqual(ensemble._reward_dict["s"], 0.65517015239)

        ensemble._transition_dicts[0] = {"s": {"n1": 0.7, "n2": 0.3}}
        ensemble._transition_dicts[1] = {"s": {"n1": 0.7, "n2": 0.3}}

        ensemble._compute_sampled_transitions_and_info_gain()
        self.assertEqual(ensemble._sampled_transition_dict["s"], {"n1": 0.7, "n2": 0.3})
        self.assertEqual(len(ensemble._reward_dict), 1)

        DBNOptionEnsemble._setup_ensemble = setup


class BuildMatricesTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None

        sf = StateFactor("sf", ["s", "t", "n1", "n2", "n3"])

        state_idx_map = {
            State({sf: "s"}): 0,
            State({sf: "t"}): 1,
            State({sf: "n1"}): 2,
            State({sf: "n2"}): 3,
            State({sf: "n3"}): 4,
        }

        ensemble = DBNOptionEnsemble(
            "option", [], 2, 100, [sf], "enabled_cond", state_idx_map
        )

        ensemble._sampled_transition_dict[State({sf: "s"})] = {
            EqCondition(sf, "n1"): 0.7,
            EqCondition(sf, "n2"): 0.3,
        }
        ensemble._sampled_transition_dict[State({sf: "t"})] = {
            EqCondition(sf, "n2"): 0.4,
            EqCondition(sf, "n3"): 0.6,
        }
        ensemble._sampled_transition_dict[State({sf: "n1"})] = None

        ensemble._reward_dict[State({sf: "s"})] = 0.65517015239
        ensemble._reward_dict[State({sf: "t"})] = 0.7

        ensemble._build_matrices()

        reward_mat = np.zeros(5)
        reward_mat[0] = 0.65517015239
        reward_mat[1] = 0.7

        self.assertTrue(np.array_equal(ensemble._reward_mat, reward_mat))

        trans_mat = np.zeros((5, 5))
        trans_mat[0, 2] = 0.7
        trans_mat[0, 3] = 0.3
        trans_mat[1, 3] = 0.4
        trans_mat[1, 4] = 0.6
        trans_mat[2, 0] = 0.2
        trans_mat[2, 1] = 0.2
        trans_mat[2, 2] = 0.2
        trans_mat[2, 3] = 0.2
        trans_mat[2, 4] = 0.2

        self.assertTrue(np.array_equal(ensemble._sampled_transition_mat, trans_mat))

        DBNOptionEnsemble._setup_ensemble = setup


class PrecomputePRISMStringsTest(unittest.TestCase):

    def test_function(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None
        sf = StateFactor("sf", ["a", "b", "c"])

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        a_state = State({sf: "a"})
        b_state = State({sf: "b"})
        c_state = State({sf: "c"})

        state_idx_map = {a_state: 0, b_state: 1, c_state: 2}

        ensemble = DBNOptionEnsemble(
            "opt", [], 2, 100, [sf], "enabled_cond", state_idx_map
        )

        ensemble._enabled_states = [a_state, b_state]

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        a_state = State({sf: "a"})
        b_state = State({sf: "b"})

        ensemble._sampled_transition_dict = {
            a_state: {b_cond: 0.6, c_cond: 0.4},
            b_state: {a_cond: 0.3, c_cond: 0.7},
        }

        ensemble._reward_dict = {
            a_state: 7.5,
            b_state: 1.3,
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

        self.assertTrue(
            EqCondition(sf, "b") in ensemble._sampled_transition_dict[a_state]
        )
        self.assertTrue(
            EqCondition(sf, "c") in ensemble._sampled_transition_dict[a_state]
        )
        self.assertTrue(
            EqCondition(sf, "a") in ensemble._sampled_transition_dict[b_state]
        )
        self.assertTrue(
            EqCondition(sf, "c") in ensemble._sampled_transition_dict[b_state]
        )

        DBNOptionEnsemble._setup_ensemble = setup

    def test_uniform(self):
        setup = DBNOptionEnsemble._setup_ensemble
        DBNOptionEnsemble._setup_ensemble = lambda s, d: None
        sf = StateFactor("sf", ["a", "b", "c"])

        a_cond = EqCondition(sf, "a")
        b_cond = EqCondition(sf, "b")
        c_cond = EqCondition(sf, "c")

        a_state = State({sf: "a"})
        b_state = State({sf: "b"})
        c_state = State({sf: "c"})

        state_idx_map = {a_state: 0, b_state: 1, c_state: 2}

        ensemble = DBNOptionEnsemble(
            "opt", [], 2, 100, [sf], "enabled_cond", state_idx_map
        )

        ensemble._enabled_states = [a_state, b_state, c_state]
        ensemble._sampled_transition_dict = {
            a_state: {b_cond: 0.6, c_cond: 0.4},
            b_state: {a_cond: 0.3, c_cond: 0.7},
            c_state: None,
        }

        ensemble._reward_dict = {
            a_state: 7.5,
            b_state: 1.3,
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
        trans_str += (
            "[opt] ((sf = 2) & (time < 100)) -> {}:(sf' = 0) & (time' = time + 1) + "
        ).format(1.0 / 3)
        trans_str += "{}:(sf' = 1) & (time' = time + 1) + ".format(1.0 / 3)
        trans_str += "{}:(sf' = 2) & (time' = time + 1);\n".format(1.0 / 3)
        self.assertEqual(ensemble._transition_prism_str, trans_str)

        reward_str = "[opt] ((sf = 0) & (time < 100)): 7.5;\n"
        reward_str += "[opt] ((sf = 1) & (time < 100)): 1.3;\n"

        self.assertEqual(ensemble._reward_prism_str, reward_str)

        self.assertTrue(
            EqCondition(sf, "b") in ensemble._sampled_transition_dict[a_state]
        )
        self.assertTrue(
            EqCondition(sf, "c") in ensemble._sampled_transition_dict[a_state]
        )
        self.assertTrue(
            EqCondition(sf, "a") in ensemble._sampled_transition_dict[b_state]
        )
        self.assertTrue(
            EqCondition(sf, "c") in ensemble._sampled_transition_dict[b_state]
        )

        DBNOptionEnsemble._setup_ensemble = setup


class SetupEnsembleTest(unittest.TestCase):

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

        ensemble = DBNOptionEnsemble(
            "check_door",
            data,
            3,
            100,
            sf_list,
            enabled_cond,
            _build_state_idx_map(sf_list),
        )

        enabled_states = []
        sf_dict = {sf.get_name(): sf for sf in sf_list}
        for loc in ["v{}".format(i) for i in range(2, 8)]:
            unused_sfs = []
            for sf_name in sf_dict:
                if sf_name != "location" and sf_name != "{}_door".format(loc):
                    unused_sfs.append(sf_dict[sf_name])
            unused_sf_vals = [sf.get_valid_values() for sf in unused_sfs]
            for rest_of_state in itertools.product(*unused_sf_vals):
                state_dict = {
                    sf_dict["location"]: loc,
                    sf_dict["{}_door".format(loc)]: "unknown",
                }
                for i in range(len(rest_of_state)):
                    state_dict[unused_sfs[i]] = rest_of_state[i]

                enabled_states.append(State(state_dict))

        self.assertEqual(len(enabled_states), len(ensemble._enabled_states))
        self.assertEqual(enabled_states, ensemble._enabled_states)

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


if __name__ == "__main__":
    unittest.main()
