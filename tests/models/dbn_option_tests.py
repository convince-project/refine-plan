#!/usr/bin/env python3
"""Unit tests for DBNOption.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import BoolStateFactor, StateFactor, IntStateFactor
from refine_plan.models.condition import (
    TrueCondition,
    NotCondition,
    AndCondition,
    EqCondition,
    OrCondition,
    AddCondition,
)
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.state import State
from pyeda.inter import expr, Not, And, Or
import xml.etree.ElementTree as et
import pyAgrum as gum
import unittest
import os


def create_two_bns():
    """Aux function which generates the test BNs."""

    # Create the transition function DBN
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

    # Save the BNs
    t_bn.saveBIFXML("transition.bifxml")
    r_bn.saveBIFXML("reward.bifxml")


class ConstructorTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]

        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        self.assertEqual(option.get_name(), "test")
        self.assertEqual(option._sf_list, sf_list)
        self.assertEqual(option._transition_dbn, gum.loadBN("transition.bifxml"))
        self.assertEqual(option._reward_dbn, gum.loadBN("reward.bifxml"))
        self.assertTrue(option._enabled_cond.is_satisfied("state"))

        option = DBNOption(
            "test",
            "transition.bifxml",
            "reward.bifxml",
            sf_list,
            NotCondition(TrueCondition()),
        )
        self.assertFalse(option._enabled_cond.is_satisfied("state"))

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class CheckValidDBNsTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]

        # Test 1: Bad variable in transition
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("xy", "xy?", ["False", "True"])
        )

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 2: Variable not in state factors in transition DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True"])
        )

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 3: Deleted variable in transition DBN
        # Allowed as not all state factors are required
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._transition_dbn.erase("x0")
        option._check_valid_dbns()

        # Test 4: Remove 'r' from reward DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._reward_dbn.erase("r")

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 5: Remove one of the state factors from reward DBN
        # Allowed as not all state factors are required
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._reward_dbn.erase("x")
        option._check_valid_dbns()

        # Test 6: Change one of the x0 value ranges (add one sf to deal with this)
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._sf_list = sf_list + [BoolStateFactor("z")]
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True", "Dunno"])
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("zt", "zt?", ["False", "True"])
        )
        option._reward_dbn.add(gum.LabelizedVariable("z", "z?", ["False", "True"]))
        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 7: Change one of the xt value ranges (add one sf to deal with this)
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._sf_list = sf_list + [BoolStateFactor("z")]
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True"])
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("zt", "zt?", ["False", "True", "Dunno"])
        )
        option._reward_dbn.add(gum.LabelizedVariable("z", "z?", ["False", "True"]))
        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 8: Change one of the reward value ranges (add one sf to deal with this)
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._sf_list = sf_list + [BoolStateFactor("z")]
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True"])
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("zt", "zt?", ["False", "True"])
        )
        option._reward_dbn.add(
            gum.LabelizedVariable("z", "z?", ["False", "True", "Dunno"])
        )
        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 9: Change one of the state factor value ranges (add one sf to deal with this)
        # This should pass because we only need a subset :)
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )
        option._sf_list = sf_list + [StateFactor("z", [False, True, "Dunno"])]
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True"])
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("zt", "zt?", ["False", "True"])
        )
        option._reward_dbn.add(gum.LabelizedVariable("z", "z?", ["False", "True"]))
        option._check_valid_dbns()

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class PruneDistsTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "False"}] = [
            0.0001,
            0.9999,
        ]
        option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "True"}] = [0.5, 0.5]
        option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "False"}] = [
            0.00001,
            0.99999,
        ]
        option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "True"}] = [0.3, 0.7]

        option._transition_dbn.cpt("yt")[{"x0": "False", "y0": "False"}] = [0.8, 0.2]
        option._transition_dbn.cpt("yt")[{"x0": "False", "y0": "True"}] = [0.1, 0.9]
        option._transition_dbn.cpt("yt")[{"x0": "True", "y0": "False"}] = [
            0.000001,
            0.999999,
        ]
        option._transition_dbn.cpt("yt")[{"x0": "True", "y0": "True"}] = [0.2, 0.8]

        option._prune_dists()

        xt_ff = option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "False"}]
        xt_ft = option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "True"}]
        xt_tf = option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "False"}]
        xt_tt = option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "True"}]

        yt_ff = option._transition_dbn.cpt("yt")[{"x0": "False", "y0": "False"}]
        yt_ft = option._transition_dbn.cpt("yt")[{"x0": "False", "y0": "True"}]
        yt_tf = option._transition_dbn.cpt("yt")[{"x0": "True", "y0": "False"}]
        yt_tt = option._transition_dbn.cpt("yt")[{"x0": "True", "y0": "True"}]

        self.assertEqual(list(xt_ff), [0.0, 1.0])
        self.assertEqual(list(xt_ft), [0.5, 0.5])
        self.assertEqual(list(xt_tf), [0.0, 1.0])
        self.assertEqual(list(xt_tt), [0.3, 0.7])

        self.assertEqual(list(yt_ff), [0.8, 0.2])
        self.assertEqual(list(yt_ft), [0.1, 0.9])
        self.assertEqual(list(yt_tf), [0.0, 1.0])
        self.assertEqual(list(yt_tt), [0.2, 0.8])

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")

        option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "False"}] = [
            0.0001,
            0.9999,
        ]
        option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "True"}] = [0.5, 0.5]
        option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "False"}] = [
            0.00001,
            0.99999,
        ]
        option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "True"}] = [0.3, 0.7]

        option._prune_dists()

        xt_ff = option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "False"}]
        xt_ft = option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "True"}]
        xt_tf = option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "False"}]
        xt_tt = option._transition_dbn.cpt("xt")[{"x0": "True", "y0": "True"}]

        self.assertFalse("yt" in option._transition_dbn.names())

        self.assertEqual(list(xt_ff), [0.0, 1.0])
        self.assertEqual(list(xt_ft), [0.5, 0.5])
        self.assertEqual(list(xt_tf), [0.0, 1.0])
        self.assertEqual(list(xt_tt), [0.3, 0.7])

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class ExpectedValFnTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        labels = option._reward_dbn.variableFromName("r").labels()
        self.assertEqual(sorted(list(labels)), ["0", "1", "2", "3"])

        x = {"r": 0}
        ex = option._expected_val_fn(x)
        self.assertTrue(isinstance(ex, float))
        self.assertEqual(ex, float(labels[0]))

        x["r"] = 1
        ex = option._expected_val_fn(x)
        self.assertTrue(isinstance(ex, float))
        self.assertEqual(ex, float(labels[1]))

        x["r"] = 2
        ex = option._expected_val_fn(x)
        self.assertTrue(isinstance(ex, float))
        self.assertEqual(ex, float(labels[2]))

        x["r"] = 3
        ex = option._expected_val_fn(x)
        self.assertTrue(isinstance(ex, float))
        self.assertEqual(ex, float(labels[3]))

        x["r"] = 4
        with self.assertRaises(Exception):
            option._expected_val_fn(x)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class StrToSfValsTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        str_sf_vals = [
            ["False", "True"],
            ["1", "2", "3"],
            ["val1", "val2", "val3"],
            ["4", "5", "6"],
        ]

        option._sf_list = [
            BoolStateFactor("a"),
            IntStateFactor("b", 1, 3),
            StateFactor("c", ["val1", "val2", "val3"]),
            StateFactor("d", ["4", "5", "6"]),
        ]

        sf_vals = option._str_to_sf_vals(str_sf_vals, option._sf_list)

        self.assertEqual(
            sf_vals,
            [[False, True], [1, 2, 3], ["val1", "val2", "val3"], ["4", "5", "6"]],
        )

        sf_vals = option._str_to_sf_vals(str_sf_vals[:-1], option._sf_list[:-1])
        self.assertEqual(
            sf_vals,
            [[False, True], [1, 2, 3], ["val1", "val2", "val3"]],
        )

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetTransitionProbTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ff), 0.32)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ft), 0.08)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tf), 0.48)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tt), 0.12)

        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ff), 0.05)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ft), 0.45)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tf), 0.05)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tt), 0.45)

        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ff), 0.42)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ft), 0.18)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tf), 0.28)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tt), 0.12)

        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ff), 0.06)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ft), 0.24)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tf), 0.14)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tt), 0.56)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_enabled_actions(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        enabled_cond = NotCondition(
            AndCondition(EqCondition(sf_list[0], True), EqCondition(sf_list[1], False))
        )
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, enabled_cond
        )

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ff), 0.32)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ft), 0.08)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tf), 0.48)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tt), 0.12)

        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ff), 0.05)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ft), 0.45)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tf), 0.05)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tt), 0.45)

        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ff), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ft), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tf), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tt), 0.0)

        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ff), 0.06)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ft), 0.24)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tf), 0.14)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tt), 0.56)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ff), 0.4)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ft), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tf), 0.6)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tt), 0.0)

        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ff), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ft), 0.5)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tf), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tt), 0.5)

        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ff), 0.6)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ft), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tf), 0.4)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tt), 0.0)

        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ff), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ft), 0.3)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tf), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tt), 0.7)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_removed_sf(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")
        option._transition_dbn.erase("y0")

        option._transition_dbn.cpt("xt")[{"x0": "False"}] = [0.4, 0.6]
        option._transition_dbn.cpt("xt")[{"x0": "True"}] = [0.5, 0.5]

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ff), 0.4)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_ft), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tf), 0.6)
        self.assertAlmostEqual(option.get_transition_prob(state_ff, state_tt), 0.0)

        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ff), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_ft), 0.4)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tf), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_ft, state_tt), 0.6)

        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ff), 0.5)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_ft), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tf), 0.5)
        self.assertAlmostEqual(option.get_transition_prob(state_tf, state_tt), 0.0)

        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ff), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_ft), 0.5)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tf), 0.0)
        self.assertAlmostEqual(option.get_transition_prob(state_tt, state_tt), 0.5)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetRewardTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_reward(state_ff), 2.3)
        self.assertAlmostEqual(option.get_reward(state_ft), 1.7)
        self.assertAlmostEqual(option.get_reward(state_tf), 1.8)
        self.assertAlmostEqual(option.get_reward(state_tt), 2.1)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_enabled_actions(self):
        create_two_bns()

        def is_enabled(s):
            if s["x"] == True and s["y"] == False:
                return False
            return True

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        enabled_cond = NotCondition(
            AndCondition(EqCondition(sf_list[0], True), EqCondition(sf_list[1], False))
        )
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, enabled_cond
        )

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        self.assertAlmostEqual(option.get_reward(state_ff), 2.3)
        self.assertAlmostEqual(option.get_reward(state_ft), 1.7)
        self.assertAlmostEqual(option.get_reward(state_tf), 0.0)
        self.assertAlmostEqual(option.get_reward(state_tt), 2.1)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        option._reward_dbn.erase("y")
        option._reward_dbn.cpt("r")[{"x": "False"}] = [0.2, 0.2, 0.2, 0.4]
        option._reward_dbn.cpt("r")[{"x": "True"}] = [0.0, 0.3, 0.4, 0.3]
        self.assertAlmostEqual(option.get_reward(state_ff), 1.8)
        self.assertAlmostEqual(option.get_reward(state_ft), 1.8)
        self.assertAlmostEqual(option.get_reward(state_tf), 2.0)
        self.assertAlmostEqual(option.get_reward(state_tt), 2.0)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetValsAndSfsTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")

        # Test 1: Transition with all variables with suffix available
        vars_used, vals, sfs_used = option._get_vals_and_sfs(
            option._transition_dbn, "0"
        )

        self.assertEqual(vars_used, ["x0", "y0"])
        self.assertEqual(vals, [[False, True], [False, True]])
        self.assertEqual(sfs_used, sf_list)

        # Test 2: Transition with some variables with suffix unavailable
        vars_used, vals, sfs_used = option._get_vals_and_sfs(
            option._transition_dbn, "t"
        )

        self.assertEqual(vars_used, ["xt"])
        self.assertEqual(vals, [[False, True]])
        self.assertEqual(sfs_used, [sf_list[0]])

        # Test 3: Reward DBN with all variables available
        vars_used, vals, sfs_used = option._get_vals_and_sfs(option._reward_dbn, "")

        self.assertEqual(vars_used, ["x", "y"])
        self.assertEqual(vals, [[False, True], [False, True]])
        self.assertEqual(sfs_used, sf_list)

        # Test 4: Reward DBN with some variables unavailable
        option._reward_dbn.erase("x")
        vars_used, vals, sfs_used = option._get_vals_and_sfs(option._reward_dbn, "")

        self.assertEqual(vars_used, ["y"])
        self.assertEqual(vals, [[False, True]])
        self.assertEqual(sfs_used, [sf_list[1]])

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class SubStateInfoIntoEnabledCondTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        v2_door_sf = BoolStateFactor("v2_door")
        v3_door_sf = BoolStateFactor("v3_door")

        cond = TrueCondition()

        self.assertEqual(
            option._sub_state_info_into_enabled_cond(None, cond), TrueCondition()
        )

        cond = NotCondition(EqCondition(loc_sf, "v2"))
        state = State({loc_sf: "v2", v2_door_sf: False})
        new_cond = option._sub_state_info_into_enabled_cond(state, cond)
        self.assertEqual(new_cond, NotCondition(TrueCondition()))

        state = State({loc_sf: "v3", v2_door_sf: False})
        new_cond = option._sub_state_info_into_enabled_cond(state, cond)
        self.assertEqual(new_cond, NotCondition(NotCondition(TrueCondition())))

        cond = NotCondition(EqCondition(v3_door_sf, True))
        new_cond = option._sub_state_info_into_enabled_cond(state, cond)
        self.assertEqual(new_cond, cond)

        int_sf = IntStateFactor("bad", 1, 3)
        cond = AddCondition(int_sf, 1)
        with self.assertRaises(Exception):
            option._sub_state_info_into_enabled_cond(state, cond)

        cond = AndCondition(EqCondition(loc_sf, "v2"), EqCondition(v2_door_sf, True))
        state = State({loc_sf: "v3", v2_door_sf: True, v3_door_sf: True})
        new_cond = option._sub_state_info_into_enabled_cond(state, cond)
        expected_cond = AndCondition(NotCondition(TrueCondition()), TrueCondition())
        self.assertEqual(new_cond, expected_cond)

        cond = OrCondition(EqCondition(loc_sf, "v2"), EqCondition(v2_door_sf, True))
        state = State({v2_door_sf: True, v3_door_sf: True})
        new_cond = option._sub_state_info_into_enabled_cond(state, cond)
        expected_cond = OrCondition(EqCondition(loc_sf, "v2"), TrueCondition())
        self.assertEqual(new_cond, expected_cond)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class PyedaToCondTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        v2_door_sf = BoolStateFactor("v2_door")

        pyeda_expr = expr(True)
        var_map = {}
        self.assertEqual(option._pyeda_to_cond(pyeda_expr, var_map), TrueCondition())

        pyeda_expr = expr(False)
        self.assertEqual(
            option._pyeda_to_cond(pyeda_expr, var_map), NotCondition(TrueCondition())
        )

        pyeda_expr = expr("locEQv2")
        var_map = {"locEQv2": EqCondition(loc_sf, "v2")}
        cond = option._pyeda_to_cond(pyeda_expr, var_map)
        self.assertEqual(cond, EqCondition(loc_sf, "v2"))

        pyeda_expr = Not(expr("locEQv2"))
        cond = option._pyeda_to_cond(pyeda_expr, var_map)
        self.assertEqual(cond, NotCondition(EqCondition(loc_sf, "v2")))

        pyeda_expr = And(Not(expr("locEQv2")), expr("v2_doorEQTrue"))
        var_map["v2_doorEQTrue"] = EqCondition(v2_door_sf, True)
        cond = option._pyeda_to_cond(pyeda_expr, var_map)
        self.assertEqual(
            cond,
            AndCondition(
                NotCondition(EqCondition(loc_sf, "v2")), EqCondition(v2_door_sf, True)
            ),
        )

        pyeda_expr = Or(Not(expr("locEQv2")), expr("v2_doorEQTrue"))
        cond = option._pyeda_to_cond(pyeda_expr, var_map)
        self.assertEqual(
            cond,
            OrCondition(
                NotCondition(EqCondition(loc_sf, "v2")), EqCondition(v2_door_sf, True)
            ),
        )

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetPrismGuardForStateTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        v2_door_sf = BoolStateFactor("v2_door")
        v3_door_sf = BoolStateFactor("v3_door")

        option._sf_list = [loc_sf, v2_door_sf, v3_door_sf]
        option._enabled_cond = EqCondition(loc_sf, "v2")

        state = State({loc_sf: "v2", v2_door_sf: True, v3_door_sf: False})
        guard = option._get_prism_guard_for_state(state)

        self.assertTrue(isinstance(guard, AndCondition))
        self.assertEqual(len(guard._cond_list), 3)
        self.assertTrue(EqCondition(loc_sf, "v2") in guard._cond_list)
        self.assertTrue(EqCondition(v2_door_sf, True) in guard._cond_list)
        self.assertTrue(EqCondition(v3_door_sf, False) in guard._cond_list)

        state = State({loc_sf: "v3", v2_door_sf: True, v3_door_sf: False})
        guard = option._get_prism_guard_for_state(state)
        self.assertEqual(guard, None)

        option._enabled_cond = AndCondition(
            EqCondition(loc_sf, "v2"),
            EqCondition(v2_door_sf, False),
            EqCondition(v3_door_sf, False),
        )

        state = State({loc_sf: "v2"})
        guard = option._get_prism_guard_for_state(state)
        self.assertTrue(isinstance(guard, AndCondition))
        self.assertEqual(len(guard._cond_list), 3)
        self.assertTrue(EqCondition(loc_sf, "v2") in guard._cond_list)
        self.assertTrue(EqCondition(v2_door_sf, False) in guard._cond_list)
        self.assertTrue(EqCondition(v3_door_sf, False) in guard._cond_list)

        option._enabled_cond = AndCondition(
            EqCondition(loc_sf, "v2"),
            OrCondition(
                EqCondition(v2_door_sf, False),
                EqCondition(v3_door_sf, False),
            ),
        )
        guard = option._get_prism_guard_for_state(state)
        self.assertTrue(isinstance(guard, AndCondition))
        self.assertEqual(len(guard._cond_list), 2)
        self.assertTrue(EqCondition(loc_sf, "v2") in guard._cond_list)
        self.assertTrue(
            OrCondition(EqCondition(v2_door_sf, False), EqCondition(v3_door_sf, False))
            in guard._cond_list
        )

        option._enabled_cond = OrCondition(
            AndCondition(EqCondition(loc_sf, "v2"), EqCondition(v2_door_sf, True)),
            AndCondition(EqCondition(loc_sf, "v3"), EqCondition(v3_door_sf, True)),
        )
        guard = option._get_prism_guard_for_state(state)
        self.assertTrue(isinstance(guard, AndCondition))
        self.assertEqual(len(guard._cond_list), 2)
        self.assertTrue(EqCondition(loc_sf, "v2") in guard._cond_list)
        self.assertTrue(EqCondition(v2_door_sf, True) in guard._cond_list)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class PrunePosteriorTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")
        option._transition_dbn.erase("y0")

        option._transition_dbn.cpt("xt")[{"x0": "False"}] = [0.4, 0.6]
        option._transition_dbn.cpt("xt")[{"x0": "True"}] = [0.001, 0.9999]

        state_f = State({sf_list[0]: False})
        state_t = State({sf_list[0]: True})

        # Testing indirectly through get_transition_prob
        ff = option.get_transition_prob(state_f, state_f)
        ft = option.get_transition_prob(state_f, state_t)
        tf = option.get_transition_prob(state_t, state_f)
        tt = option.get_transition_prob(state_t, state_t)

        self.assertEqual(ff, 0.4)
        self.assertEqual(ft, 0.6)
        self.assertEqual(tf, 0.0)
        self.assertEqual(tt, 1.0)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetTransitionPrismStringTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        prism_str = option.get_transition_prism_string()

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        ff_ff = option.get_transition_prob(state_ff, state_ff)
        ff_ft = option.get_transition_prob(state_ff, state_ft)
        ff_tf = option.get_transition_prob(state_ff, state_tf)
        ff_tt = option.get_transition_prob(state_ff, state_tt)

        ft_ff = option.get_transition_prob(state_ft, state_ff)
        ft_ft = option.get_transition_prob(state_ft, state_ft)
        ft_tf = option.get_transition_prob(state_ft, state_tf)
        ft_tt = option.get_transition_prob(state_ft, state_tt)

        tf_ff = option.get_transition_prob(state_tf, state_ff)
        tf_ft = option.get_transition_prob(state_tf, state_ft)
        tf_tf = option.get_transition_prob(state_tf, state_tf)
        tf_tt = option.get_transition_prob(state_tf, state_tt)

        tt_ff = option.get_transition_prob(state_tt, state_ff)
        tt_ft = option.get_transition_prob(state_tt, state_ft)
        tt_tf = option.get_transition_prob(state_tt, state_tf)
        tt_tt = option.get_transition_prob(state_tt, state_tt)

        expected = "[test] ((x = 0) & (y = 0)) -> {}:(x' = 0) & (y' = 0) + ".format(
            ff_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            ff_ft, ff_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(ff_tt)

        expected += "[test] ((x = 0) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            ft_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            ft_ft, ft_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(ft_tt)

        expected += "[test] ((x = 1) & (y = 0)) -> {}:(x' = 0) & (y' = 0) + ".format(
            tf_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            tf_ft, tf_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(tf_tt)

        expected += "[test] ((x = 1) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            tt_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            tt_ft, tt_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(tt_tt)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_zero_cost_self_loops(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.cpt("xt")[{"x0": "False", "y0": "False"}] = [1.0, 0.0]
        option._transition_dbn.cpt("yt")[{"x0": "False", "y0": "False"}] = [1.0, 0.0]
        option._reward_dbn.cpt("r")[{"x": "False", "y": "False"}] = [1.0, 0.0, 0.0, 0.0]

        prism_str = option.get_transition_prism_string()

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        ft_ff = option.get_transition_prob(state_ft, state_ff)
        ft_ft = option.get_transition_prob(state_ft, state_ft)
        ft_tf = option.get_transition_prob(state_ft, state_tf)
        ft_tt = option.get_transition_prob(state_ft, state_tt)

        tf_ff = option.get_transition_prob(state_tf, state_ff)
        tf_ft = option.get_transition_prob(state_tf, state_ft)
        tf_tf = option.get_transition_prob(state_tf, state_tf)
        tf_tt = option.get_transition_prob(state_tf, state_tt)

        tt_ff = option.get_transition_prob(state_tt, state_ff)
        tt_ft = option.get_transition_prob(state_tt, state_ft)
        tt_tf = option.get_transition_prob(state_tt, state_tf)
        tt_tt = option.get_transition_prob(state_tt, state_tt)

        expected = "[test] ((x = 0) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            ft_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            ft_ft, ft_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(ft_tt)

        expected += "[test] ((x = 1) & (y = 0)) -> {}:(x' = 0) & (y' = 0) + ".format(
            tf_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            tf_ft, tf_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(tf_tt)

        expected += "[test] ((x = 1) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            tt_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            tt_ft, tt_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(tt_tt)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_enabled_actions(self):
        create_two_bns()

        def is_enabled(s):
            if s["x"] == True and s["y"] == False:
                return False
            return True

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        enabled_cond = NotCondition(
            AndCondition(EqCondition(sf_list[0], True), EqCondition(sf_list[1], False))
        )
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, enabled_cond
        )

        prism_str = option.get_transition_prism_string()

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        ff_ff = option.get_transition_prob(state_ff, state_ff)
        ff_ft = option.get_transition_prob(state_ff, state_ft)
        ff_tf = option.get_transition_prob(state_ff, state_tf)
        ff_tt = option.get_transition_prob(state_ff, state_tt)

        ft_ff = option.get_transition_prob(state_ft, state_ff)
        ft_ft = option.get_transition_prob(state_ft, state_ft)
        ft_tf = option.get_transition_prob(state_ft, state_tf)
        ft_tt = option.get_transition_prob(state_ft, state_tt)

        tt_ff = option.get_transition_prob(state_tt, state_ff)
        tt_ft = option.get_transition_prob(state_tt, state_ft)
        tt_tf = option.get_transition_prob(state_tt, state_tf)
        tt_tt = option.get_transition_prob(state_tt, state_tt)

        expected = "[test] ((x = 0) & (y = 0)) -> {}:(x' = 0) & (y' = 0) + ".format(
            ff_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            ff_ft, ff_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(ff_tt)

        expected += "[test] ((x = 0) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            ft_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            ft_ft, ft_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(ft_tt)

        expected += "[test] ((x = 1) & (y = 1)) -> {}:(x' = 0) & (y' = 0) + ".format(
            tt_ff
        )
        expected += "{}:(x' = 0) & (y' = 1) + {}:(x' = 1) & (y' = 0) + ".format(
            tt_ft, tt_tf
        )
        expected += "{}:(x' = 1) & (y' = 1); \n".format(tt_tt)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")

        prism_str = option.get_transition_prism_string()

        state_ff = State({sf_list[0]: False, sf_list[1]: False})
        state_ft = State({sf_list[0]: False, sf_list[1]: True})
        state_tf = State({sf_list[0]: True, sf_list[1]: False})
        state_tt = State({sf_list[0]: True, sf_list[1]: True})

        ff_f = option.get_transition_prob(state_ff, state_ff)
        ff_t = option.get_transition_prob(state_ff, state_tf)

        ft_f = option.get_transition_prob(state_ft, state_ft)
        ft_t = option.get_transition_prob(state_ft, state_tt)

        tf_f = option.get_transition_prob(state_tf, state_ff)
        tf_t = option.get_transition_prob(state_tf, state_tf)

        tt_f = option.get_transition_prob(state_tt, state_ft)
        tt_t = option.get_transition_prob(state_tt, state_tt)

        expected = "[test] ((x = 0) & (y = 0)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
            ff_f, ff_t
        )
        expected += (
            "[test] ((x = 0) & (y = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
                ft_f, ft_t
            )
        )
        expected += (
            "[test] ((x = 1) & (y = 0)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
                tf_f, tf_t
            )
        )
        expected += (
            "[test] ((x = 1) & (y = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
                tt_f, tt_t
            )
        )

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_removed_sf(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._transition_dbn.erase("yt")
        option._transition_dbn.erase("y0")

        option._transition_dbn.cpt("xt")[{"x0": "False"}] = [0.4, 0.6]
        option._transition_dbn.cpt("xt")[{"x0": "True"}] = [0.5, 0.5]

        prism_str = option.get_transition_prism_string()

        state_f = State({sf_list[0]: False})
        state_t = State({sf_list[0]: True})

        f_f = option.get_transition_prob(state_f, state_f)
        f_t = option.get_transition_prob(state_f, state_t)
        t_f = option.get_transition_prob(state_t, state_f)
        t_t = option.get_transition_prob(state_t, state_t)

        expected = "[test] ((x = 0)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(f_f, f_t)
        expected += "[test] ((x = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(t_f, t_t)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs_and_extra_enabled_sf(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        option._sf_list.append(loc_sf)
        option._enabled_cond = AndCondition(
            OrCondition(EqCondition(sf_list[0], False), EqCondition(sf_list[0], True)),
            EqCondition(loc_sf, "v2"),
        )

        option._transition_dbn.erase("yt")

        prism_str = option.get_transition_prism_string()

        state_ff = State({sf_list[0]: False, sf_list[1]: False, loc_sf: "v2"})
        state_ft = State({sf_list[0]: False, sf_list[1]: True, loc_sf: "v2"})
        state_tf = State({sf_list[0]: True, sf_list[1]: False, loc_sf: "v2"})
        state_tt = State({sf_list[0]: True, sf_list[1]: True, loc_sf: "v2"})

        ff_f = option.get_transition_prob(state_ff, state_ff)
        ff_t = option.get_transition_prob(state_ff, state_tf)

        ft_f = option.get_transition_prob(state_ft, state_ft)
        ft_t = option.get_transition_prob(state_ft, state_tt)

        tf_f = option.get_transition_prob(state_tf, state_ff)
        tf_t = option.get_transition_prob(state_tf, state_tf)

        tt_f = option.get_transition_prob(state_tt, state_ft)
        tt_t = option.get_transition_prob(state_tt, state_tt)

        expected = "[test] ((x = 0) & (y = 0) & (loc = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
            ff_f, ff_t
        )
        expected += "[test] ((x = 0) & (y = 1) & (loc = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
            ft_f, ft_t
        )
        expected += "[test] ((x = 1) & (y = 0) & (loc = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
            tf_f, tf_t
        )
        expected += "[test] ((x = 1) & (y = 1) & (loc = 1)) -> {}:(x' = 0) + {}:(x' = 1); \n".format(
            tt_f, tt_t
        )
        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetRewardPrismStringTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        prism_str = option.get_reward_prism_string()

        ff_r = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0
        ft_r = 0.6 * 1.0 + 0.1 * 2.0 + 0.3 * 3.0
        tf_r = 0.4 * 1.0 + 0.4 * 2.0 + 0.2 * 3.0
        tt_r = 0.3 * 1.0 + 0.3 * 2.0 + 0.4 * 3.0

        expected = "[test] ((x = 0) & (y = 0)): {};\n".format(ff_r)
        expected += "[test] ((x = 0) & (y = 1)): {};\n".format(ft_r)
        expected += "[test] ((x = 1) & (y = 0)): {};\n".format(tf_r)
        expected += "[test] ((x = 1) & (y = 1)): {};\n".format(tt_r)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_enabled_actions(self):
        create_two_bns()

        def is_enabled(s):
            if s["x"] == True and s["y"] == False:
                return False
            return True

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        enabled_cond = NotCondition(
            AndCondition(EqCondition(sf_list[0], True), EqCondition(sf_list[1], False))
        )
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, enabled_cond
        )

        prism_str = option.get_reward_prism_string()

        ff_r = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0
        ft_r = 0.6 * 1.0 + 0.1 * 2.0 + 0.3 * 3.0
        tt_r = 0.3 * 1.0 + 0.3 * 2.0 + 0.4 * 3.0

        expected = "[test] ((x = 0) & (y = 0)): {};\n".format(ff_r)
        expected += "[test] ((x = 0) & (y = 1)): {};\n".format(ft_r)
        expected += "[test] ((x = 1) & (y = 1)): {};\n".format(tt_r)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        option._reward_dbn.erase("y")
        option._reward_dbn.cpt("r")[{"x": "False"}] = [0.2, 0.2, 0.2, 0.4]
        option._reward_dbn.cpt("r")[{"x": "True"}] = [0.0, 0.3, 0.4, 0.3]

        prism_str = option.get_reward_prism_string()

        f_r = 0.2 * 1.0 + 0.2 * 2.0 + 0.4 * 3.0
        t_r = 0.3 * 1.0 + 0.4 * 2.0 + 0.3 * 3.0

        expected = "[test] ((x = 0)): {};\n".format(f_r)
        expected += "[test] ((x = 1)): {};\n".format(t_r)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")

    def test_fewer_sfs_and_extra_enabled_sf(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        option._sf_list.append(loc_sf)
        option._enabled_cond = AndCondition(
            OrCondition(EqCondition(sf_list[0], False), EqCondition(sf_list[0], True)),
            EqCondition(loc_sf, "v2"),
        )

        option._reward_dbn.erase("y")
        option._reward_dbn.cpt("r")[{"x": "False"}] = [0.2, 0.2, 0.2, 0.4]
        option._reward_dbn.cpt("r")[{"x": "True"}] = [0.0, 0.3, 0.4, 0.3]

        prism_str = option.get_reward_prism_string()

        f_r = 0.2 * 1.0 + 0.2 * 2.0 + 0.4 * 3.0
        t_r = 0.3 * 1.0 + 0.4 * 2.0 + 0.3 * 3.0

        expected = "[test] ((x = 0) & (loc = 1)): {};\n".format(f_r)
        expected += "[test] ((x = 1) & (loc = 1)): {};\n".format(t_r)

        self.assertEqual(prism_str, expected)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class GetSCXMLTransitionsTest(unittest.TestCase):
    def test_function(self):
        # Testing is fairly small here as this uses many functions tested
        # extensively in option.py and get_transition_prism_string in this file.
        # Just a sanity check test that everything links together properly.
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, TrueCondition()
        )

        loc_sf = StateFactor("loc", ["v1", "v2", "v3", "v4"])
        option._sf_list.append(loc_sf)
        option._enabled_cond = AndCondition(
            OrCondition(EqCondition(sf_list[0], False), EqCondition(sf_list[0], True)),
            EqCondition(loc_sf, "v2"),
        )

        option._transition_dbn.erase("yt")

        state_ff = State({sf_list[0]: False, sf_list[1]: False, loc_sf: "v2"})
        state_ft = State({sf_list[0]: False, sf_list[1]: True, loc_sf: "v2"})
        state_tf = State({sf_list[0]: True, sf_list[1]: False, loc_sf: "v2"})
        state_tt = State({sf_list[0]: True, sf_list[1]: True, loc_sf: "v2"})

        ff_f = option.get_transition_prob(state_ff, state_ff)
        ft_f = option.get_transition_prob(state_ft, state_ft)
        tf_f = option.get_transition_prob(state_tf, state_ff)
        tt_f = option.get_transition_prob(state_tt, state_ft)

        scxml_transitions = option.get_scxml_transitions()
        self.assertEqual(len(scxml_transitions), 4)

        xml_string_1 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_1 += '<transition target="init" event="test" cond="x==0 &amp;&amp; y==0 &amp;&amp; loc==1">'
        xml_string_1 += '<assign location="rand" expr="Math.random()" />'
        xml_string_1 += '<if cond="rand &lt;= {}">'.format(ff_f)
        xml_string_1 += '<assign location="x" expr="0" />'
        xml_string_1 += "<else />"
        xml_string_1 += '<assign location="x" expr="1" />'
        xml_string_1 += "</if>"
        xml_string_1 += "</transition>"

        trans_str = et.tostring(scxml_transitions[0], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_1)

        xml_string_2 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_2 += '<transition target="init" event="test" cond="x==0 &amp;&amp; y==1 &amp;&amp; loc==1">'
        xml_string_2 += '<assign location="rand" expr="Math.random()" />'
        xml_string_2 += '<if cond="rand &lt;= {}">'.format(ft_f)
        xml_string_2 += '<assign location="x" expr="0" />'
        xml_string_2 += "<else />"
        xml_string_2 += '<assign location="x" expr="1" />'
        xml_string_2 += "</if>"
        xml_string_2 += "</transition>"

        trans_str = et.tostring(scxml_transitions[1], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_2)

        xml_string_3 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_3 += '<transition target="init" event="test" cond="x==1 &amp;&amp; y==0 &amp;&amp; loc==1">'
        xml_string_3 += '<assign location="rand" expr="Math.random()" />'
        xml_string_3 += '<if cond="rand &lt;= {}">'.format(tf_f)
        xml_string_3 += '<assign location="x" expr="0" />'
        xml_string_3 += "<else />"
        xml_string_3 += '<assign location="x" expr="1" />'
        xml_string_3 += "</if>"
        xml_string_3 += "</transition>"

        trans_str = et.tostring(scxml_transitions[2], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_3)

        xml_string_4 = "<?xml version='1.0' encoding='utf8'?>\n"
        xml_string_4 += '<transition target="init" event="test" cond="x==1 &amp;&amp; y==1 &amp;&amp; loc==1">'
        xml_string_4 += '<assign location="rand" expr="Math.random()" />'
        xml_string_4 += '<if cond="rand &lt;= {}">'.format(tt_f)
        xml_string_4 += '<assign location="x" expr="0" />'
        xml_string_4 += "<else />"
        xml_string_4 += '<assign location="x" expr="1" />'
        xml_string_4 += "</if>"
        xml_string_4 += "</transition>"

        trans_str = et.tostring(scxml_transitions[3], encoding="utf8").decode("utf8")
        self.assertEqual(trans_str, xml_string_4)

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


if __name__ == "__main__":
    unittest.main()
