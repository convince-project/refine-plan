#!/usr/bin/env python3
""" Unit tests for DBNOption.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import BoolStateFactor, StateFactor
from refine_plan.models.dbn_option import DBNOption
from refine_plan.models.state import State
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )

        self.assertEqual(option.get_name(), "test")
        self.assertEqual(option._sf_list, sf_list)
        self.assertEqual(option._transition_dbn, gum.loadBN("transition.bifxml"))
        self.assertEqual(option._reward_dbn, gum.loadBN("reward.bifxml"))
        self.assertTrue(option._is_enabled("state"))

        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: False
        )
        self.assertFalse(option._is_enabled("state"))

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class CheckValidDBNsTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]

        # Test 1: Bad variable in transition
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("xy", "xy?", ["False", "True"])
        )

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 2: Variable not in state factors in transition DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )
        option._transition_dbn.add(
            gum.LabelizedVariable("z0", "z0?", ["False", "True"])
        )

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 3: Deleted variable in transition DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )
        option._transition_dbn.erase("x0")

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 4: Remove 'r' from reward DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )
        option._reward_dbn.erase("r")

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 5: Remove one of the state factors from reward DBN
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
        )
        option._reward_dbn.erase("x")

        with self.assertRaises(Exception):
            option._check_valid_dbns()

        # Test 6: Change one of the x0 value ranges (add one sf to deal with this)
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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


class ExpectedValFnTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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


class GetTransitionProbTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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

        def is_enabled(s):
            if s["x"] == True and s["y"] == False:
                return False
            return True

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, is_enabled
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


class GetRewardTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, is_enabled
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


class GetTransitionPrismStringTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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


class GetRewardPrismStringTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]
        option = DBNOption(
            "test", "transition.bifxml", "reward.bifxml", sf_list, lambda s: True
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


if __name__ == "__main__":
    unittest.main()
