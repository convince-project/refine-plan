#!/usr/bin/env python3
""" Unit tests for DBNOption.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import BoolStateFactor
from refine_plan.models.dbn_option import DBNOption
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
    _ = r_bn.add(gum.LabelizedVariable("r", "r?", ["1", "2", "3"]))

    r_bn.cpt("x").fillWith([0.5, 0.5])
    r_bn.cpt("y").fillWith([0.5, 0.5])
    r_bn.cpt("r")[{"x": "False", "y": "False"}] = [0.2, 0.3, 0.5]
    r_bn.cpt("r")[{"x": "False", "y": "True"}] = [0.6, 0.1, 0.3]
    r_bn.cpt("r")[{"x": "True", "y": "False"}] = [0.4, 0.4, 0.2]
    r_bn.cpt("r")[{"x": "True", "y": "True"}] = [0.3, 0.3, 0.4]

    # Save the BNs
    t_bn.saveBIFXML("transition.bifxml")
    r_bn.saveBIFXML("reward.bifxml")


class ConstructorTest(unittest.TestCase):

    def test_function(self):
        create_two_bns()

        sf_list = [BoolStateFactor("x"), BoolStateFactor("y")]

        option = DBNOption("test", "transition.bifxml", "reward.bifxml", sf_list)

        self.assertEqual(option.get_name(), "test")
        self.assertEqual(option._sf_list, sf_list)
        self.assertEqual(option._transition_dbn, gum.loadBN("transition.bifxml"))
        self.assertEqual(option._reward_dbn, gum.loadBN("reward.bifxml"))

        os.remove("transition.bifxml")
        os.remove("reward.bifxml")


class CheckValidDBNs(unittest.TestCase):

    def test_function(self):
        pass


class ExpectedValFnTest(unittest.TestCase):

    def test_function(self):
        pass


class GetTransitionProbTest(unittest.TestCase):

    def test_function(self):
        pass


class GetRewardTest(unittest.TestCase):

    def test_function(self):
        pass


class GetTransitionPrismStringTest(unittest.TestCase):

    def test_function(self):
        pass


class GetRewardPrismStringTest(unittest.TestCase):

    def test_function(self):
        pass


if __name__ == "__main__":
    unittest.main()
