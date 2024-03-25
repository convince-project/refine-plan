#!/usr/bin/env python3
""" Unit tests for behaviour_tree.py.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import TrueCondition
from refine_plan.models.behaviour_tree import (
    ActionNode,
    ConditionNode,
    SequenceNode,
    FallbackNode,
    BehaviourTree,
)
import unittest


class ActionNodeTest(unittest.TestCase):

    def test_function(self):
        node = ActionNode("action")
        self.assertEqual(node.get_name(), "action")

        with self.assertRaises(NotImplementedError):
            node.to_BT_XML()


class ConditionNodeTest(unittest.TestCase):

    def test_function(self):
        node = ConditionNode("condition", TrueCondition())

        self.assertEqual(node.get_name(), "condition")
        self.assertEqual(node.get_cond(), TrueCondition())

        with self.assertRaises(NotImplementedError):
            node.to_BT_XML()


class SequenceNodeTest(unittest.TestCase):

    def test_function(self):
        node = SequenceNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        self.assertEqual(len(node._children), 2)
        node.add_child(ActionNode("act_2"))
        self.assertEqual(len(node._children), 3)

        with self.assertRaises(NotImplementedError):
            node.to_BT_XML()

        with self.assertRaises(Exception):
            SequenceNode("bad")

        with self.assertRaises(Exception):
            node.add_child("bad")


class FallbackNodeTest(unittest.TestCase):

    def test_function(self):
        node = FallbackNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        self.assertEqual(len(node._children), 2)
        node.add_child(ActionNode("act_2"))
        self.assertEqual(len(node._children), 3)

        with self.assertRaises(NotImplementedError):
            node.to_BT_XML()

        with self.assertRaises(Exception):
            FallbackNode("bad")

        with self.assertRaises(Exception):
            node.add_child("bad")


class BehaviourTreeTest(unittest.TestCase):

    def test_function(self):
        root_node = FallbackNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        bt = BehaviourTree(root_node)
        self.assertEqual(bt.get_root_node(), root_node)

        with self.assertRaises(Exception):
            bt.to_BT_XML()


if __name__ == "__main__":
    unittest.main()
