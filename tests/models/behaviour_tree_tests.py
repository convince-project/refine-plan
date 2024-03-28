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
import xml.etree.ElementTree as et
import tempfile
import unittest


class ActionNodeTest(unittest.TestCase):

    def test_function(self):
        node = ActionNode("action")
        self.assertEqual(node.get_name(), "action")

        xml = node.to_BT_XML()
        self.assertEqual(len(xml), 0)
        self.assertEqual(xml.tag, "Action")
        self.assertEqual(xml.attrib, {"name": "action"})

        self.assertEqual(repr(node), "Action(action)")
        self.assertEqual(str(node), "Action(action)")


class ConditionNodeTest(unittest.TestCase):

    def test_function(self):
        node = ConditionNode("condition", TrueCondition())

        self.assertEqual(node.get_name(), "condition")
        self.assertEqual(node.get_cond(), TrueCondition())

        xml = node.to_BT_XML()
        self.assertEqual(len(xml), 0)
        self.assertEqual(xml.tag, "Condition")
        self.assertEqual(xml.attrib, {"name": "condition"})

        self.assertEqual(repr(node), "Condition(condition; true)")
        self.assertEqual(str(node), "Condition(condition; true)")


class SequenceNodeTest(unittest.TestCase):

    def test_function(self):
        node = SequenceNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        self.assertEqual(len(node._children), 2)
        node.add_child(ActionNode("act_2"))
        self.assertEqual(len(node._children), 3)

        xml = node.to_BT_XML()
        self.assertEqual(len(xml), 3)
        self.assertEqual(xml.tag, "Sequence")
        self.assertEqual(xml.attrib, {})
        self.assertEqual(len(xml[0]), 0)
        self.assertEqual(xml[0].tag, "Condition")
        self.assertEqual(xml[0].attrib, {"name": "condition"})
        self.assertEqual(len(xml[1]), 0)
        self.assertEqual(xml[1].tag, "Action")
        self.assertEqual(xml[1].attrib, {"name": "action"})
        self.assertEqual(len(xml[2]), 0)
        self.assertEqual(xml[2].tag, "Action")
        self.assertEqual(xml[2].attrib, {"name": "act_2"})

        with self.assertRaises(Exception):
            SequenceNode("bad")

        with self.assertRaises(Exception):
            node.add_child("bad")

        self.assertEqual(
            repr(node),
            "Sequence(Condition(condition; true), Action(action), Action(act_2))",
        )

        self.assertEqual(
            str(node),
            "Sequence(Condition(condition; true), Action(action), Action(act_2))",
        )


class FallbackNodeTest(unittest.TestCase):

    def test_function(self):
        node = FallbackNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        self.assertEqual(len(node._children), 2)
        node.add_child(ActionNode("act_2"))
        self.assertEqual(len(node._children), 3)

        xml = node.to_BT_XML()
        self.assertEqual(len(xml), 3)
        self.assertEqual(xml.tag, "Fallback")
        self.assertEqual(xml.attrib, {})
        self.assertEqual(len(xml[0]), 0)
        self.assertEqual(xml[0].tag, "Condition")
        self.assertEqual(xml[0].attrib, {"name": "condition"})
        self.assertEqual(len(xml[1]), 0)
        self.assertEqual(xml[1].tag, "Action")
        self.assertEqual(xml[1].attrib, {"name": "action"})
        self.assertEqual(len(xml[2]), 0)
        self.assertEqual(xml[2].tag, "Action")
        self.assertEqual(xml[2].attrib, {"name": "act_2"})
        with self.assertRaises(Exception):
            FallbackNode("bad")

        with self.assertRaises(Exception):
            node.add_child("bad")

        self.assertEqual(
            repr(node),
            "Fallback(Condition(condition; true), Action(action), Action(act_2))",
        )

        self.assertEqual(
            str(node),
            "Fallback(Condition(condition; true), Action(action), Action(act_2))",
        )


class BehaviourTreeTest(unittest.TestCase):

    def test_function(self):
        root_node = FallbackNode(
            ConditionNode("condition", TrueCondition()), ActionNode("action")
        )

        bt = BehaviourTree(root_node)
        self.assertEqual(bt.get_root_node(), root_node)

        self.assertEqual(
            repr(bt), "Fallback(Condition(condition; true), Action(action))"
        )

        self.assertEqual(
            str(bt), "Fallback(Condition(condition; true), Action(action))"
        )

        tmp = tempfile.NamedTemporaryFile()
        tree = bt.to_BT_XML(tmp.name)
        xml = tree.getroot()

        self.assertEqual(len(xml), 1)
        self.assertEqual(xml.tag, "root")
        self.assertEqual(xml.attrib, {"main_tree_to_execute": "MainTree"})
        self.assertEqual(len(xml[0]), 1)
        self.assertEqual(xml[0].tag, "BehaviorTree")
        self.assertEqual(xml[0].attrib, {"ID": "MainTree"})
        self.assertEqual(len(xml[0][0]), 2)
        self.assertEqual(xml[0][0].tag, "Fallback")
        self.assertEqual(xml[0][0].attrib, {})
        self.assertEqual(len(xml[0][0][0]), 0)
        self.assertEqual(xml[0][0][0].tag, "Condition")
        self.assertEqual(xml[0][0][0].attrib, {"name": "condition"})
        self.assertEqual(len(xml[0][0][1]), 0)
        self.assertEqual(xml[0][0][1].tag, "Action")
        self.assertEqual(xml[0][0][1].attrib, {"name": "action"})

        tree = et.parse(tmp.name)
        xml_read = tree.getroot()

        self.assertEqual(len(xml_read), 1)
        self.assertEqual(xml_read.tag, "root")
        self.assertEqual(xml_read.attrib, {"main_tree_to_execute": "MainTree"})
        self.assertEqual(len(xml_read[0]), 1)
        self.assertEqual(xml_read[0].tag, "BehaviorTree")
        self.assertEqual(xml_read[0].attrib, {"ID": "MainTree"})
        self.assertEqual(len(xml_read[0][0]), 2)
        self.assertEqual(xml_read[0][0].tag, "Fallback")
        self.assertEqual(xml_read[0][0].attrib, {})
        self.assertEqual(len(xml_read[0][0][0]), 0)
        self.assertEqual(xml_read[0][0][0].tag, "Condition")
        self.assertEqual(xml_read[0][0][0].attrib, {"name": "condition"})
        self.assertEqual(len(xml_read[0][0][1]), 0)
        self.assertEqual(xml_read[0][0][1].tag, "Action")
        self.assertEqual(xml_read[0][0][1].attrib, {"name": "action"})


if __name__ == "__main__":
    unittest.main()
