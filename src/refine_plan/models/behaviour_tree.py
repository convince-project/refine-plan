#!/usr/bin/env python3
""" A very lightweight set of classes for BTs.

This is effectively a BT data structure which allows us to write out to 
a BT.cpp XML file recursively. I don't want the heavyweight baggage of pytree etc.

Author: Charlie Street
Owner: Charlie Street
"""


class BTNode(object):
    """Base class for individual BT nodes."""

    def to_BT_XML(self):
        """Outputs a string containing the BT.cpp XML for this node.

        Returns:
            xml_string: The BT XML for this node as a string
        """
        raise NotImplementedError("to_BT_XML not implemented in BTNode abstract class")


class ActionNode(BTNode):
    """Node class for BT action nodes.

    Attributes:
        _name: The name of the action being executed.
    """

    def __init__(self, name):
        """Initialise attributes.

        Args:
            name: The name of the action
        """
        self._name = name

    def get_name(self):
        """Getter for self._name.

        Returns:
            name: The action name
        """
        return self._name

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the action node.

        Returns:
            xml_string: The BT.cpp XML string for the action node
        """
        raise NotImplementedError("to_BT_XML not implemented in ActionNode.")


class ConditionNode(BTNode):
    """Node class for BT condition nodes.

    Attributes:
        _name: The name of the condition node
        _cond: The condition which the node checks
    """

    def __init__(self, name, cond):
        """Initialise attributes.

        Args:
            name: The condition node name
            cond: The condition being checked
        """
        self._name = name
        self._cond = cond

    def get_name(self):
        """Getter for self._name.

        Returns:
            name: The condition node name
        """
        return self._name

    def get_cond(self):
        """Getter for self._cond.

        Returns:
            cond: The condition being checked
        """
        return self._cond

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the condition node.

        Returns:
            xml_string: The BT.cpp XML string for the condition node
        """
        raise NotImplementedError("to_BT_XML not implemented in ConditionNode.")


class CompositeNode(BTNode):
    """A class for BT composite nodes (sequence/fallback).

    This class currently represents synchronous composite nodes,
    i.e. composite with memory nodes.

    Attributes:
        _children: The child nodes
    """

    def __init__(self, *children):
        """Initialise attributes.

        Args:
            children: A number of child BTNodes

        Raises:
            invalid_child_exception: Raised if a child is not a BT node
        """
        self._children = []

        for child in children:
            if not isinstance(child, BTNode):
                raise Exception("CompositeNodes can only have BTNode children")
            self._children.append(child)

    def add_child(self, child):
        """Add a new child to the composite node.

        Args:
            child: The new child

        Raises:
            invalid_child_exception: Raised if a child is not a BT node
        """
        if not isinstance(child, BTNode):
            raise Exception("Composite can only have BTNode children")
        self._children.append(child)

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the composite node.

        Returns:
            xml_string: The BT.cpp XML string for the composite node
        """
        raise NotImplementedError("to_BT_XML not implemented in CompositeNode.")


class SequenceNode(CompositeNode):
    """Subclass of CompositeNode for sequence nodes.

    Attributes:
        Same as superclass.
    """

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the sequence node.

        Returns:
            xml_string: The BT.cpp XML string for the sequence node
        """
        raise NotImplementedError("to_BT_XML not implemented in SequenceNode.")


class FallbackNode(CompositeNode):
    """Subclass of CompositeNode for fallback nodes.

    Attributes:
        Same as superclass.
    """

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the fallback node.

        Returns:
            xml_string: The BT.cpp XML string for the fallback node
        """
        raise NotImplementedError("to_BT_XML not implemented in FallbackNode.")


class BehaviourTree(object):
    """Class for a complete behaviour tree.

    This class just enforces a root node and adds the outer layers on the
    BT.cpp XML file.

    Attributes:
        _root_node: A root BTNode
    """

    def __init__(self, root_node):
        """Initialise attributes

        Args:
            root_node: The root node of the BT
        """
        self._root_node = root_node

    def get_root_node(self):
        """Getter for self._root_node.

        Returns:
            root_node: The root node of the BT
        """
        return self._root_node

    def to_BT_XML(self, out_file=None):
        """Produces the BT.cpp XML file for the BT.

        Args:
            out_file: Optional. If specified, will write the BT XML string to file

        Returns:
            xml_string: The BT.cpp XML string
        """
        raise NotImplementedError("to_BT_XML not implemented in BehaviourTree.")