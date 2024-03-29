#!/usr/bin/env python3
""" A very lightweight set of classes for BTs.

This is effectively a BT data structure which allows us to write out to 
a BT.cpp XML file recursively. I don't want the heavyweight baggage of pytree etc.

Author: Charlie Street
Owner: Charlie Street
"""

import xml.etree.ElementTree as et


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
            xml: The BT.cpp XML for the action node
        """
        # TODO: Make this actually work properly with BT.cpp
        return et.Element("Action", name=self.get_name())

    def tick_at_state(self, state):
        """Computes node return value if ticked in a given state.

        This function ignores any node memory, and should only be used
        to check correspondence with a Markov policy.

        Args:
            state: The state to check

        Returns:
            tick_return: The return value of this node being ticked (action name)
        """
        return self.get_name()

    def __repr__(self):
        """Printable version of node.

        Returns:
            repr: Printable version of node
        """
        return "Action({})".format(self.get_name())

    def __str__(self):
        """String version of node.

        Returns:
            str: String version of node
        """
        return "Action({})".format(self.get_name())


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
            xml: The BT.cpp XML for the condition node
        """
        # TODO: Make this actually work with BT.cpp
        return et.Element("Condition", name=self.get_name())

    def tick_at_state(self, state):
        """Computes node return value if ticked in a given state.

        This function ignores any node memory, and should only be used
        to check correspondence with a Markov policy.

        Args:
            state: The state to check

        Returns:
            tick_return: The return value of this node being ticked (True or False)
        """
        return self.get_cond().is_satisfied(state)

    def __repr__(self):
        """Printable version of node.

        Returns:
            repr: Printable version of node
        """
        return "Condition({}; {})".format(self.get_name(), self.get_cond())

    def __str__(self):
        """String version of node.

        Returns:
            str: String version of node
        """
        return "Condition({}; {})".format(self.get_name(), self.get_cond())


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
            xml: The BT.cpp XML for the sequence node
        """
        sequence = et.Element("Sequence")
        for child in self._children:
            sequence.append(child.to_BT_XML())
        return sequence

    def tick_at_state(self, state):
        """Computes node return value if ticked in a given state.

        This function ignores any node memory, and should only be used
        to check correspondence with a Markov policy.

        Args:
            state: The state to check

        Returns:
            tick_return: The return value of this node being ticked
        """
        for child in self._children:
            child_return = child.tick_at_state(state)

            if isinstance(child_return, str):  # Action (equivalent to RUNNING)
                return child_return
            elif not child_return:  # returns False
                return False

        return True

    def __repr__(self):
        """Printable version of node.

        Returns:
            repr: Printable version of node
        """
        return "Sequence({})".format(", ".join([repr(c) for c in self._children]))

    def __str__(self):
        """String version of node.

        Returns:
            str: String version of node
        """
        return "Sequence({})".format(", ".join([str(c) for c in self._children]))


class FallbackNode(CompositeNode):
    """Subclass of CompositeNode for fallback nodes.

    Attributes:
        Same as superclass.
    """

    def to_BT_XML(self):
        """Generates the BT.cpp XML string for the fallback node.

        Returns:
            xml: The BT.cpp XML for the fallback node
        """
        fallback = et.Element("Fallback")
        for child in self._children:
            fallback.append(child.to_BT_XML())
        return fallback

    def tick_at_state(self, state):
        """Computes node return value if ticked in a given state.

        This function ignores any node memory, and should only be used
        to check correspondence with a Markov policy.

        This is because it won't work if we have two actions in sequence...
        The convert_policy functionality ensures this will never happen.

        Args:
            state: The state to check

        Returns:
            tick_return: The return value of this node being ticked
        """
        for child in self._children:
            child_return = child.tick_at_state(state)

            if isinstance(child_return, str):  # Action (equivalent to RUNNING)
                return child_return
            elif child_return:  # returns True
                return True

        return False

    def __repr__(self):
        """Printable version of node.

        Returns:
            repr: Printable version of node
        """
        return "Fallback({})".format(", ".join([repr(c) for c in self._children]))

    def __str__(self):
        """String version of node.

        Returns:
            str: String version of node
        """
        return "Fallback({})".format(", ".join([str(c) for c in self._children]))


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
            xml: The BT.cpp XML
        """
        # The first couple of tags in any BT.cpp file
        root = et.Element("root", main_tree_to_execute="MainTree")
        main_tree = et.SubElement(root, "BehaviorTree", ID="MainTree")
        # Actually add in the nodes proper
        main_tree.append(self._root_node.to_BT_XML())

        xml = et.ElementTree(root)
        # Add indentations to XML file so its more readable!
        et.indent(xml, space="\t", level=0)
        if out_file is not None:
            xml.write(out_file)

        return xml

    def tick_at_state(self, state):
        """Computes node return value if ticked in a given state.

        This function ignores any node memory, and should only be used
        to check correspondence with a Markov policy.

        This is because it won't work if we have two actions in sequence in the fallback.
        The convert_policy functionality ensures this will never happen.

        Args:
            state: The state to check

        Returns:
            tick_return: The return value of this node being ticked
        """
        return self.get_root_node().tick_at_state(state)

    def __repr__(self):
        """Printable version of tree.

        Returns:
            repr: Printable version of tree
        """
        return repr(self.get_root_node())

    def __str__(self):
        """String version of tree.

        Returns:
            str: String version of tree
        """
        return str(self.get_root_node())
