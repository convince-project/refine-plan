#!/usr/bin/env python
""" An alternative policy to BT converter that is simpler but builds larger BTs.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.algorithms.policy_to_bt import PolicyBTConverter
from refine_plan.models.condition import EqCondition
from refine_plan.models.behaviour_tree import (
    BehaviourTree,
    FallbackNode,
    SequenceNode,
    ConditionNode,
    ActionNode,
)
from refine_plan.models.state import State
import copy


class SimplePolicyBTConverter(PolicyBTConverter):
    """This class contains a simple policy->BT converter that does no reduction.

    We effectively just resolve each state factor depth first.
    """

    def __init__(self):
        """Overwriting superclass policy to do nothing."""
        pass

    def _simple_bt_builder(self, state_action_dict, partial_state_dict, sf_list):
        """Build a BT from a policy by filtering by each state factor.

        This function returns the root node of the behaviour tree.

        Args:
            state_action_dict: A partial dictionary from states to actions
            partial_state_dict: A partial state summarising the route down the tree
            sf_list: A list of remaining state factors
        """

        if sf_list == []:
            return ActionNode(state_action_dict[State(partial_state_dict)])

        # Filter states by each value in the first state factor
        sf = sf_list[0]
        filtered_state_action_dict = {}
        for state in state_action_dict:
            if state_action_dict[state] != None:
                sf_val = state[sf.get_name()]
                if sf_val not in filtered_state_action_dict:
                    filtered_state_action_dict[sf_val] = {}
                filtered_state_action_dict[sf_val][state] = state_action_dict[state]

        # Now build a big fallback which checks for each of the state factor values
        root = FallbackNode()
        for sf_val in filtered_state_action_dict:
            sequence = SequenceNode()
            cond_name = "{}EQ{}".format(sf.get_name(), sf_val)
            sequence.add_child(ConditionNode(cond_name, EqCondition(sf, sf_val)))
            new_partial_state_dict = copy.deepcopy(partial_state_dict)
            new_partial_state_dict[sf] = sf_val
            sub_bt = self._simple_bt_builder(
                filtered_state_action_dict[sf_val], new_partial_state_dict, sf_list[1:]
            )
            sequence.add_child(sub_bt)
            root.add_child(sequence)

        # In this case, we can just return the sequence node as the fallback is useless
        if len(root._children) == 1:
            return root._children[0]
        return root

    def convert_policy(self, policy, out_file=None):
        """Convert a Policy into a BehaviourTree and write the BT to file.

        Args:
            policy: A deterministic, memoryless policy
            out_file: Optional. The output files for the BT

        Returns:
            The converted BT
        """

        # Sort in descending order of values in the state factor
        sf_list = list(list(policy._state_action_dict.keys())[0]._sf_dict.values())
        sf_list.sort(key=lambda sf: len(sf._values), reverse=True)
        root_node = self._simple_bt_builder(policy._state_action_dict, {}, sf_list)
        bt = BehaviourTree(root_node)

        # Write the BT to file if desired
        if out_file is not None:
            bt.to_BT_XML(out_file)

        return bt
