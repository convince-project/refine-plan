#!/usr/bin/env python3
"""Class for options, which are temporally extended actions.

Author: Charlie Street
Owner: Charlie Street
"""

import xml.etree.ElementTree as et
import numpy as np


class Option(object):
    """An option is a temporally extended behaviour.

    Options are traditionally defined in terms of an initiation set I,
    a termination condition beta, and a policy pi.

    Here, we just define the option in terms of its transition probabilities
    (assuming no preemption occurs) and reward model.

    Attributes:
        _name: The option's name
        _transition_list: The option's transition model
        _reward_list: The option's reward model
    """

    def __init__(self, name, transition_list, reward_list):
        """Initialise attributes.

        Args:
            name: The option's name
            transition_list: A list of (pre_cond, prob_post_conds), where
                             prob_post_conds is a dictionary from
                             postconditions to probabilities
            reward_list: A list of (pre_cond, reward) pairs
        """

        self._check_valid_probs(transition_list)

        self._name = name
        self._transition_list = transition_list
        self._reward_list = reward_list

    def _check_valid_probs(self, transition_list):
        """Check all probabilistic post conditions sum to one.

        Args:
            transition_list: The list of (pre_cond, prob_post_conds)

        Raises:
            invalid_probs_exception: Raised if invalid distributions are given
        """

        for trans in transition_list:
            prob_post_conds = trans[1]
            total = sum(list(prob_post_conds.values()))
            if not np.isclose(total, 1.0):
                raise Exception(
                    "Invalid transition distribution for option {}".format(
                        self.get_name()
                    )
                )

    def get_name(self):
        """Return the option's name.

        Returns:
            The option's name
        """
        return self._name

    def get_transition_prob(self, state, next_state):
        """Return the transition probability for a (s,s') pair.

        Assumes there is only one precondition which holds for each state.

        Args:
            state: The first state
            next_state: The next state

        Returns:
            The transition probability
        """
        for trans in self._transition_list:
            pre_cond, prob_post_conds = trans
            if pre_cond.is_satisfied(state):
                for post_cond in prob_post_conds:
                    if post_cond.is_satisfied(next_state):
                        return prob_post_conds[post_cond]
                return 0.0
        return 0.0  # If not in initiation set, 0 probably seems a fair return value

    def get_reward(self, state):
        """Return the reward for executing an option in a given state.

        Args:
            state: The state we want to check

        Returns:
            The reward for the state
        """

        total_reward = 0.0
        for reward_pair in self._reward_list:
            pre_cond, reward = reward_pair
            if pre_cond.is_satisfied(state):
                total_reward += reward

        return total_reward

    def _add_datamodel_update_scxml(self, sf_names, policy_name):
        """Add a generic SCXML block for sending datamodel updates.

        Args:
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Return:
            Datamodel update SCXML block
        """
        event = et.Element("send", event="update_datamodel", target=policy_name)
        for sf in sf_names:
            event.append(et.Element("param", name=sf, expr=sf))

        return event

    def _build_single_scxml_transition(
        self, pre_cond, prob_post_conds, sf_names, policy_name
    ):
        """Build a single SCXML transition given pre and post conds.

        Args:
            pre_cond: A precondition
            prob_post_conds: A dictionary from post condition to probability
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Assumes len(prob_post_conds) >= 1

        Return:
            An SCXML transition element
        """
        scxml_trans = et.Element(
            "transition",
            target="init",
            event=self.get_name(),
            cond=pre_cond.to_scxml_cond(False),
        )

        if len(prob_post_conds) == 1:  # If deterministic
            post_cond = list(prob_post_conds.keys())[0]
            for cond in post_cond.to_scxml_cond(is_post_cond=True):
                scxml_trans.append(cond)
        else:  # Do if else structure
            rand = et.Element("assign", location="rand", expr="Math.random()")
            scxml_trans.append(rand)
            prob_sum = 0.0
            if_block = None
            for post_cond in prob_post_conds:
                prob_sum += prob_post_conds[post_cond]
                cond_str = "rand <= {}".format(prob_sum)
                if if_block is None:
                    if_block = et.Element("if", cond=cond_str)
                elif np.isclose(prob_sum, 1):
                    if_block.append(et.Element("else"))
                else:
                    if_block.append(et.Element("elseif", cond=cond_str))

                for cond in post_cond.to_scxml_cond(is_post_cond=True):
                    if_block.append(cond)
            scxml_trans.append(if_block)

        scxml_trans.append(self._add_datamodel_update_scxml(sf_names, policy_name))

        return scxml_trans

    def get_scxml_transitions(self, sf_names, policy_name):
        """Return a list of SCXML transition elements for this option.

        Args:
            sf_names: The list of state factor names
            policy_name: The name of the policy in SCXML

        Returns:
            A list of SCXML transition elements
        """
        transitions = []

        for trans in self._transition_list:
            pre_cond, prob_post_conds = trans
            scxml_trans = self._build_single_scxml_transition(
                pre_cond, prob_post_conds, sf_names, policy_name
            )
            transitions.append(scxml_trans)

        return transitions

    def get_transition_prism_string(self):
        """Return a PRISM string which captures all transitions for this option.

        Returns:
            The transition PRISM string
        """
        prism_str = ""
        for trans in self._transition_list:
            pre_cond, prob_post_conds = trans
            prism_str += "[{}] {} -> ".format(
                self.get_name(), pre_cond.to_prism_string()
            )

            for post_cond in prob_post_conds:
                prism_str += "{}:{} + ".format(
                    prob_post_conds[post_cond],
                    post_cond.to_prism_string(is_post_cond=True),
                )

            prism_str = prism_str[:-3] + ";\n"  # Remove final " + "

        return prism_str

    def get_reward_prism_string(self):
        """Return a PRISM string which captures all rewards for this option.

        Returns:
            The reward PRISM string
        """
        prism_str = ""
        for reward_pair in self._reward_list:
            pre_cond, reward = reward_pair

            prism_str += "[{}] {}: {};\n".format(
                self.get_name(), pre_cond.to_prism_string(), reward
            )

        return prism_str
