#!/usr/bin/env python3
""" Class for deterministic memoryless policies.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor, BoolStateFactor, StateFactor
from refine_plan.models.state import State
import yaml


class Policy(object):
    """Data class for deterministic memoryless policies.

    Attributes:
        _state_action_dict: A dictionary from states to actions
        _value_dict: A dictionary from states to values under that policy
    """

    def __init__(self, state_action_dict, value_dict=None, policy_file=None):
        """Initialise attributes.

        Args:
            state_action_dict: The state action mapping
            value_dict: Optional. A state value mapping
            policy_file: Optional. A policy file. Ignores other args if set.
        """
        if policy_file is not None:
            self._read_policy(policy_file)
        else:
            self._state_action_dict = state_action_dict
            self._value_dict = value_dict

    def get_action(self, state):
        """Return the policy action for a given state.

        Args:
            state: The state we want an action for

        Returns:
            The policy action
        """
        if state not in self._state_action_dict:
            return None
        return self._state_action_dict[state]

    def get_value(self, state):
        """Return the value at a given state.

        Args:
            state: The state we want to retrieve the value for

        Returns:
            The value at state

        Raises:
            no_value_dict_exception: Raised if there is no value dictionary
        """
        if self._value_dict is None:
            raise Exception("No value dictionary provided to policy")

        if state not in self._value_dict:
            return None

        return self._value_dict[state]

    def __getitem__(self, state):
        """Syntactic sugar for get_action.

        Args:
            state: The state we want an action for

        Returns:
            The policy action
        """
        return self.get_action(state)

    def write_policy(self, out_file):
        """Write a policy with all state factor information to a YAML file.

        Args:
            out_file: The yaml file to write to
        """

        # First write out state factor info
        sf_dict = {}
        example_state = list(self._state_action_dict.keys())[0]
        for sf_name in example_state._sf_dict:
            sf_dict[sf_name] = {}
            sf = example_state._sf_dict[sf_name]
            if isinstance(sf, BoolStateFactor):
                sf_dict[sf_name]["type"] = "BoolStateFactor"
            elif isinstance(sf, IntStateFactor):
                sf_dict[sf_name]["type"] = "IntStateFactor"
                sf_dict[sf_name]["min"] = sf._values[0]
                sf_dict[sf_name]["max"] = sf._values[-1]
            else:
                sf_dict[sf_name]["type"] = "StateFactor"
                sf_dict[sf_name]["values"] = sf._values

        # Now write out the policy
        policy_list = []
        for state in self._state_action_dict:
            state_action = {}
            for sf in state._state_dict:
                state_action[sf] = state[sf]
            state_action["action"] = self.get_action(state)
            if state_action["action"] is None:
                state_action["action"] = "None"
            policy_list.append(state_action)

        # Now write out the value function (if not None)
        value_list = []
        if self._value_dict is not None:
            for state in self._value_dict:
                state_value = {}
                for sf in state._state_dict:
                    state_value[sf] = state[sf]
                state_value["value"] = self.get_value(state)
                value_list.append(state_value)

        full_dict = {"state_factors": sf_dict, "state_action_map": policy_list}
        if value_list != []:
            full_dict["state_value_map"] = value_list

        with open(out_file, "w") as yaml_out:
            yaml.dump(full_dict, yaml_out)

    def _read_policy(self, in_file):
        """Read in a policy from file.

        Args:
            in_file: The path to the policy YAML file
        """

        with open(in_file, "r") as yaml_in:
            policy_yaml = yaml.load(yaml_in, Loader=yaml.FullLoader)

        # Get all state factor info
        sf_dict = {}
        for sf_name in policy_yaml["state_factors"]:
            sf_info = policy_yaml["state_factors"][sf_name]
            if sf_info["type"] == "BoolStateFactor":
                sf_dict[sf_name] = BoolStateFactor(sf_name)
            elif sf_info["type"] == "IntStateFactor":
                sf_dict[sf_name] = IntStateFactor(
                    sf_name, sf_info["min"], sf_info["max"]
                )
            else:
                sf_dict[sf_name] = StateFactor(sf_name, sf_info["values"])

        # Get the policy mapping
        self._state_action_dict = {}
        for state_action_pair in policy_yaml["state_action_map"]:
            state_dict = {}
            action = None
            for key in state_action_pair:
                if key == "action":
                    if state_action_pair[key] != "None":
                        action = state_action_pair[key]
                else:
                    state_dict[sf_dict[key]] = state_action_pair[key]
            self._state_action_dict[State(state_dict)] = action

        # Get the value function
        self._value_dict = None
        if "state_value_map" in policy_yaml:
            self._value_dict = {}
            for state_action_pair in policy_yaml["state_value_map"]:
                state_dict = {}
                value = None
                for key in state_action_pair:
                    if key == "value":
                        value = state_action_pair[key]
                    elif key != "action":
                        state_dict[sf_dict[key]] = state_action_pair[key]
                self._value_dict[State(state_dict)] = value
