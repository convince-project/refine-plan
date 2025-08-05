#!/usr/bin/env python3
"""Class for deterministic memoryless policies and time-dependent policies.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor, BoolStateFactor, StateFactor
from refine_plan.models.condition import EqCondition
from refine_plan.models.state import State
import xml.etree.ElementTree as et
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

    def _hierarchical_rep(self):
        """Convert the policy into a hierarchical representation.

        This produces a nested dictionary where each new level captures
        a different state factor.

        They keys in the dictionary levels are SCXML conditions.

        Returns:
            hier_policy: A hierarchical version of the policy
        """

        sf_list = list(list(self._state_action_dict.keys())[0]._sf_dict.values())
        hier_policy = {}

        for state in self._state_action_dict:
            current_hier = hier_policy
            action = self._state_action_dict[state]
            if action is not None:
                for i in range(len(sf_list)):
                    sf = sf_list[i]
                    scxml = EqCondition(sf, state[sf.get_name()]).to_scxml_cond()
                    if scxml not in current_hier:
                        if i == len(sf_list) - 1:  # At the bottom, add the action
                            current_hier[scxml] = action
                        else:
                            current_hier[scxml] = {}
                    current_hier = current_hier[scxml]

        return hier_policy

    def _build_up_nested_scxml(self, hier_policy, model_name):
        """Recursively build up SCXML elements for the policy.

        hier_policy is the nested policy starting from the current depth.

        Args:
            hier_policy: The nested policy from the current depth
            model_name: The name of the model actions are executed on

        Returns:
            scxml_elem: An SCXML element for the hierarchical policy
        """

        if isinstance(hier_policy, str):  # If an action (base case)
            return et.Element("send", event=hier_policy, target=model_name)
        else:  # We need to recurse down the policy
            scxml_elem = None
            conds = list(hier_policy.keys())
            for i in range(len(conds)):
                if scxml_elem is None:
                    scxml_elem = et.Element("if", cond=conds[i])
                else:
                    # Avoid else catch all as it will give action to invalid states
                    scxml_elem.append(et.Element("elseif", cond=conds[i]))
                next_level = hier_policy[conds[i]]
                scxml_elem.append(self._build_up_nested_scxml(next_level, model_name))

            return scxml_elem

    def to_scxml(self, output_file, model_name, initial_state, name="policy"):
        """Write the policy out to SCXML for verification.

        Args:
            output_file: The file to write out to
            model_name: The name of the model actions are executed on
            initial_state: The initial state
            name: The name for the policy in SCXML
        """
        # Root of SCXML file
        scxml = et.Element(
            "scxml",
            initial="init",
            version="1.0",
            name=name,
            model_src="",
            xmlns="http://www.w3.org/2005/07/scxml",
        )

        # Add in the state factors
        sf_list = list(list(self._state_action_dict.keys())[0]._sf_dict.values())
        data_model = et.SubElement(scxml, "datamodel")
        for sf in sf_list:
            data_model.append(sf.to_scxml_element(initial_state[sf.get_name()]))

        state_elem = et.SubElement(scxml, "state", id="init")
        onentry = et.SubElement(state_elem, "onentry")

        hier_policy = self._hierarchical_rep()

        # Recursively build up the nested SCXML if conditions
        onentry.append(self._build_up_nested_scxml(hier_policy, model_name))

        data_transition = et.SubElement(
            state_elem, "transition", target="init", event="update_datamodel"
        )
        for sf in sf_list:
            name = sf.get_name()
            data_transition.append(
                et.Element("assign", location=name, expr="_event.data.{}".format(name))
            )

        # Now handle the writing out
        # Now deal with the writing out
        xml = et.ElementTree(scxml)
        et.indent(xml, space="\t", level=0)  # Indent to improve readability
        xml.write(output_file, encoding="UTF-8", xml_declaration=True)


class TimeDependentPolicy(Policy):
    """Data class for time dependent policies.

    In _state_action_dicts and _value_dicts, the list index refers to the timestep.

    Attributes:
        _state_action_dicts: A list of dictionaries from states to actions
        _value_dicts: A list of dictionaries from states to values under that policy
    """

    def __init__(self, state_action_dicts, value_dicts=None):
        """Initialise attributes.

        Args:
            state_action_dicts: The state action mapping for each timestep
            value_dicts: Optional. A state value mapping for each timestep
        """
        self._state_action_dicts = state_action_dicts
        self._value_dicts = value_dicts

    def get_action(self, state, time):
        """Return the policy action for a given state and time.

        Args:
            state: The state we want an action for
            time: The current timestep

        Returns:
            The policy action
        """
        if time < 0 or time >= len(self._state_action_dicts):
            return None

        if state not in self._state_action_dicts[time]:
            return None

        return self._state_action_dicts[time][state]

    def get_value(self, state, time):
        """Return the value at a given state and time.

        Args:
            state: The state we want to retrieve the value for
            time: The current timestep

        Returns:
            The value at state at time

        Raises:
            no_value_dict_exception: Raised if there is no value dictionary
        """
        if self._value_dicts is None:
            raise Exception("No value dictionaries provided to policy")

        if time < 0 or time >= len(self._value_dicts):
            return None
        if state not in self._value_dicts[time]:
            return None

        return self._value_dicts[time][state]

    def __getitem__(self, state_time):
        """Syntactic sugar for get_action.

        Args:
            state_time: A tuple with a state and a timestep

        Returns:
            The policy action
        """
        return self.get_action(state_time[0], state_time[1])

    def write_policy(self, out_file):
        # TODO: Implement
        raise NotImplementedError()

    def _read_policy(self, in_file):
        # TODO: Implement
        raise NotImplementedError()

    def _hierarchical_rep(self):
        raise NotImplementedError()

    def _build_up_nested_scxml(self, hier_policy, model_name):
        raise NotImplementedError()

    def to_scxml(self, output_file, model_name, initial_state, name="policy"):
        raise NotImplementedError()
