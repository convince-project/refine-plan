#!/usr/bin/env python3
"""Class for semi-MDP based on options.

Author: Charlie Street
Owner: Charlie Street
"""

import xml.etree.ElementTree as et
from datetime import datetime


class SemiMDP(object):
    """Class for semi-MDPs where actions=options.

    Attributes:
        _state_factors: A dictionary from state factor name to state factor
        _options: A dictionary from option name to options
        _labels: A list of Labels
        _initial_state: A deterministic initial state (if there is one)
    """

    def __init__(self, sf_list, option_list, labels, initial_state=None):
        """Initialise attributes.

        Args:
            sf_list: A list of state factors
            option_list: A list of Options
            labels: A list of Labels
            initial_state: Optional. An initial state of the semi-MDP
        """

        self._state_factors = {sf.get_name(): sf for sf in sf_list}
        self._options = {opt.get_name(): opt for opt in option_list}
        self._labels = labels
        self._initial_state = initial_state

    def get_state_factors(self):
        """Return the state factor dictionary.

        Returns:
            A dictionary from state factor name to state factor
        """
        return self._state_factors

    def get_det_initial_state(self):
        """Return the deterministic initial state, if there is one.

        Returns:
            The initial state if there is one, None otherwise
        """
        return self._initial_state

    def get_labels(self):
        """Return the list of labels.

        Returns:
            The list of labels
        """
        return self._labels

    def get_transition_prob(self, state, option, next_state):
        """Get the transition probability for executing an option in a state.

        Args:
            state: The current state
            option: The name of the option (action) being executed
            next_state: The successor state

        Returns:
            The semi-MDP transition probability

        Raises:
            invalid_opt_exception: Raised if an invalid option is passed in
        """
        if option not in self._options:
            raise Exception("{} is an invalid option".format(option.get_name()))

        return self._options[option].get_transition_prob(state, next_state)

    def get_reward(self, state, option):
        """Get the reward for executing an option in a state.

        Args:
            state: The current state
            option: The name of the option (action) being executed

        Returns:
            The reward for that option in the semi-MDP

        Raises:
            invalid_opt_exception: Raised if an invalid option is passed in
        """
        if option not in self._options:
            raise Exception("{} is an invalid option".format(option.get_name()))

        return self._options[option].get_reward(state)

    def to_scxml_file(self, output_file, policy_name, name="mdp"):
        """Convert the semi-MDP into an SCXML file for use with SCAN.

        Args:
            output_file: The file path for the SCXML file
            policy_name: The SCXML name for the policy (needed for comms)
            name: Optional. The name of the MDP
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
        data_model = et.SubElement(scxml, "datamodel")

        # For the SCXML file our semi-MDP needs an initial state
        assert self._initial_state is not None
        for sf in self._state_factors:
            data_model.append(
                self._state_factors[sf].to_scxml_element(self._initial_state[sf])
            )

        # Add rand state factor
        data_model.append(et.Element("data", id="rand", expr="0", type="float64"))

        # Add transitions
        state = et.SubElement(scxml, "state", id="init")
        sf_names = list(self._state_factors.keys())
        for option in self._options:
            for trans in self._options[option].get_scxml_transitions(
                sf_names, policy_name
            ):
                state.append(trans)

        # Now deal with the writing out
        xml = et.ElementTree(scxml)
        et.indent(xml, space="\t", level=0)  # Indent to improve readability
        xml.write(output_file, encoding="UTF-8", xml_declaration=True)

    def to_prism_string(self, output_file=None):
        """Convert the semi-MDP into a PRISM model.

        Because of the objectives we are handling this is written as an MDP.

        Args:
            output_file: Optional. A file path to write the PRISM string to.

        Returns:
            The PRISM string
        """

        # Opening comments
        prism_str = "// Auto-generated semi-MDP for REFINE-PLAN\n"

        now = datetime.now()
        prism_str += "// Date generated: {}/{}/{}\n\n".format(
            now.day, now.month, now.year
        )

        # MDP declaration
        prism_str += "mdp\n\nmodule semimdp\n\n"

        # State factors
        for sf in self._state_factors:
            prism_str += self._state_factors[sf].to_prism_string(
                self._initial_state[sf]
                if self._initial_state is not None and sf in self._initial_state
                else None
            )

        prism_str += "\n"

        # Transitions
        for option in self._options:
            prism_str += self._options[option].get_transition_prism_string()
        prism_str += "\nendmodule\n\n"

        # Write labels
        for label in self._labels:
            prism_str += label.to_prism_string()

        # Write rewards
        prism_str += "\nrewards\n"
        for option in self._options:
            prism_str += self._options[option].get_reward_prism_string()
        prism_str += "endrewards\n"

        # All states can be initial states
        if self._initial_state is None:
            prism_str += "\ninit true endinit\n"

        if output_file is not None:
            with open(output_file, "w") as out_nm:
                out_nm.write(prism_str)

        return prism_str
