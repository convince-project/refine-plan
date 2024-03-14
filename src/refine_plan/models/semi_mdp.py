#!/usr/bin/env python3
""" Class for semi-MDP based on options.

Author: Charlie Street
Owner: Charlie Street
"""


class SemiMDP(object):
    """Class for semi-MDPs where actions=options.

    Attributes:
        _state_factors: A dictionary from state factor name to state factor
        _options: A dictionary from option name to options
        _props: A list of Propoisitons
        _initial_state: A deterministic initial state (if there is one)
    """

    def __init__(self, sf_list, option_list, props, initial_state=None):
        """Initialise attributes.

        Args:
            sf_list: A list of state factors
            option_list: A list of Options
            props: A list of propositions
            initial_state: Optional. An initial state of the semi-MDP
        """

        self._state_factors = {sf.get_name(): sf for sf in sf_list}
        self._options = {opt.get_name() for opt in option_list}
        self._props = props
        self._initial_state = initial_state

    def get_det_initial_state(self):
        """Return the deterministic initial state, if there is one.

        Returns:
            init_state: The initial state if there is one, None otherwise
        """
        return self._initial_state

    def get_props(self):
        """Return the list of propositions.

        Returns:
            props: The list of propositions
        """
        return self._props

    def get_transition_prob(self, state, option, next_state):
        """Get the transition probability for executing an option in a state.

        Args:
            state: The current state
            option: The name of the option (action) being executed
            next_state: The successor state

        Returns:
            transition_prob: The semi-MDP transition probability

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
            reward: The reward for that option in the semi-MDP

        Raises:
            invalid_opt_exception: Raised if an invalid option is passed in
        """
        if option not in self._options:
            raise Exception("{} is an invalid option".format(option.get_name()))

        return self._options[option].get_reward(state)
