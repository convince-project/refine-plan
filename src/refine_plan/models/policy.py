#!/usr/bin/env python3
""" Class for deterministic memoryless policies.

Author: Charlie Street
Owner: Charlie Street
"""


class Policy(object):
    """Data class for deterministic memoryless policies.

    Attributes:
        _state_action_dict: A dictionary from states to actions
        _value_dict: A dictionary from states to values under that policy
    """

    def __init__(self, state_action_dict, value_dict=None):
        """Initialise attributes.

        Args:
            state_action_dict: The state action mapping
            value_dict: Optional. A state value mapping
        """
        self._state_action_dict = state_action_dict
        self._value_dict = value_dict

    def get_action(self, state):
        """Return the policy action for a given state.

        Args:
            state: The state we want an action for

        Returns:
            action: The policy action
        """
        if state not in self._state_action_dict:
            return None
        return self._state_action_dict[state]

    def get_value(self, state):
        """Return the value at a given state.

        Args:
            state: The state we want to retrieve the value for

        Returns:
            value: The value at state

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
            action: The policy action
        """
        return self.get_action(state)
