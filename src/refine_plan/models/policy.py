#!/usr/bin/env python3
""" Class for deterministic memoryless policies.

Author: Charlie Street
Owner: Charlie Street
"""


class Policy(object):
    """Data class for deterministic memoryless policies.

    Attributes:
        _state_action_dict: A dictionary from states to actions
    """

    def __init__(self, state_action_dict):
        """Initialise attributes.

        Args:
            state_action_dict: The state action mapping
        """
        self._state_action_dict = state_action_dict

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

    def __getitem__(self, state):
        """Syntactic sugar for get_action.

        Args:
            state: The state we want an action for

        Returns:
            action: The policy action
        """
        return self.get_action(state)
