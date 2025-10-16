#!/usr/bin/env python3
"""State class for Markov models.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import AndCondition, EqCondition, AddCondition


class State(object):
    """State class for Markov models.

    Attributes:
        _state_dict: A dictionary from state factor name to value
        _sf_dict: A dictionary from state factor name to state factor
        _hash_val: The hash of the state
    """

    def __init__(self, sf_value_dict):
        """Initialise attributes.

        Args:
            sf_value_dict: A dictionary from state factor to value in that state

        Raises:
            Invalid_value_exception: Raised if an invalid value is given for a state factor.
        """

        self._state_dict = {}
        self._sf_dict = {}
        self._hash_val = None

        for sf in sf_value_dict:
            self._sf_dict[sf.get_name()] = sf
            if not sf.is_valid_value(sf_value_dict[sf]):
                raise Exception("Invalid state factor value provided for state.")
            self._state_dict[sf.get_name()] = sf_value_dict[sf]

    def __getitem__(self, sf_name):
        """Overwrite [] for state value access.

        Args:
            sf_name: The state factor to check the value for

        Returns:
            The value for the state factor in this state

        Raises:
            bad_sf_exception: Raised if an invalid state factor is provided
        """
        if sf_name not in self._state_dict:
            raise Exception("{} not in state".format(sf_name))
        return self._state_dict[sf_name]

    def __hash__(self):
        """Overwrite so states with the same info have the same hash.

        Returns:
            The state's hash
        """
        if self._hash_val is None:
            self._hash_val = hash(tuple(sorted(tuple(self._state_dict.items()))))

        return self._hash_val

    def __repr__(self):
        """Overwrite for printing purposes.

        Returns:
            The representation of this object
        """
        internal = ""
        for sf in self._state_dict:
            internal += "{}: {}, ".format(sf, self._state_dict[sf])

        return "State(" + internal[:-2] + ")"

    def __str__(self):
        """Overwrite for printing purposes.

        Returns:
            A string representation of this object
        """
        return repr(self)

    def __contains__(self, sf):
        """Check if a certain state factor name is present in a state.

        Args:
            sf: The state factor to check

        Returns:
            True if sf in state, False otherwise
        """
        return sf in self._state_dict and sf in self._sf_dict

    def __eq__(self, other):
        """Overwrite as I've overwritten hash.

        Args:
            other: The state to compare against

        Returns:
            True if equal, False otherwise
        """
        if len(self._state_dict) != len(other._state_dict):
            return False

        for sf in self._state_dict:
            if sf not in other:
                return False
            if self[sf] != other[sf]:
                return False

        return True

    def to_and_cond(self):
        """Converts a state into a conjunction of conditions.

        Returns:
            The AddCondition representing the state
        """
        cond = AndCondition()
        for sf in self._sf_dict:
            cond.add_cond(EqCondition(self._sf_dict[sf], self._state_dict[sf]))

        return cond

    def _get_applied_state_dict(self, state_dict, cond):
        """Build up the state dictionary after applying a post condition.

        Args:
            state_dict: The current applied state dictionary
            cond: The condition being applied
        """

        if isinstance(cond, EqCondition):
            state_dict[cond._sf] = cond._value
        elif isinstance(cond, AddCondition):
            state_dict[cond._sf] = (
                self._state_dict[cond._sf.get_name()] + cond._inc_value
            )
        elif isinstance(cond, AndCondition):
            for sub_cond in cond._cond_list:
                self._get_applied_state_dict(state_dict, sub_cond)

    def apply_post_cond(self, cond):
        """Apply a postcondition to produce a new state.

        Args:
            cond: The postcondition to apply

        Returns:
            The new state
        """

        assert cond.is_post_cond()

        state_dict = {}
        self._get_applied_state_dict(state_dict, cond)

        for sf in self._sf_dict:
            sf_obj = self._sf_dict[sf]
            if sf_obj not in state_dict:
                state_dict[sf_obj] = self._state_dict[sf]

        return State(state_dict)
