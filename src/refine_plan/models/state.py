#!/usr/bin/env python3
""" State class for Markov models. 

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.condition import AndCondition, EqCondition


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
            value: The value for the state factor in this state

        Raises:
            bad_sf_exception: Raised if an invalid state factor is provided
        """
        if sf_name not in self._state_dict:
            raise Exception("{} not in state".format(sf_name))
        return self._state_dict[sf_name]

    def __hash__(self):
        """Overwrite so states with the same info have the same hash.

        Returns:
            hash_val: The state's hash
        """
        if self._hash_val is None:
            self._hash_val = hash(tuple(sorted(tuple(self._state_dict.items()))))

        return self._hash_val

    def to_add_cond(self):
        """Converts a state into a conjunction of conditions.

        Returns:
            cond: The AddCondition representing the state
        """
        cond = AndCondition()
        for sf in self._sf_dict:
            cond.add_cond(EqCondition(self._sf_dict[sf], self._state_dict[sf]))

        return cond
