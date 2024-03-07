#!/usr/bin/env python3
""" Classes for PRISM conditions and state labels.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor


class Label(object):
    """A PRISM label is a condition with a name.

    Attributes:
        _name: The label name
        _cond: The corresponding condition
    """

    def __init__(self, name, cond):
        """Initialise attributes.

        Args:
            name: The label name
            cond: The corresponding condition

        Raises:
            invalid_pre_cond: Raised if cond is not a precondition
        """
        if not cond.is_pre_cond():
            raise Exception("Label requires a valid pre-condition")

        self._name = name
        self._cond = cond

    def to_prism_string(self):
        """Converts the label into a PRISM string.

        Returns:
            prism_str: The corresponding PRISM string.
        """
        return 'label "{}" = {};\n'.format(
            self._name, self._cond.to_prism_string(is_post_cond=False)
        )


class Condition(object):
    """Base class for conditions.

    Condition objects directly correspond to conditions in PRISM.
    """

    def is_satisfied(self, state, prev_state=None):
        """Checks if a state satisfies a given condition.

        Args:
            state: The state to check
            prev_state: The previous state (necessary for some post conditions)

        Returns:
            is_satisfied: Is the condition satisfied?
        """
        raise NotImplementedError()

    def is_pre_cond(self):
        """Returns True if condition can be used as a precondition.

        Returns:
            is_pre_cond: Can the condition be used as a precondition?
        """
        raise NotImplementedError()

    def is_post_cond(self):
        """Returns True if condition can be used as a postcondition.

        Returns:
            is_post_cond: Can the condition be used as a postcondition?
        """
        raise NotImplementedError()

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        raise NotImplementedError()


class EqCondition(Condition):
    """A condition which checks for equality.

    Attributes:
        _sf: The state factor we're checking
        _value: The value to check against
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check

        Raises:
            invalid_value: Raised if value is invalid for sf
        """

        if not sf.is_valid_value(value):
            raise Exception("EqCondition: value is an invalid value for state factor")

        self._sf = sf
        self._value = value

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state matches _value.

        Args:
            state: The state to check
            prev_state: Not used here

        Returns:
            is_satisfied: Is the condition satisfied?

        Raises:
            invalid_value: Raised if state has an invalid value for _sf
        """

        sf_name = self._sf.get_name()
        if not self._sf.is_valid_value(state[sf_name]):
            raise Exception(
                "EqCondition: state has an invalid value for {}".format(sf_name)
            )

        return state[sf_name] == self._value

    def is_pre_cond(self):
        """EqConditions are valid preconditions.
        Returns:
            is_pre_cond: Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """EqConditions are valid postconditions.

        Returns:
            is_post_cond: Can the condition be used as a postcondition?
        """
        return True

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        post_cond_part = "'" if is_post_cond else ""
        return "({}{} = {})".format(self._sf.get_name(), post_cond_part, self._value)


class AddCondition(Condition):
    """Condition for adding a value to a state factor.

    Only valid for IntStateFactors.

    Attributes:
        Same as superclass, plus:
        _sf: The state factor
        _inc_value: The increment value
    """

    def __init__(self, sf, inc_value):
        """Initialise attributes.

        Args:
            sf: The state factor
            inc_value: The increment value
        """
        if not isinstance(sf, IntStateFactor):
            raise Exception("AddCondition: sf must be IntStateFactor")

        self._sf = sf
        self._inc_value = inc_value

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state = prev_state + _inc_value

        Args:
            state: The state to check
            prev_state: The previous state

        Returns:
            is_satisfied: Is the condition satisfied?

        Raises:
            invalid_value: Raised if the incremented value is out of bounds
            no_prev_state: Raised if prev_state not specified
        """

        if prev_state is None:
            raise Exception("AddCondition: No previous state specified")

        sf_name = self._sf.get_name()
        if not self._sf.is_valid_value(state[sf_name]):
            raise Exception(
                "AddCondition: state has an invalid value for {}".format(sf_name)
            )

        if not self._sf.is_valid_value(prev_state[sf_name]):
            raise Exception(
                "AddCondition: prev_state has an invalid value for {}".format(sf_name)
            )

        computed_val = prev_state[sf_name] + self._inc_value

        if not self._sf.is_valid_value(computed_val):
            raise Exception(
                "AddCondition: prev_state + inc_value has an invalid value for {}".format(
                    sf_name
                )
            )

        return state[sf_name] == computed_val

    def is_pre_cond(self):
        """AddConditions are not valid preconditions.
        Returns:
            is_pre_cond: Can the condition be used as a precondition?
        """
        return False

    def is_post_cond(self):
        """AdsConditions are valid postconditions.

        Returns:
            is_post_cond: Can the condition be used as a postcondition?
        """
        return True

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        if not is_post_cond:
            raise Exception("AddCondition can only output as a postcondition")

        return "({}' = {} + {})".format(
            self._sf.get_name(), self._sf.get_name(), self._value
        )
