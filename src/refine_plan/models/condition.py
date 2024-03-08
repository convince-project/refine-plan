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

    def __repr__(self):
        """Make the label human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the label human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()


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

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())

    def __str__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())


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

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(True)

    def __str__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(True)


class CompareCondition(Condition):
    """A precondition which compares a state factor to a value.

    EqCondition is not included here, as it can be a postcondition.
    This class can (and will) be used for <, >, <=, >=

    Attributes:
        _sf: The state factor we're checking
        _comp_fn: The int x int -> bool comparison function
        _comp_str: The PRISM symbol for the comparison operation
        _value: The value to check against
    """

    def __init__(self, sf, value, comp_fn, comp_str):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check
            comp_fn: The int x int -> bool comparison function
            comp_str: The PRISM symbol for the comparison operation

        Raises:
            invalid_value: Raised if value is invalid for sf
        """

        if not sf.is_valid_value(value):
            raise Exception("EqCondition: value is an invalid value for state factor")

        self._sf = sf
        self._value = value
        self._comp_fn = comp_fn
        self._comp_str = comp_str

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

        return self._comp_fn(state[sf_name], self._value)

    def is_pre_cond(self):
        """CompareConditions are valid preconditions.
        Returns:
            is_pre_cond: Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """CompareConditions are valid postconditions.

        Returns:
            is_post_cond: Can the condition be used as a postcondition?
        """
        return True

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?

        Raises:
            post_cond_exception: Raised if is_post_cond is True
        """

        if is_post_cond:
            raise Exception("CompareCondition cannot be postcondition.")

        return "({} {} {})".format(self._sf.get_name(), self._comp_str, self._value)

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()


class LtCondition(CompareCondition):
    """A precondition for <.

    Attributes:
        Same as superclass.
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check
        """
        super(LtCondition, self).__init__(sf, value, lambda x, y: x < y, "<")


class GtCondition(CompareCondition):
    """A precondition for >.

    Attributes:
        Same as superclass.
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check
        """
        super(GtCondition, self).__init__(sf, value, lambda x, y: x > y, ">")


class LeqCondition(CompareCondition):
    """A precondition for <=.

    Attributes:
        Same as superclass.
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check
        """
        super(LeqCondition, self).__init__(sf, value, lambda x, y: x <= y, "<=")


class GeqCondition(CompareCondition):
    """A precondition for >=.

    Attributes:
        Same as superclass.
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check
        """
        super(LeqCondition, self).__init__(sf, value, lambda x, y: x >= y, ">=")


class AndCondition(Condition):
    """Composite condition which captures conjunctions.

    Attributes:
        _cond_list: A list of conditions
    """

    def __init__(self, *conds):
        """Initialise attributes.

        Args:
            conds: The conditions to combine
        """
        self._cond_list = conds

    def add_cond(self, cond):
        """Add a new condition to the conjunction.

        Args:
            cond: The new condition
        """
        self._cond_list.append(cond)

    def is_satisfied(self, state, prev_state=None):
        """Check if conjunction is satisfied.

        Args:
            state: The current state to check
            prev_state: The previous statem (if needed)

        Returns:
            is_satisfied: True if condition satisfied, else False
        """

        for cond in self._cond_list:
            if not cond.is_satisfied(state, prev_state):
                return False
        return True

    def is_pre_cond(self):
        """Conjunction is pre cond if all conditions are preconditions.

        Returns:
            is_pre_cond: True if all conditions are preconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_pre_cond():
                return False

        return True

    def is_post_cond(self):
        """Conjunction is post cond if all conditions are postconditions.

        Returns:
            is_post_cond: True if all conditions are postconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_post_cond():
                return False

        return True

    def to_prism_string(self, is_post_cond=False):
        """Output condition into prism string format.

        Args:
            is_post_cond: Is the condition a post condition?"""
        prism_str = "("

        for i in range(len(self._cond_list)):
            prism_str += self._cond_list[i].to_prism_string(is_post_cond)
            if i < len(self._cond_list) - 1:
                prism_str += " & "

        prism_str += ")"

        return prism_str

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())

    def __str__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())


class OrCondition(Condition):
    """Composite condition which captures disjunctions.

    Attributes:
        _cond_list: A list of conditions
    """

    def __init__(self, *conds):
        """Initialise attributes.

        Args:
            conds: The conditions to combine
        """
        self._cond_list = conds

    def add_cond(self, cond):
        """Add a new condition to the conjunction.

        Args:
            cond: The new condition
        """
        self._cond_list.append(cond)

    def is_satisfied(self, state, prev_state=None):
        """Check if disjunction is satisfied.

        Args:
            state: The current state to check
            prev_state: The previous statem (if needed)

        Returns:
            is_satisfied: True if condition satisfied, else False
        """

        for cond in self._cond_list:
            if cond.is_satisfied(state, prev_state):
                return True
        return False

    def is_pre_cond(self):
        """Disjunction is pre cond if all conditions are preconditions.

        Returns:
            is_pre_cond: True if all conditions are preconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_pre_cond():
                return False

        return True

    def is_post_cond(self):
        """Disjunction cannot be a postcondition.

        Returns:
            is_post_cond: False for disjunctions
        """
        return False

    def to_prism_string(self, is_post_cond=False):
        """Output condition into prism string format.

        Args:
            is_post_cond: Is the condition a post condition?"""

        if self.is_post_cond():
            raise Exception("OrCondition cannot be a postcondition.")

        prism_str = "("

        for i in range(len(self._cond_list)):
            prism_str += self._cond_list[i].to_prism_string(is_post_cond)
            if i < len(self._cond_list) - 1:
                prism_str += " | "

        prism_str += ")"

        return prism_str

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            label: A str representation of the label
        """
        return self.to_prism_string()
