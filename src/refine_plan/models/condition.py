#!/usr/bin/env python3
""" Classes for PRISM conditions and state labels.

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import IntStateFactor
from pyeda.inter import expr, And, Or, Not


class Label(object):
    """A PRISM label is a condition with a name.

    Attributes:
        _name: The label name
        _cond: The corresponding condition
        _hash_val: The cached hash value for the label
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
        self._hash_val = None

    def to_prism_string(self):
        """Converts the label into a PRISM string.

        Returns:
            The corresponding PRISM string
        """
        return 'label "{}" = {};\n'.format(
            self._name, self._cond.to_prism_string(is_post_cond=False)
        )

    def __repr__(self):
        """Make the label human readable.

        Returns:
            A str representation of the label
        """
        return 'label "{}" = {};'.format(
            self._name, self._cond.to_prism_string(is_post_cond=False)
        )

    def __str__(self):
        """Make the label human readable.

        Returns:
            A str representation of the label
        """
        return 'label "{}" = {};'.format(
            self._name, self._cond.to_prism_string(is_post_cond=False)
        )

    def __hash__(self):
        """Overwriting so labels with equal contents have the same hash.

        Returns:
            The hash value of the label
        """
        if self._hash_val is None:
            self._hash_val = hash((self._name, self._cond))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so labels with equal contents are equal.

        Args:
            other: The other label

        Returns:
            Are the two Labels equal?
        """
        return self._name == other._name and self._cond == other._cond


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
            Is the condition satisfied?
        """
        raise NotImplementedError()

    def is_pre_cond(self):
        """Returns True if condition can be used as a precondition.

        Returns:
            Can the condition be used as a precondition?
        """
        raise NotImplementedError()

    def is_post_cond(self):
        """Returns True if condition can be used as a postcondition.

        Returns:
            Can the condition be used as a postcondition?
        """
        raise NotImplementedError()

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        raise NotImplementedError()

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        This allows it to be worked with and minimised etc.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition. Only returned if return_var_map
        """
        raise NotImplementedError()

    def range_of_values(self):
        """Returns the range of values that can be satisfied with this condition.

        Returns:
            A list of dictionaries from state factor to a list of values
        """
        raise NotImplementedError()


class TrueCondition(Condition):
    """A condition which is always true."""

    def __init__(self):
        """No attributes to initialise."""
        super(TrueCondition, self).__init__()

    def is_satisfied(self, state, prev_state=None):
        """Always returns True.

        Args:
            state: Not used
            prev_state: Not used

        Returns:
            Always True
        """
        return True

    def is_pre_cond(self):
        """TrueCondition is a valid precondition

        Returns:
            True
        """
        return True

    def is_post_cond(self):
        """TrueConditions can be used to signify self loops in PRISM

        Returns:
            True for TrueConditions
        """
        return True

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        return "true"

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression (1)

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        return expr(True), {}

    def range_of_values(self):
        """Raises exception as it is impossible to return everything.

        Raises:
            cant_return_everything: Raised as there is no range as its everything
        """
        raise Exception("Can't return range for TrueCondition")

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        # No caching here as its so simple
        return hash(type(self))

    def __eq__(self, other):
        """Overwriting for use in dictionaries etc.

        Args:
            other: The other condition

        Returns:
            Are the two conditions equal?
        """
        return type(self) == type(other)


class EqCondition(Condition):
    """A condition which checks for equality.

    Attributes:
        _sf: The state factor we're checking
        _value: The value to check against
        _hash_val: The cached hash
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
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state matches _value.

        Args:
            state: The state to check
            prev_state: Not used here

        Returns:
            Is the condition satisfied?

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
            Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """EqConditions are valid postconditions.

        Returns:
            Can the condition be used as a postcondition?
        """
        return True

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        post_cond_part = "'" if is_post_cond else ""
        return "({}{} = {})".format(
            self._sf.get_name(), post_cond_part, self._sf.get_idx(self._value)
        )

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        We create a variable name for the condition <sf_name>EQ<value>.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        var_name = "{}EQ{}".format(self._sf.get_name(), self._value)
        return expr(var_name), {var_name: self}

    def range_of_values(self):
        """Return the one value that satisfies this condition.

        Returns:
            The range of values (the one value) that satisfies the condition
        """
        return [{self._sf: [self._value]}]

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            self._hash_val = hash((type(self), self._sf, self._value))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        return (
            type(self) == type(other)
            and self._sf == other._sf
            and self._value == other._value
        )


class NeqCondition(Condition):
    """A condition which checks if a state factor is not equal to a value.

    Attributes:
        _sf: The state factor we're checking
        _value: The value to check against
        _hash_val: The cached hash
    """

    def __init__(self, sf, value):
        """Initialises attributes.

        Args:
            sf: The state factor
            value: The state factor value to check

        Raises:
            Raised if value is invalid for sf
        """

        if not sf.is_valid_value(value):
            raise Exception("NeqCondition: value is an invalid value for state factor")

        self._sf = sf
        self._value = value
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state matches _value.

        Args:
            state: The state to check
            prev_state: Not used here

        Returns:
            Is the condition satisfied?

        Raises:
            invalid_value: Raised if state has an invalid value for _sf
        """

        sf_name = self._sf.get_name()
        if not self._sf.is_valid_value(state[sf_name]):
            raise Exception(
                "NeqCondition: state has an invalid value for {}".format(sf_name)
            )

        return state[sf_name] != self._value

    def is_pre_cond(self):
        """NeqConditions are valid preconditions.
        Returns:
            Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """NeqConditions are not valid postconditions.

        Returns:
            Can the condition be used as a postcondition?
        """
        return False

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        post_cond_part = "'" if is_post_cond else ""
        return "({}{} != {})".format(
            self._sf.get_name(), post_cond_part, self._sf.get_idx(self._value)
        )

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        We don't want to create variables like sfNEQval here, as it might
        get confused as being different to Not(EqCondition), which is equivalent.
        Therefore, we create a variable for the flipped EQCondition and use
        a not operator.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        eq_expr, var_map = EqCondition(self._sf, self._value).to_pyeda_expr()
        return Not(eq_expr), var_map

    def range_of_values(self):
        """Returns the state factor values without self._value.

        Returns:
            The range of values that satisfy the condition.
        """

        sf_values = set(self._sf.get_valid_values())
        sf_values.difference_update(set([self._value]))
        return [{self._sf: list(sf_values)}]

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            self._hash_val = hash((type(self), self._sf, self._value))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        return (
            type(self) == type(other)
            and self._sf == other._sf
            and self._value == other._value
        )


class NotCondition(Condition):
    """A condition which negates another condition.

    Attributes:
        _cond: The condition we're negating
        _hash_val: The cached hash
    """

    def __init__(self, cond):
        """Initialises attributes.

        Args:
            cond: The condition we want to negate

        Raises:
            invalid_cond: Raised if cond is not a valid precondition
        """

        if not cond.is_pre_cond():
            raise Exception("NotCondition: Condition is not a valid precondition")

        self._cond = cond
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Flips the return value of self._cond

        Args:
            state: The state to check
            prev_state: Not used here

        Returns:
            Is the condition satisfied?
        """
        return not self._cond.is_satisfied(state, prev_state=prev_state)

    def is_pre_cond(self):
        """NotConditions are valid preconditions.
        Returns:
            Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """NotConditions are not valid postconditions.

        Returns:
            Can the condition be used as a postcondition?
        """
        return False

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?
        """
        if is_post_cond:
            raise Exception("NotConditions cannot be postconditions")

        return "!{}".format(self._cond.to_prism_string())

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        This just negates the condition included in the object.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        cond_expr, var_map = self._cond.to_pyeda_expr()
        return Not(cond_expr), var_map

    def range_of_values(self):
        """Raises NotImplementedError as not needed currently."""
        raise NotImplementedError("range_of_values not implemented in NotCondition.")

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            self._hash_val = hash((type(self), self._cond))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        return type(self) == type(other) and self._cond == other._cond


class AddCondition(Condition):
    """Condition for adding a value to a state factor.

    Only valid for IntStateFactors.

    Attributes:
        Same as superclass, plus:
        _sf: The state factor
        _inc_value: The increment value
        _hash_val: The cached hash
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
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state = prev_state + _inc_value

        Args:
            state: The state to check
            prev_state: The previous state

        Returns:
            Is the condition satisfied?

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
            Can the condition be used as a precondition?
        """
        return False

    def is_post_cond(self):
        """AdsConditions are valid postconditions.

        Returns:
            Can the condition be used as a postcondition?
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
            self._sf.get_name(), self._sf.get_name(), self._inc_value
        )

    def to_pyeda_expr(self):
        """Throws exception as AddConditions are only postconditions."""
        raise Exception("No pyeda expression for AddCondition as postcondition.")

    def range_of_values(self):
        """Raises Exception as not applicable to AddConditions.

        Raises:
            not_for_post_cond: Raised as function doesn't apply to postconditions"""
        raise Exception("range_of_values can't be used for postconditions.")

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string(True)

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string(True)

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            self._hash_val = hash((type(self), self._sf, self._inc_value))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        return (
            type(self) == type(other)
            and self._sf == other._sf
            and self._inc_value == other._inc_value
        )


class InequalityCondition(Condition):
    """A precondition which compares a state factor to a value.

    EqCondition is not included here, as it can be a postcondition.
    This class can (and will) be used for <, >, <=, >=

    Attributes:
        _sf: The state factor we're checking
        _comp_fn: The int x int -> bool comparison function
        _comp_str: The PRISM symbol for the comparison operation
        _value: The value to check against
        _hash_val: The cached hash
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
            not_int: Raised if sf is not an IntStateFactor
        """

        if not isinstance(sf, IntStateFactor):
            raise Exception("InequalityCondition must use IntStateFactor")

        if not sf.is_valid_value(value):
            raise Exception(
                "InequalityCondition: value is an invalid value for state factor"
            )

        self._sf = sf
        self._value = value
        self._comp_fn = comp_fn
        self._comp_str = comp_str
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Checks if the value of _sf in state matches _value.

        Args:
            state: The state to check
            prev_state: Not used here

        Returns:
            Is the condition satisfied?

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
        """InequalityConditions are valid preconditions.
        Returns:
            Can the condition be used as a precondition?
        """
        return True

    def is_post_cond(self):
        """InequalityConditions are valid postconditions.

        Returns:
            Can the condition be used as a postcondition?
        """
        return False

    def to_prism_string(self, is_post_cond=False):
        """Outputs the prism string for this condition.

        Args:
            is_post_cond: Should the condition be written as a postcondition?

        Raises:
            post_cond_exception: Raised if is_post_cond is True
        """

        if is_post_cond:
            raise Exception("InequalityCondition cannot be postcondition.")

        return "({} {} {})".format(
            self._sf.get_name(), self._comp_str, self._sf.get_idx(self._value)
        )

    def range_of_values(self):
        """Returns the values which satisfy the inequality.

        Returns:
            The range of values which satisfy this inequality
        """
        sf_vals = self._sf.get_valid_values()

        satisfying = filter(lambda v: self._comp_fn(v, self._value), sf_vals)
        return [{self._sf: list(satisfying)}]

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            self._hash_val = hash((type(self), self._sf, self._value))
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        return (
            type(self) == type(other)
            and self._sf == other._sf
            and self._value == other._value
        )


class LtCondition(InequalityCondition):
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

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        We need to be careful with inequalities when minimising.
        Following from https://github.com/SimonaGug/BT-from-planning-experts
        < is represented logically as Not(>=).
        We have a variable for the >= as well, not the <
        We can then convert this back after doing all the logical work we need.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition
        """
        geq_cond = GeqCondition(self._sf, self._value)
        geq_expr, var_map = geq_cond.to_pyeda_expr()
        return Not(geq_expr), var_map


class GtCondition(InequalityCondition):
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

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        > are represented just as >. Its < and <= which are flipped in terms of
        variables.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        var_name = "{}GT{}".format(self._sf.get_name(), self._value)
        return expr(var_name), {var_name: self}


class LeqCondition(InequalityCondition):
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

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        We need to be careful with inequalities when minimising.
        Following from https://github.com/SimonaGug/BT-from-planning-experts
        <= is represented logically as Not(>).
        We have a variable for the > as well, not the <=
        We can then convert this back after doing all the logical work we need.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        gt_cond = GtCondition(self._sf, self._value)
        gt_expr, var_map = gt_cond.to_pyeda_expr()
        return Not(gt_expr), var_map


class GeqCondition(InequalityCondition):
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
        super(GeqCondition, self).__init__(sf, value, lambda x, y: x >= y, ">=")

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        >= are represented just as >=. Its < and <= which are flipped in terms of
        variables.

        Returns:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        var_name = "{}GEQ{}".format(self._sf.get_name(), self._value)
        return expr(var_name), {var_name: self}


class AndCondition(Condition):
    """Composite condition which captures conjunctions.

    Attributes:
        _cond_list: A list of conditions
        _hash_val: The cached hash
    """

    def __init__(self, *conds):
        """Initialise attributes.

        Args:
            *conds: The conditions to combine
        """
        self._cond_list = []
        for cond in conds:
            self._cond_list.append(cond)
        self._hash_val = None

    def add_cond(self, cond):
        """Add a new condition to the conjunction.

        Args:
            cond: The new condition
        """
        self._cond_list.append(cond)
        self._hash_val = None  # Needs recomputation

    def is_satisfied(self, state, prev_state=None):
        """Check if conjunction is satisfied.

        Args:
            state: The current state to check
            prev_state: The previous statem (if needed)

        Returns:
            True if condition satisfied, else False
        """

        for cond in self._cond_list:
            if not cond.is_satisfied(state, prev_state):
                return False
        return True

    def is_pre_cond(self):
        """Conjunction is pre cond if all conditions are preconditions.

        Returns:
            True if all conditions are preconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_pre_cond():
                return False

        return True

    def is_post_cond(self):
        """Conjunction is post cond if all conditions are postconditions.

        Returns:
            True if all conditions are postconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_post_cond():
                return False

        return True

    def to_prism_string(self, is_post_cond=False):
        """Output condition into prism string format.

        Args:
            is_post_cond: Is the condition a post condition?"""

        prism_str = ""

        if not is_post_cond:
            prism_str += "("

        for i in range(len(self._cond_list)):
            prism_str += self._cond_list[i].to_prism_string(is_post_cond)
            if i < len(self._cond_list) - 1:
                prism_str += " & "

        if not is_post_cond:
            prism_str += ")"

        return prism_str

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        Here we just do an And() of all sub-conditions, and combine the variable maps.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        expr_list = []
        var_map = {}

        for cond in self._cond_list:
            expr, sub_var_map = cond.to_pyeda_expr()
            expr_list.append(expr)
            for var in sub_var_map:
                if var not in var_map:  # Add in new variables
                    var_map[var] = sub_var_map[var]
                else:  # A small test that they are the same condition
                    assert var_map[var] == sub_var_map[var]

        return And(*expr_list), var_map

    def range_of_values(self):
        """Raises NotImplementedError as not needed currently."""
        raise NotImplementedError("range_of_values not implemented in AndCondition.")

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string(self.is_post_cond())

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            hash_sorter = lambda c: hash(c)
            self._hash_val = hash(
                (type(self), tuple(sorted(self._cond_list, key=hash_sorter)))
            )
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        first_part = type(self) == type(other) and len(self._cond_list) == len(
            other._cond_list
        )

        if not first_part:
            return False

        for cond in self._cond_list:
            if cond not in other._cond_list:
                return False

        return True


class OrCondition(Condition):
    """Composite condition which captures disjunctions.

    Attributes:
        _cond_list: A list of conditions
        _hash_val: The cached hash
    """

    def __init__(self, *conds):
        """Initialise attributes.

        Args:
            *conds: The conditions to combine
        """
        self._cond_list = []
        for cond in conds:
            self._cond_list.append(cond)
        self._hash_val = None

    def add_cond(self, cond):
        """Add a new condition to the conjunction.

        Args:
            cond: The new condition
        """
        self._cond_list.append(cond)
        self._hash_val = None

    def is_satisfied(self, state, prev_state=None):
        """Check if disjunction is satisfied.

        Args:
            state: The current state to check
            prev_state: The previous statem (if needed)

        Returns:
            True if condition satisfied, else False
        """

        for cond in self._cond_list:
            if cond.is_satisfied(state, prev_state):
                return True
        return False

    def is_pre_cond(self):
        """Disjunction is pre cond if all conditions are preconditions.

        Returns:
            True if all conditions are preconditions, else False
        """
        for cond in self._cond_list:
            if not cond.is_pre_cond():
                return False

        return True

    def is_post_cond(self):
        """Disjunction cannot be a postcondition.

        Returns:
            False for disjunctions
        """
        return False

    def to_prism_string(self, is_post_cond=False):
        """Output condition into prism string format.

        Args:
            is_post_cond: Is the condition a post condition?"""

        if is_post_cond:
            raise Exception("OrCondition cannot be a postcondition.")

        prism_str = "("

        for i in range(len(self._cond_list)):
            prism_str += self._cond_list[i].to_prism_string(is_post_cond)
            if i < len(self._cond_list) - 1:
                prism_str += " | "

        prism_str += ")"

        return prism_str

    def to_pyeda_expr(self):
        """Converts the condition into a pyeda logical expression.

        Here we just do an Or() of all sub-conditions, and combine the variable maps.

        Returns:
            A tuple containing:
            - The corresponding pyeda expression
            - A mapping from var_name to condition.
        """
        expr_list = []
        var_map = {}

        for cond in self._cond_list:
            expr, sub_var_map = cond.to_pyeda_expr()
            expr_list.append(expr)
            for var in sub_var_map:
                if var not in var_map:  # Add in new variables
                    var_map[var] = sub_var_map[var]
                else:  # A small test that they are the same condition
                    assert var_map[var] == sub_var_map[var]

        return Or(*expr_list), var_map

    def range_of_values(self):
        """Raises NotImplementedError as not needed currently."""
        raise NotImplementedError("range_of_values not implemented in OrCondition.")

    def __repr__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __str__(self):
        """Make the condition human readable.

        Returns:
            A str representation of the label
        """
        return self.to_prism_string()

    def __hash__(self):
        """Overwriting for use in dictionaries etc.

        Returns:
            The hash
        """
        if self._hash_val is None:
            hash_sorter = lambda c: hash(c)
            self._hash_val = hash(
                (type(self), tuple(sorted(self._cond_list, key=hash_sorter)))
            )
        return self._hash_val

    def __eq__(self, other):
        """Overwriting so conditions with identical information are equal.

        Args:
            other: The other condition

        Returns:
            Are the conditions equal?
        """
        first_part = type(self) == type(other) and len(self._cond_list) == len(
            other._cond_list
        )

        if not first_part:
            return False

        for cond in self._cond_list:
            if cond not in other._cond_list:
                return False

        return True
