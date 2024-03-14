#!/usr/bin/env python3
""" Classes for representing state factors in Markov models.

Author: Charlie Street
Owner: Charlie Street
"""


class StateFactor(object):
    """Class representing a generic state factor.

    The state factor values are converted into list indexes when writing out
    in the PRISM language.

    Attributes:
        _name: The name of the state factor
        _values: The valid values for that state factor
    """

    def __init__(self, name, values):
        """Initialise attributes.

        Args:
            name: The state factor name
            values: The valid state factor values
        """
        self._name = name
        self._values = values

    def get_value(self, idx):
        """Get the state factor value at a given index.

        Args:
            idx: The index

        Returns:
            value: The corresponding state factor value

        Throws:
            _bad_idx_exception: Thrown if an invalid index provided
        """

        if idx not in range(len(self._values)):
            raise Exception("Invalid index given to StateFactor::get_value")

        return self._values[idx]

    def get_idx(self, value):
        """Get the index for a state factor value.

        Args:
            value: The state factor value

        Returns:
            idx: The corresponding index

        Throws:
            _bad_val_exception: Raised if an invalid state factor value is given
        """

        if not self.is_valid_value(value):
            raise Exception("Invalid value given to StateFactor::get_idx")

        return self._values.index(value)

    def is_valid_value(self, value):
        """Test if a value is valid for the state factor.

        Args:
            value: The value to test

        Returns:
            is_valid: True if in self._values, False otherwise
        """

        return value in self._values

    def get_valid_values(self):
        """Return the valid list of values for the state factor.

        Returns:
            values: The state factor values
        """
        return self._values

    def get_name(self):
        """Return the name of the state factor.

        Returns:
            name: The state factor name
        """

        return self._name

    def to_prism_string(self, initial_value=None):
        """Write the state factor out in PRISM format.

        Args:
            initial_value: The initial value of that state factor, if applicable.

        Returns:
            prism_str: The PRISM string for this state factor

        Raises:
            bad_init_exception: Raised if initial_value is not valid value
        """

        prism_str = "{}: [{}..{}]".format(
            self.get_name(),
            self.get_idx(self._values[0]),
            self.get_idx(self._values[-1]),
        )

        if initial_value is not None:
            if not self.is_valid_value(initial_value):
                raise Exception(
                    "{} is invalid value for {}".format(initial_value, self.get_name)
                )

            prism_str += " init {}".format(self.get_idx(initial_value))

        prism_str += ";\n"

        return prism_str


class BoolStateFactor(StateFactor):
    """A subclass of StateFactor which can only take values True and False.

    Attributes:
        Same as superclass.
    """

    def __init__(self, name):
        """Initialise attributes.

        Here, False will be at index 0, and True will be at index 1.

        Args:
            name: The state factor's name
        """
        super(BoolStateFactor, self).__init__(name, [False, True])


class IntStateFactor(StateFactor):
    """A subclass of StateFactor purely for integer state factors.

    Here, the values and indexes are the same thing.

    Attributes:
        Same as superclass.
    """

    def __init__(self, name, min, max):
        """Initialise attributes.

        Args:
            name: The state factor name
            min: The minimum of the state factor (inclusive)
            max: The maximum of the state factor (inclusive)
        """
        super(IntStateFactor, self).__init__(name, list(range(min, max + 1)))

    def get_value(self, idx):
        """Identity function as indexes and values are the same here.

        Args:
            idx: The index

        Returns:
            value: The corresponding state factor value

        Throws:
            _bad_idx_exception: Thrown if an invalid index provided
        """

        if not self.is_valid_value(idx):
            raise Exception("Invalid index given to IntStateFactor::get_value")

        return idx

    def get_idx(self, value):
        """Identity function as indexes and values are the same here.

        Args:
            value: The state factor value

        Returns:
            idx: The corresponding index

        Throws:
            _bad_val_exception: Raised if an invalid state factor value is given
        """

        if not self.is_valid_value(value):
            raise Exception("Invalid value given to IntStateFactor::get_idx")

        return value
