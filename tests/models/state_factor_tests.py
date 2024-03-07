#!/usr/bin/env python3
""" Unit tests for state_factor.py

Author: Charlie Street
Owner: Charlie Street
"""

from refine_plan.models.state_factor import StateFactor, BoolStateFactor, IntStateFactor
import unittest


class StateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = StateFactor("sf", ["a", "b", "c"])

        self.assertEqual(sf.get_value(0), "a")
        self.assertEqual(sf.get_value(1), "b")
        self.assertEqual(sf.get_value(2), "c")
        with self.assertRaises(Exception):
            sf.get_value(3)

        self.assertEqual(sf.get_idx("a"), 0)
        self.assertEqual(sf.get_idx("b"), 1)
        self.assertEqual(sf.get_idx("c"), 2)
        with self.assertRaises(Exception):
            sf.get_idx("d")

        self.assertTrue(sf.is_valid_value("a"))
        self.assertTrue(sf.is_valid_value("b"))
        self.assertTrue(sf.is_valid_value("c"))
        self.assertFalse(sf.is_valid_value("d"))
        self.assertFalse(sf.is_valid_value(1))

        self.assertEqual(sf.get_valid_values(), ["a", "b", "c"])
        self.assertEqual(sf.get_name(), "sf")


class BoolStateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = BoolStateFactor("bool_sf")

        self.assertEqual(sf.get_value(0), False)
        self.assertEqual(sf.get_value(1), True)
        with self.assertRaises(Exception):
            sf.get_value(2)

        self.assertEqual(sf.get_idx(False), 0)
        self.assertEqual(sf.get_idx(True), 1)
        with self.assertRaises(Exception):
            sf.get_idx(2)

        self.assertTrue(sf.is_valid_value(False))
        self.assertTrue(sf.is_valid_value(True))
        self.assertFalse(sf.is_valid_value("d"))

        self.assertEqual(sf.get_valid_values(), [False, True])
        self.assertEqual(sf.get_name(), "bool_sf")


class IntStateFactorTest(unittest.TestCase):

    def test_function(self):
        sf = IntStateFactor("int_sf", 5, 10)

        for i in range(5, 11):
            self.assertEqual(sf.get_value(i), i)
            self.assertEqual(sf.get_idx(i), i)

        with self.assertRaises(Exception):
            sf.get_value(2)

        with self.assertRaises(Exception):
            sf.get_idx(11)

        self.assertTrue(sf.is_valid_value(5))
        self.assertTrue(sf.is_valid_value(6))
        self.assertTrue(sf.is_valid_value(7))
        self.assertTrue(sf.is_valid_value(8))
        self.assertTrue(sf.is_valid_value(9))
        self.assertTrue(sf.is_valid_value(10))
        self.assertFalse(sf.is_valid_value(4))
        self.assertFalse(sf.is_valid_value(11))

        self.assertEqual(sf.get_valid_values(), [5, 6, 7, 8, 9, 10])
        self.assertEqual(sf.get_name(), "int_sf")


if __name__ == "__main__":
    unittest.main()
